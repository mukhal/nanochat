"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from typing import List, Optional
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.block_manager import BlockManager, Sequence, SamplingParams, BLOCK_SIZE
from nanochat.engine_standard import KVCache, Engine
from contextlib import nullcontext

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
## Paged Attention Engine

class PagedKVCache:
    """
    Physical storage for KV cache blocks on GPU.
    
    This stores the actual K/V tensors in a block-organized layout.
    The BlockManager handles logical allocation; this class handles physical storage.
    
    Memory layout: (num_blocks, num_layers, 2, block_size, num_heads, head_dim)
    - num_blocks: total physical blocks available
    - num_layers: transformer layers
    - 2: K and V
    - block_size: tokens per block
    - num_heads: number of KV heads
    - head_dim: dimension per head
    """
    
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim, device, dtype):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # Main cache storage: (num_blocks, num_layers, 2, block_size, num_heads, head_dim)
        self.k_cache = torch.zeros(
            num_blocks, num_layers, block_size, num_heads, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            num_blocks, num_layers, block_size, num_heads, head_dim,
            device=device, dtype=dtype
        )
        
    def reset(self):
        """Reset cache to empty state (zero out all blocks)."""
        self.k_cache.zero_()
        self.v_cache.zero_()
    
    def get_layer_cache(self, layer_idx, block_ids):
        """
        Get K/V cache for specific blocks at a given layer.
        
        Args:
            layer_idx: which transformer layer
            block_ids: list of physical block IDs to fetch
            
        Returns:
            k: (num_blocks, block_size, num_heads, head_dim)
            v: (num_blocks, block_size, num_heads, head_dim)
        """
        k = self.k_cache[block_ids, layer_idx]  # (num_blocks, block_size, num_heads, head_dim)
        v = self.v_cache[block_ids, layer_idx]
        return k, v
    
    def write_layer_cache(self, layer_idx, block_id, slot_idx, k, v):
        """
        Write K/V values to a specific slot within a block.
        
        Args:
            layer_idx: which transformer layer
            block_id: physical block ID
            slot_idx: position within the block (0 to block_size-1)
            k: (num_heads, head_dim) or (seq_len, num_heads, head_dim)
            v: (num_heads, head_dim) or (seq_len, num_heads, head_dim)
        """
        if k.dim() == 2:
            # Single token: (num_heads, head_dim)
            self.k_cache[block_id, layer_idx, slot_idx] = k
            self.v_cache[block_id, layer_idx, slot_idx] = v
        else:
            # Multiple tokens: (seq_len, num_heads, head_dim)
            seq_len = k.size(0)
            self.k_cache[block_id, layer_idx, slot_idx:slot_idx + seq_len] = k
            self.v_cache[block_id, layer_idx, slot_idx:slot_idx + seq_len] = v
    
    def gather_kv_for_sequence(self, layer_idx, block_table, seq_len):
        """
        Gather K/V cache for a sequence into contiguous tensors.
        
        This is the "gather then attend" approach - less efficient than true
        paged attention kernels, but works with standard Flash Attention.
        
        #TODO: use dedicated kernels
        
        Args:
            layer_idx: which transformer layer
            block_table: list of physical block IDs for this sequence
            seq_len: total number of tokens to gather
            
        Returns:
            k: (1, seq_len, num_heads, head_dim) - contiguous K cache
            v: (1, seq_len, num_heads, head_dim) - contiguous V cache
        """
        num_full_blocks = seq_len // self.block_size
        remainder = seq_len % self.block_size
        
        k_parts = []
        v_parts = []
        
        # Gather full blocks
        for i in range(num_full_blocks):
            block_id = block_table[i]
            k_parts.append(self.k_cache[block_id, layer_idx])  # (block_size, num_heads, head_dim)
            v_parts.append(self.v_cache[block_id, layer_idx])
        
        # Gather partial last block if needed
        if remainder > 0:
            block_id = block_table[num_full_blocks]
            k_parts.append(self.k_cache[block_id, layer_idx, :remainder])
            v_parts.append(self.v_cache[block_id, layer_idx, :remainder])
        
        # Concatenate into contiguous tensors
        k = torch.cat(k_parts, dim=0).unsqueeze(0)  # (1, seq_len, num_heads, head_dim)
        v = torch.cat(v_parts, dim=0).unsqueeze(0)
        
        return k, v
    
    def memory_usage_bytes(self):
        """Return total memory used by the cache in bytes."""
        return self.k_cache.numel() * self.k_cache.element_size() * 2  # *2 for K and V


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


class SequenceKVView:
    """
    Per-sequence view into the shared PagedKVCache.
    
    This is a lightweight wrapper that provides the same interface as KVCache
    but reads/writes to scattered blocks in the PagedKVCache based on the
    sequence's block_table. 
    
    This is done so we don't have to modify gpt.py too much. Basically, the model's attention code doesn't need to know about paging - it just calls get_layer_cache() like before.
    """
    
    def __init__(self, paged_cache: PagedKVCache, block_table: List[int], seq_len: int, block_size: int):
        self.paged_cache = paged_cache
        self.block_table = block_table
        self.seq_len = seq_len  # Total tokens in sequence (prompt + generated)
        self.cached_len = 0     # Tokens actually written to cache so far
        self.block_size = block_size
        self.n_layers = paged_cache.num_layers
        self.is_paged = True  # Flag for attention code to detect paged cache
        # cache_seqlens tracks what's in the cache (for compatibility)
        self.cache_seqlens = torch.tensor([0], dtype=torch.int32, device=paged_cache.device)
    
    def get_pos(self):
        """Get current position (number of tokens cached)."""
        return self.cached_len
    
    def get_layer_cache(self, layer_idx):
        """
        Return (k_cache, v_cache) for a specific layer.
        
        Gathers scattered blocks into contiguous tensors that Flash Attention expects.
        Returns shape (1, cached_len, num_heads, head_dim) for both K and V.
        Only returns tokens that have actually been written to the cache.
        """
        if self.cached_len == 0:
            # No cached tokens yet - return empty tensors
            device = self.paged_cache.device
            dtype = self.paged_cache.dtype
            return (
                torch.empty(1, 0, self.paged_cache.num_heads, self.paged_cache.head_dim, device=device, dtype=dtype),
                torch.empty(1, 0, self.paged_cache.num_heads, self.paged_cache.head_dim, device=device, dtype=dtype),
            )
        return self.paged_cache.gather_kv_for_sequence(
            layer_idx, self.block_table, self.cached_len
        )
    
    def advance(self, num_tokens):
        """Advance the cached length by num_tokens (called after write_kv)."""
        self.cached_len += num_tokens
        self.cache_seqlens[0] = self.cached_len
    
    def write_kv(self, layer_idx, k, v):
        """
        Write new K/V values to the cache at the current cached_len position.
        
        Args:
            layer_idx: which transformer layer
            k: (batch=1, num_new_tokens, num_heads, head_dim)
            v: (batch=1, num_new_tokens, num_heads, head_dim)
        """
        # k, v are (1, T, H, D) - remove batch dim
        k = k.squeeze(0)  # (T, H, D)
        v = v.squeeze(0)
        
        num_new_tokens = k.size(0)
        start_pos = self.cached_len  # Write at current cache position
        
        for i in range(num_new_tokens):
            pos = start_pos + i
            block_idx = pos // self.block_size
            slot_idx = pos % self.block_size
            block_id = self.block_table[block_idx]
            
            self.paged_cache.k_cache[block_id, layer_idx, slot_idx] = k[i]
            self.paged_cache.v_cache[block_id, layer_idx, slot_idx] = v[i]


# -----------------------------------------------------------------------------
# PagedEngine: Engine with paged attention support

class PagedEngine:
    """
    Inference engine using paged attention for memory-efficient KV caching.
    
    Unlike the standard Engine which allocates contiguous KV cache per request,
    PagedEngine uses a shared PagedKVCache with block-based allocation.
    """
    
    def __init__(self, model, tokenizer, num_blocks: int = 1000, block_size: int = BLOCK_SIZE):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Get model config
        m = model.config
        device = model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        
        # Create shared resources
        self.paged_kv_cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=m.n_layer,
            num_heads=m.n_kv_head,
            head_dim=m.n_embd // m.n_head,
            device=device,
            dtype=dtype,
        )
        self.block_manager = BlockManager(num_blocks, block_size)
        
    @torch.inference_mode()
    def generate(self, tokens: List[int], max_tokens: Optional[int] = None, 
                 temperature: float = 1.0, top_k: Optional[int] = None, seed: int = 42):
        """
        Generate tokens using paged attention.
        
        Yields (token, mask) tuples where mask=1 for sampled tokens, mask=0 for forced tokens.
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        
        # Get special tokens for tool use
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        
        # Create sequence and allocate blocks
        seq = Sequence(tokens, SamplingParams(temperature=temperature, max_tokens=max_tokens or 256, top_k=top_k))
        self.block_manager.allocate(seq)
        
        # Create KV view for this sequence
        seq_view = SequenceKVView(
            self.paged_kv_cache,
            seq.block_table,
            seq.num_tokens,
            self.block_size,
        )
        
        # Tool use state
        forced_tokens = deque()
        in_python_block = False
        python_expr_tokens = []
        
        try:
            # Prefill: process all prompt tokens at once
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = self.model.forward(ids, kv_cache=seq_view)
            logits = logits[:, -1, :]  # (1, vocab_size)
            
            # Decode loop
            num_generated = 0
            while True:
                # Stop conditions
                if max_tokens is not None and num_generated >= max_tokens:
                    break
                
                # Sample or use forced token
                is_forced = len(forced_tokens) > 0
                if is_forced:
                    next_token = forced_tokens.popleft()
                else:
                    next_ids = sample_next_token(logits, rng, temperature, top_k)
                    next_token = next_ids[0, 0].item()
                
                # Check for completion
                if next_token == assistant_end or next_token == bos:
                    break
                
                # Update sequence
                seq.append_token(next_token)
                
                # Allocate new block if needed (when current block is full)
                if seq.num_tokens > len(seq.block_table) * self.block_size:
                    new_block = self.block_manager._allocate_block()
                    seq.block_table.append(new_block.block_id)
                
                # Handle tool logic
                if next_token == python_start:
                    in_python_block = True
                    python_expr_tokens = []
                elif next_token == python_end and in_python_block:
                    in_python_block = False
                    if python_expr_tokens:
                        expr = self.tokenizer.decode(python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            forced_tokens.append(output_start)
                            forced_tokens.extend(result_tokens)
                            forced_tokens.append(output_end)
                    python_expr_tokens = []
                elif in_python_block:
                    python_expr_tokens.append(next_token)
                
                # Yield token
                yield next_token, (0 if is_forced else 1)
                num_generated += 1
                
                # Forward pass for next token
                next_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
                logits = self.model.forward(next_ids, kv_cache=seq_view)[:, -1, :]
                
        finally:
            # Always deallocate blocks when done
            self.block_manager.deallocate(seq)
    
    def generate_batch(self, tokens: List[int], **kwargs):
        """
        Non-streaming generation that returns the final token sequence.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        
        result = tokens.copy()
        masks = [0] * len(tokens)
        
        for token, mask in self.generate(tokens, **kwargs):
            if token == assistant_end or token == bos:
                break
            result.append(token)
            masks.append(mask)
        
        return result, masks


# -----------------------------------------------------------------------------
# Scheduler and LLM: vLLM-style batched inference
# -----------------------------------------------------------------------------

class Scheduler:
    """
    Manages sequence batching and memory allocation for efficient inference.
    
    The scheduler maintains two queues:
    - waiting: sequences that need prefill (haven't started generating)
    - running: sequences in decode phase (generating tokens one at a time)
    
    Each step, it decides whether to do prefill (process waiting sequences)
    or decode (generate next token for running sequences).
    """
    
    def __init__(self, block_manager: BlockManager, max_batch_size: int = 32):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
    
    def add(self, seq: Sequence):
        """Add a new sequence to the waiting queue."""
        self.waiting.append(seq)
    
    def schedule(self) -> tuple:
        """
        Decide which sequences to run this step.
        
        Returns:
            (sequences, is_prefill): List of sequences to process and whether it's prefill
        """
        # Priority: prefill waiting sequences first (they're blocking)
        if self.waiting:
            # Take sequences from waiting queue up to max_batch_size
            batch = []
            while self.waiting and len(batch) < self.max_batch_size:
                seq = self.waiting.pop(0)
                # Try to allocate blocks for this sequence
                try:
                    self.block_manager.allocate(seq)
                    batch.append(seq)
                except RuntimeError:
                    # Out of blocks - put back in waiting and stop
                    self.waiting.insert(0, seq)
                    break
            
            if batch:
                return batch, True  # is_prefill=True
        
        # No waiting sequences (or couldn't allocate) - decode running sequences
        if self.running:
            # Return all running sequences for decode
            return self.running, False  # is_prefill=False
        
        return [], False
    
    def update(self, seqs: List[Sequence], new_tokens: List[int], eos_token_id: int):
        """
        Update sequences after a forward pass.
        
        Args:
            seqs: Sequences that were processed
            new_tokens: New token for each sequence
            eos_token_id: Token ID that signals end of generation
        """
        for seq, token in zip(seqs, new_tokens):
            seq.append_token(token)
            
            # Check completion conditions
            if token == eos_token_id:
                seq.status = SequenceStatus.FINISHED
            elif seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
            
            # Allocate new block if needed
            if not seq.is_finished:
                blocks_needed = seq.num_blocks
                if blocks_needed > len(seq.block_table):
                    try:
                        new_block = self.block_manager._allocate_block()
                        seq.block_table.append(new_block.block_id)
                    except RuntimeError:
                        # Out of blocks - mark as finished (truncated)
                        seq.status = SequenceStatus.FINISHED
    
    def finish_prefill(self, seqs: List[Sequence]):
        """Move sequences from prefill to running state."""
        for seq in seqs:
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
    
    def collect_finished(self) -> List[Sequence]:
        """Remove and return finished sequences, deallocating their blocks."""
        finished = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        
        for seq in finished:
            self.block_manager.deallocate(seq)
        
        return finished
    
    def is_finished(self) -> bool:
        """Check if all sequences have completed."""
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def num_waiting(self) -> int:
        return len(self.waiting)
    
    def num_running(self) -> int:
        return len(self.running)


class BatchedSequenceKVView:
    """
    KV cache view for a batch of sequences with different lengths.
    
    Handles padding to create uniform tensors for batched attention.
    """
    
    def __init__(self, paged_cache: PagedKVCache, sequences: List[Sequence], block_size: int):
        self.paged_cache = paged_cache
        self.sequences = sequences
        self.block_size = block_size
        self.n_layers = paged_cache.num_layers
        self.is_paged = True
        
        # Track cached length per sequence
        self.cached_lens = [0] * len(sequences)
        
        # For compatibility with model code
        self.cache_seqlens = torch.zeros(len(sequences), dtype=torch.int32, device=paged_cache.device)
    
    def get_pos(self) -> int:
        """Get position for rotary embeddings (use max cached length)."""
        return max(self.cached_lens) if self.cached_lens else 0
    
    def get_layer_cache(self, layer_idx: int):
        """
        Gather and pad K/V cache for all sequences in the batch.
        
        Returns:
            k: (batch_size, max_cached_len, num_heads, head_dim)
            v: (batch_size, max_cached_len, num_heads, head_dim)
        """
        if all(cl == 0 for cl in self.cached_lens):
            # No cached tokens yet
            device = self.paged_cache.device
            dtype = self.paged_cache.dtype
            batch_size = len(self.sequences)
            return (
                torch.empty(batch_size, 0, self.paged_cache.num_heads, self.paged_cache.head_dim, device=device, dtype=dtype),
                torch.empty(batch_size, 0, self.paged_cache.num_heads, self.paged_cache.head_dim, device=device, dtype=dtype),
            )
        
        max_cached = max(self.cached_lens)
        batch_size = len(self.sequences)
        device = self.paged_cache.device
        dtype = self.paged_cache.dtype
        
        # Pre-allocate padded tensors
        k_batch = torch.zeros(batch_size, max_cached, self.paged_cache.num_heads, self.paged_cache.head_dim, device=device, dtype=dtype)
        v_batch = torch.zeros(batch_size, max_cached, self.paged_cache.num_heads, self.paged_cache.head_dim, device=device, dtype=dtype)
        
        # Gather each sequence's cache
        for i, (seq, cached_len) in enumerate(zip(self.sequences, self.cached_lens)):
            if cached_len > 0:
                k, v = self.paged_cache.gather_kv_for_sequence(layer_idx, seq.block_table, cached_len)
                k_batch[i, :cached_len] = k.squeeze(0)
                v_batch[i, :cached_len] = v.squeeze(0)
        
        return k_batch, v_batch
    
    def advance(self, num_tokens: int):
        """Advance cached length for all sequences."""
        for i in range(len(self.cached_lens)):
            self.cached_lens[i] += num_tokens
        self.cache_seqlens[:] = torch.tensor(self.cached_lens, dtype=torch.int32)
    
    def write_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """
        Write K/V for all sequences in the batch.
        
        Args:
            k: (batch_size, num_new_tokens, num_heads, head_dim)
            v: (batch_size, num_new_tokens, num_heads, head_dim)
        """
        num_new_tokens = k.size(1)
        
        for i, (seq, cached_len) in enumerate(zip(self.sequences, self.cached_lens)):
            k_seq = k[i]  # (num_new_tokens, num_heads, head_dim)
            v_seq = v[i]
            
            for j in range(num_new_tokens):
                pos = cached_len + j
                block_idx = pos // self.block_size
                slot_idx = pos % self.block_size
                block_id = seq.block_table[block_idx]
                
                self.paged_cache.k_cache[block_id, layer_idx, slot_idx] = k_seq[j]
                self.paged_cache.v_cache[block_id, layer_idx, slot_idx] = v_seq[j]


class LLM:
    """
    vLLM-style interface for batched LLM inference.
    
    Example:
        llm = LLM(model_path="/path/to/model")
        outputs = llm.generate(
            ["Hello, how are you?", "What is 2+2?"],
            sampling_params=SamplingParams(temperature=0.7, max_tokens=100)
        )
        for output in outputs:
            print(output["text"])
    """
    
    def __init__(
        self,
        model_path: str,
        step: Optional[int] = None,
        device: str = "cuda",
        num_blocks: int = 2000,
        block_size: int = BLOCK_SIZE,
        max_batch_size: int = 32,
    ):
        """
        Initialize the LLM engine.
        
        Args:
            model_path: Path to the model checkpoint directory
            step: Checkpoint step to load (default: latest)
            device: Device to run on ("cuda" or "cpu")
            num_blocks: Number of KV cache blocks to allocate
            block_size: Tokens per block
            max_batch_size: Maximum sequences to batch together
        """
        from nanochat.checkpoint_manager import build_model, find_last_step
        
        # Load model and tokenizer
        self.device = torch.device(device)
        step = step if step is not None else find_last_step(model_path)
        self.model, self.tokenizer, _ = build_model(model_path, step, self.device, phase="eval")
        
        # Get model config
        m = self.model.config
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        
        # Create shared KV cache space across all sequences
        self.paged_kv_cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=m.n_layer,
            num_heads=m.n_kv_head,
            head_dim=m.n_embd // m.n_head,
            device=self.device,
            dtype=dtype,
        )
        
        # Create block manager and scheduler
        self.block_manager = BlockManager(num_blocks, block_size)
        self.scheduler = Scheduler(self.block_manager, max_batch_size)
        
        self.block_size = block_size
        self.eos_token_id = self.tokenizer.encode_special("<|assistant_end|>")
        
        # For autocast
        self.autocast_ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device == "cuda" else nullcontext()
    
    def add_request(self, prompt, sampling_params: Optional[SamplingParams] = None):
        """
        Add a generation request to the scheduler.
        
        Args:
            prompt: String or list of token IDs
            sampling_params: Generation parameters
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # Tokenize if needed
        if isinstance(prompt, str):
            bos = self.tokenizer.get_bos_token_id()
            tokens = self.tokenizer.encode(prompt, prepend=bos)
        else:
            tokens = prompt
        
        seq = Sequence(tokens, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id
    
    @torch.inference_mode()
    def step(self) -> tuple:
        """
        Execute one generation step.
        
        Returns:
            (finished_outputs, num_tokens):
                - finished_outputs: List of (seq_id, generated_tokens) for completed sequences
                - num_tokens: Positive for prefill tokens, negative for decode batch size
        """
        # Get sequences to process
        seqs, is_prefill = self.scheduler.schedule()
        
        if not seqs:
            return [], 0
        
        with self.autocast_ctx:
            if is_prefill:
                # Prefill: process full prompts
                outputs = self._run_prefill(seqs)
                self.scheduler.finish_prefill(seqs)
                num_tokens = sum(seq.num_prompt_tokens for seq in seqs)
            else:
                # Decode: generate one token per sequence
                outputs = self._run_decode(seqs)
                num_tokens = -len(seqs)
        
        # Collect finished sequences
        finished = self.scheduler.collect_finished()
        finished_outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished]
        
        return finished_outputs, num_tokens
    
    def _run_prefill(self, seqs: List[Sequence]) -> List[int]:
        """Run prefill for a batch of sequences."""
        # For simplicity, process one at a time (can optimize later with padding)
        new_tokens = []
        
        
        print(f"Running prefill for {len(seqs)} sequences")
        
        ### for now, we do one sequence at a time
        
        for seq in seqs:
            # Create KV view for this sequence
            kv_view = SequenceKVView(
                self.paged_kv_cache,
                seq.block_table,
                seq.num_tokens,
                self.block_size,
            )
            
            # Forward pass
            ids = torch.tensor([seq.token_ids], dtype=torch.long, device=self.device)
            logits = self.model.forward(ids, kv_cache=kv_view)
            logits = logits[:, -1, :]  # (1, vocab_size)
            
            # Sample next token
            rng = torch.Generator(device=self.device)
            rng.manual_seed(42 + seq.seq_id)  # Deterministic per sequence
            next_token = sample_next_token(logits, rng, seq.temperature, seq.top_k)
            new_tokens.append(next_token[0, 0].item())
            
            # Store the kv_view for decode phase
            seq._kv_view = kv_view
        
        # Update sequences with new tokens
        self.scheduler.update(seqs, new_tokens, self.eos_token_id)
        
        return new_tokens
    
    def _run_decode(self, seqs: List[Sequence]) -> List[int]:
        """Run decode for a batch of sequences (one token each)."""
        new_tokens = []
        
        # For simplicity, process one at a time (can optimize with batching later)
        for seq in seqs:
            kv_view = seq._kv_view
            
            # Forward pass with just the last token B=1, T=1
            last_token = seq.token_ids[-1]
            ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
            logits = self.model.forward(ids, kv_cache=kv_view)[:, -1, :]
            
            # Sample next token
            rng = torch.Generator(device=self.device)
            rng.manual_seed(42 + seq.seq_id + seq.num_tokens)
            next_token = sample_next_token(logits, rng, seq.temperature, seq.top_k)
            new_tokens.append(next_token[0, 0].item())
        
        # Update sequences
        self.scheduler.update(seqs, new_tokens, self.eos_token_id)
        
        return new_tokens
    
    def is_finished(self) -> bool:
        """Check if all requests have completed."""
        return self.scheduler.is_finished()
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[dict]:
        """
        Generate completions for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            sampling_params: Generation parameters (applied to all prompts)
            use_tqdm: Whether to show progress bar
            
        Returns:
            List of dicts with "text" and "token_ids" keys
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # Setup progress tracking
        if use_tqdm:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
            except ImportError:
                use_tqdm = False
        
        # Add all prompts to scheduler
        seq_ids = []
        for prompt in prompts:
            seq_id = self.add_request(prompt, sampling_params)
            seq_ids.append(seq_id)
        
        # Collect results
        outputs = {}
        
        # Main generation loop
        while not self.is_finished():
            finished_outputs, num_tokens = self.step()
            
            for seq_id, token_ids in finished_outputs:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        if use_tqdm:
            pbar.close()
        
        # Sort by seq_id to match input order and decode
        results = []
        for seq_id in seq_ids:
            token_ids = outputs.get(seq_id, [])
            text = self.tokenizer.decode(token_ids)
            results.append({"text": text, "token_ids": token_ids})
        
        return results


# Import SequenceStatus for scheduler
from nanochat.block_manager import SequenceStatus


# -----------------------------------------------------------------------------
# Legacy Engine (uses contiguous KVCache, supports num_samples > 1)
# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation


if __name__ == "__main__":
    """
    Test script for inference engines.
    
    Tests:
    1. model.generate() - naive reference implementation
    2. Engine.generate() - standard KV cache
    3. PagedEngine.generate() - paged attention
    
    All three should produce identical outputs with the same seed.
    """
    import argparse
    import time
    from nanochat.checkpoint_manager import build_model, find_last_step
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/data2/khalifam/fun/nanochat/models/nanochat-d34", 
                        help="Path to model directory")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--test", type=str, default="all", choices=["all", "paged"], help="Which test to run")
    args = parser.parse_args()

    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # load the model and tokenizer
    step = args.step if args.step is not None else find_last_step(args.model_dir)
    print(f"Loading model from {args.model_dir} at step {step}...")
    model, tokenizer, meta = build_model(args.model_dir, step, device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.7, seed=42)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("Hello, how are you?", prepend=bos_token_id)
    print(f"Prompt ({len(prompt_tokens)} tokens): {tokenizer.decode(prompt_tokens)}")
    print("=" * 60)

    if args.test == "all":
        # =========================================================================
        # Test 1: Reference implementation (model.generate)
        # =========================================================================
        print("\n[1] Testing model.generate() (reference)...")
        reference_tokens = []
        torch.cuda.synchronize() if device_type == "cuda" else None
        t0 = time.time()
        stream = model.generate(prompt_tokens, max_tokens=kwargs["max_tokens"], 
                               temperature=kwargs["temperature"], seed=kwargs["seed"])
        with autocast_ctx:
            for token in stream:
                reference_tokens.append(token)
                print(tokenizer.decode([token]), end="", flush=True)
        print()
        torch.cuda.synchronize() if device_type == "cuda" else None
        t1 = time.time()
        print(f"Reference: {len(reference_tokens)} tokens in {t1 - t0:.2f}s")

        # =========================================================================
        # Test 2: Standard Engine (contiguous KV cache)
        # =========================================================================
        print("\n[2] Testing Engine.generate() (standard KV cache)...")
        engine_tokens = []
        engine = Engine(model, tokenizer)
        torch.cuda.synchronize() if device_type == "cuda" else None
        t0 = time.time()
        with autocast_ctx:
            for token_column, token_masks in engine.generate(prompt_tokens, num_samples=1, **kwargs):
                token = token_column[0]
                engine_tokens.append(token)
                print(tokenizer.decode([token]), end="", flush=True)
        print()
        torch.cuda.synchronize() if device_type == "cuda" else None
        t1 = time.time()
        print(f"Engine: {len(engine_tokens)} tokens in {t1 - t0:.2f}s")
        
        # Compare
        match = reference_tokens == engine_tokens
        print(f"Engine vs Reference: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            for i, (r, e) in enumerate(zip(reference_tokens, engine_tokens)):
                if r != e:
                    print(f"  First mismatch at position {i}: ref={r} ({tokenizer.decode([r])!r}) vs engine={e} ({tokenizer.decode([e])!r})")
                    break

    # =========================================================================
    # Test 3: PagedEngine (paged attention)
    # =========================================================================
    print("\n[3] Testing PagedEngine.generate() (paged attention)...")
    paged_tokens = []
    
    # Calculate number of blocks needed
    m = model.config
    max_seq_len = len(prompt_tokens) + kwargs["max_tokens"]
    num_blocks_needed = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 10  # extra buffer
    print(f"    Block size: {BLOCK_SIZE}, allocating {num_blocks_needed} blocks")
    
    paged_engine = PagedEngine(model, tokenizer, num_blocks=num_blocks_needed, block_size=BLOCK_SIZE)
    torch.cuda.synchronize() if device_type == "cuda" else None
    t0 = time.time()
    with autocast_ctx:
        for token, mask in paged_engine.generate(prompt_tokens, **kwargs):
            paged_tokens.append(token)
            print(tokenizer.decode([token]), end="", flush=True)
    print()
    torch.cuda.synchronize() if device_type == "cuda" else None
    t1 = time.time()
    print(f"PagedEngine: {len(paged_tokens)} tokens in {t1 - t0:.2f}s")
    
    if args.test == "all":
        # Compare with reference
        match = reference_tokens == paged_tokens
        print(f"PagedEngine vs Reference: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            for i, (r, p) in enumerate(zip(reference_tokens, paged_tokens)):
                if r != p:
                    print(f"  First mismatch at position {i}: ref={r} ({tokenizer.decode([r])!r}) vs paged={p} ({tokenizer.decode([p])!r})")
                    break
    
    # =========================================================================
    # Test 4: LLM class (batched inference)
    # =========================================================================
    print("\n[4] Testing LLM.generate() (batched inference)...")
    
    # Create LLM instance
    llm = LLM(
        model_path=args.model_dir,
        step=step,
        device=str(device),
        num_blocks=200,
        block_size=BLOCK_SIZE,
        max_batch_size=4,
    )
    
    # Test with multiple prompts
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of Egypt?",
        "Explain quantum computing in simple terms.",
    ]
    
    torch.cuda.synchronize() if device_type == "cuda" else None
    t0 = time.time()
    outputs = llm.generate(
        test_prompts,
        sampling_params=SamplingParams(temperature=0.7, max_tokens=32),
        use_tqdm=True,
    )
    torch.cuda.synchronize() if device_type == "cuda" else None
    t1 = time.time()
    
    print(f"\nLLM batched generation completed in {t1 - t0:.2f}s")
    print("-" * 40)
    for i, (prompt, output) in enumerate(zip(test_prompts, outputs)):
        print(f"Prompt {i+1}: {prompt}")
        print(f"Output {i+1}: {output['text'][:100]}...")
        print()
    
    print("\n" + "=" * 60)
    print("Tests complete!")

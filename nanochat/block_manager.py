"""
BlockManager: Implements PagedAttention memory management.
Largely based on nanovllm's BlockManager.

This is the core of vLLM's memory efficiency. Instead of allocating
contiguous memory for each sequence's KV cache, we:

1. Divide GPU memory into fixed-size BLOCKS (e.g., 256 tokens each)
2. Each sequence gets a BLOCK TABLE mapping logical -> physical blocks
3. Allocate blocks on-demand as sequences grow
4. Free blocks when sequences finish

Benefits:
- No memory fragmentation (all blocks are same size)
- Sequences can use non-contiguous memory
- Prefix caching: sequences with same prefix share KV cache blocks

Example:
    Sequence A: "Hello world, how are" (tokens 0-99)
    Sequence B: "Hello world, what is" (tokens 0-99)
    
    With prefix caching, both sequences share the same physical block
    for "Hello world" since the tokens are identical.
"""

from collections import deque
from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from typing import List, Dict, Set, Optional
import xxhash
import numpy as np

BLOCK_SIZE = 16

# -----------------------------------------------------------------------------
# Sequence: Represents a single generation request

class SequenceStatus(Enum):
    """Possible states for a sequence."""
    WAITING = auto()   # In queue, waiting to be prefilled
    RUNNING = auto()   # Actively generating tokens
    FINISHED = auto()  # Generation complete


@dataclass
class SamplingParams:
    """Parameters for token sampling."""
    temperature: float = 1.0
    max_tokens: int = 256
    top_k: Optional[int] = None
    ignore_eos: bool = False


class Sequence:
    """
    Represents a single prompt being processed.
    
    Example:
        Prompt: "Hello world" -> tokens [1, 2, 3]
        After generating 2 tokens: [1, 2, 3, 4, 5]
        - num_prompt_tokens = 3
        - num_completion_tokens = 2
        - completion_token_ids = [4, 5]
    """
    
    block_size = BLOCK_SIZE         # Tokens per KV cache block (must match BlockManager)
    counter = count()               # Global counter to assign unique seq_ids

    def __init__(self, token_ids: List[int], sampling_params: SamplingParams = None):
        """
        Create a new sequence from prompt tokens.
        
        Args:
            token_ids: The tokenized prompt
            sampling_params: Temperature, max_tokens, etc.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
            
        self.seq_id = next(Sequence.counter)  # Unique ID for this sequence
        self.status = SequenceStatus.WAITING  # Start in waiting state
        self.token_ids = copy(token_ids)      # Copy to avoid mutating original
        self.last_token = token_ids[-1]       # Cache last token for decode phase
        self.num_tokens = len(self.token_ids) # Total tokens (prompt + completion)
        self.num_prompt_tokens = len(token_ids)  # Original prompt length (fixed)
        self.num_cached_tokens = 0            # Tokens with KV already in cache (prefix caching)
        
        # block_table maps logical block index -> physical block ID in KV cache
        # Example: [5, 12, 3] means:
        #   - Block 0 of this sequence is stored in physical block 5
        #   - Block 1 is in physical block 12
        #   - Block 2 is in physical block 3
        self.block_table: List[int] = []
        
        # Copy sampling parameters
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.top_k = sampling_params.top_k
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """Return total number of tokens (prompt + completion)."""
        return self.num_tokens

    def __getitem__(self, key):
        """Allow indexing into token_ids like seq[0] or seq[1:5]."""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """Check if generation is complete."""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """Number of tokens generated so far (not including prompt)."""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """Return just the original prompt tokens."""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """Return just the generated tokens (what we return to user)."""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """Number of complete blocks already in KV cache (for prefix caching)."""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """Total number of blocks needed for all tokens (ceiling division)."""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        How many tokens are in the last (possibly partial) block.
        
        Example with block_size=256:
            - 300 tokens -> 2 blocks, last block has 300 - 256 = 44 tokens
            - 256 tokens -> 1 block, last block has 256 tokens
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> List[int]:
        """
        Get the token IDs for block i.
        
        Args:
            i: Block index (0, 1, 2, ...)
            
        Returns:
            List of token IDs in that block (up to block_size tokens)
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """Add a newly generated token to the sequence."""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        Custom pickle serialization for sending sequence to worker processes.
        
        Optimization: During decode, we don't need the full token_ids list
        (worker only needs last_token). This reduces data transfer.
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """Custom pickle deserialization."""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            # Prefill: we need full token list
            self.token_ids = state[-1]
        else:
            # Decode: we only have last token
            self.last_token = state[-1]


# -----------------------------------------------------------------------------
# Block and BlockManager

class Block:
    """
    Represents a single KV cache block. 
    """
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.token_ids: List[int] = [] 
        self.hash: int = -1 
        
    def update(self, hash: int, token_ids: List[int]):
        self.hash = hash
        self.token_ids = token_ids
        self.ref_count += 1
        
    def reset(self): 
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}  # hash -> block id
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()
        
    @classmethod
    def compute_hash(cls, token_ids: List[int], parent_hash: int = -1) -> int:
        """
        Compute the hash of a list of token ids.
        The hash of a block depends on the hash of previous blocks and the current token ids.
        
        This is a chain where hash(block_i) = hash(hash(block_i-1), token_ids_i).
        """
        h = xxhash.xxh64()
        if parent_hash != -1:
            h.update(parent_hash.to_bytes(8, 'big', signed=True))
        # Convert token_ids to bytes for hashing
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    def num_free_blocks(self) -> int:
        """Return number of available blocks."""
        return len(self.free_block_ids)
        
    def _allocate_block(self) -> Block:
        """Allocate a single free block."""
        if not self.free_block_ids:
            raise RuntimeError("Out of KV cache blocks")
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        block.reset()
        self.used_block_ids.add(block_id)
        return block
    
    def _deallocate_block(self, block_id: int):
        """Return a block to the free pool."""
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} still has ref_count={block.ref_count}"
        # Remove from hash table if present
        if block.hash != -1 and block.hash in self.hash_to_block_id:
            if self.hash_to_block_id[block.hash] == block_id:
                del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.discard(block_id)
        self.free_block_ids.append(block_id)

    def allocate(self, seq: Sequence):
        """
        Most important method of the BlockManager.
        It allocates blocks for a sequence if needed and updates the block table of the sequence.
        For each block:
        1. Compute the hash of the block.
        2. Check if the block already exists in the hash_to_block_id.
        3. If there is a hit, reuse that block 
        4. If there is no hit, allocate a new block.
        """
        h = -1  # no parent hash at the beginning
        for i in range(seq.num_blocks):
            
            token_ids = seq.block(i)  # token ids for the i-th block
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                # cache hit -- reuse the existing block
                if block_id in self.used_block_ids:
                    print(f"Block {block_id} is used by another sequence")
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # block was freed but still in hash table -- reclaim it
                    self.used_block_ids.add(block_id)
                    self.free_block_ids.remove(block_id)
                    block = self.blocks[block_id]
                    block.ref_count = 1
            else:
                # cache miss -- allocate a new block
                block = self._allocate_block()
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block.block_id
            
            seq.block_table.append(block.block_id)
            
    def deallocate(self, seq: Sequence):
        """
        Deallocate blocks for a sequence.
        Decrements ref_count and frees blocks that are no longer used.
        """
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
                
        seq.num_cached_tokens = 0
        seq.block_table.clear()

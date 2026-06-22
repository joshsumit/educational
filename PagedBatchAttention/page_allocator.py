
import numpy as np
from tensor_utils import subsection


class PhysicalBlock:
    def __init__(self, block_id: int, block_size: int, n_heads: int, d_k: int):
        self.block_id = block_id
        self.block_size = block_size
        self.n_heads = n_heads
        self.d_k = d_k
        self.K = np.zeros((n_heads, block_size, d_k), dtype=np.float32)
        self.V = np.zeros((n_heads, block_size, d_k), dtype=np.float32)
        self.used_tokens = 0

    def has_space(self) -> bool:
        return self.used_tokens < self.block_size

    def append_token(self, k_token: np.ndarray, v_token: np.ndarray) -> int:
        if not self.has_space():
            raise RuntimeError(f"Physical block {self.block_id} is full")
        slot = self.used_tokens
        self.K[:, slot, :] = k_token
        self.V[:, slot, :] = v_token
        self.used_tokens += 1
        return slot

    def active_k(self):
        return self.K[:, :self.used_tokens, :]

    def active_v(self):
        return self.V[:, :self.used_tokens, :]


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int, n_heads: int, d_k: int):
        self.free_blocks = [PhysicalBlock(i, block_size, n_heads, d_k) for i in range(num_blocks)]
        self.allocated_blocks = []

    def allocate_block(self) -> PhysicalBlock:
        if not self.free_blocks:
            raise RuntimeError("Out of physical blocks. Increase num_blocks for the demo.")
        block = self.free_blocks.pop(0)
        self.allocated_blocks.append(block)
        subsection("ALLOCATOR: NEW PHYSICAL BLOCK ALLOCATED")
        print(f"Allocated physical block id: {block.block_id}")
        print(f"Block capacity in tokens: {block.block_size}")
        print(f"Free blocks remaining: {len(self.free_blocks)}")
        return block

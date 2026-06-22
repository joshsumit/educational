import numpy as np
from page_allocator import BlockAllocator
from tensor_utils import subsection, print_tensor


class PagedKVCache:
    """
    Educational paged KV cache.

    Core idea:
    - logical token order is the order tokens appear in the sequence
    - physical storage uses fixed-size blocks/pages
    - each token is mapped to (physical_block_id, slot_within_block)

    Instead of storing the whole cache in one giant contiguous tensor,
    we allocate fixed-size blocks and keep a mapping table.
    """

    def __init__(self, n_heads: int, d_k: int, block_size: int = 2, num_blocks: int = 32):
        self.n_heads = n_heads
        self.d_k = d_k
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size, n_heads=n_heads, d_k=d_k)

        # Ordered list of physical blocks that belong to this sequence.
        self.sequence_blocks = []

        # One mapping entry per logical token.
        # Each entry is a dict with keys:
        #   token_index, block_id, slot
        self.token_table = []

        self.total_tokens = 0

    def _ensure_writable_block(self):
        if not self.sequence_blocks or not self.sequence_blocks[-1].has_space():
            new_block = self.allocator.allocate_block()
            self.sequence_blocks.append(new_block)

    def append_token(self, k_token: np.ndarray, v_token: np.ndarray):
        """
        Append one token into the paged cache.

        k_token: [H, Dk]
        v_token: [H, Dk]
        """
        self._ensure_writable_block()
        block = self.sequence_blocks[-1]
        slot = block.append_token(k_token, v_token)

        record = {
            "token_index": self.total_tokens,
            "block_id": block.block_id,
            "slot": slot,
        }
        self.token_table.append(record)
        self.total_tokens += 1

        subsection("CACHE WRITE: APPEND ONE TOKEN INTO PAGED KV CACHE")
        print(f"Logical token index written: {record['token_index']}")
        print(f"Physical destination block: {record['block_id']}")
        print(f"Slot inside that block: {record['slot']}")
        print_tensor("K token being written", k_token, "One token's K across all heads", dim_names=["H", "Dk"])
        print_tensor("V token being written", v_token, "One token's V across all heads", dim_names=["H", "Dk"])

    def append_sequence(self, K_sequence: np.ndarray, V_sequence: np.ndarray):
        """
        Append a whole sequence of tokens during prefill.

        K_sequence: [H, S, Dk]
        V_sequence: [H, S, Dk]
        """
        S = K_sequence.shape[1]
        subsection("CACHE WRITE: PREFILL APPENDS PROMPT TOKENS ONE BY ONE")
        for t in range(S):
            self.append_token(K_sequence[:, t, :], V_sequence[:, t, :])

    def dump_page_table(self):
        subsection("PAGE TABLE: LOGICAL TOKEN -> PHYSICAL BLOCK / SLOT")
        for row in self.token_table:
            print(
                f"Logical token {row['token_index']} -> physical block {row['block_id']}, slot {row['slot']}"
            )

    def dump_physical_blocks(self):
        subsection("PHYSICAL BLOCK CONTENTS")
        for block in self.sequence_blocks:
            print(f"Physical block {block.block_id}: used_tokens={block.used_tokens}/{block.block_size}")
            print_tensor(
                f"Block {block.block_id} active K",
                block.active_k(),
                "Only the used token slots in this block for K",
                dim_names=["H", "tokens_in_block", "Dk"],
            )
            print_tensor(
                f"Block {block.block_id} active V",
                block.active_v(),
                "Only the used token slots in this block for V",
                dim_names=["H", "tokens_in_block", "Dk"],
            )

    def gather_active_kv(self):
        """
        Gather active K/V by walking the physical blocks in sequence order.

        Returns:
            K_gathered: [H, T, Dk]
            V_gathered: [H, T, Dk]
        """
        subsection("PAGE WALK: GATHER K/V FOR ATTENTION")
        k_parts = []
        v_parts = []

        for block in self.sequence_blocks:
            print(f"Reading physical block {block.block_id} during gather")
            k_block = block.active_k()
            v_block = block.active_v()
            k_parts.append(k_block)
            v_parts.append(v_block)

        if not k_parts:
            raise RuntimeError("Cannot gather KV from an empty cache")

        K_gathered = np.concatenate(k_parts, axis=1)
        V_gathered = np.concatenate(v_parts, axis=1)

        print_tensor(
            "K gathered from all blocks",
            K_gathered,
            "All cached keys stitched together in logical token order",
            dim_names=["H", "T", "Dk"],
        )
        print_tensor(
            "V gathered from all blocks",
            V_gathered,
            "All cached values stitched together in logical token order",
            dim_names=["H", "T", "Dk"],
        )
        return K_gathered, V_gathered

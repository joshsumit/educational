
import numpy as np
from page_allocator import BlockAllocator
from tensor_utils import subsection, print_tensor


class SequencePagedState:
    """
    Per-sequence state for paged KV cache.

    Each sequence has:
    - its own ordered list of physical blocks
    - its own logical-token -> (block, slot) mapping table
    - its own token count

    This is the key abstraction needed for continuous batching:
    multiple users can share the same global physical allocator,
    while each user's attention only sees that user's own token history.
    """

    def __init__(self, sequence_id: str):
        self.sequence_id = sequence_id
        self.sequence_blocks = []
        self.token_table = []
        self.total_tokens = 0

    def has_blocks(self):
        return len(self.sequence_blocks) > 0


class MultiSequencePagedKVCache:
    """
    Educational multi-sequence paged KV cache.

    Design goals:
    - multiple sequences/users coexist at the same time
    - no padding is used anywhere
    - each sequence keeps independent logical ordering
    - all sequences share a single global physical block allocator

    This mirrors the core idea behind continuous batching / inflight batching:
    several requests can be active at once, have different prompt lengths,
    and grow independently over time.
    """

    def __init__(self, n_heads: int, d_k: int, block_size: int = 2, num_blocks: int = 64):
        self.n_heads = n_heads
        self.d_k = d_k
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size, n_heads=n_heads, d_k=d_k)
        self.sequences = {}

    def ensure_sequence(self, sequence_id: str):
        if sequence_id not in self.sequences:
            self.sequences[sequence_id] = SequencePagedState(sequence_id)
            subsection("CACHE MANAGER: NEW SEQUENCE REGISTERED")
            print(f"Sequence id created: {sequence_id}")

    def _ensure_writable_block(self, sequence_id: str):
        self.ensure_sequence(sequence_id)
        state = self.sequences[sequence_id]
        if not state.sequence_blocks or not state.sequence_blocks[-1].has_space():
            new_block = self.allocator.allocate_block()
            state.sequence_blocks.append(new_block)
            subsection("SEQUENCE RECEIVED A NEW WRITABLE PHYSICAL BLOCK")
            print(f"Sequence id: {sequence_id}")
            print(f"New physical block assigned: {new_block.block_id}")

    def append_token(self, sequence_id: str, k_token: np.ndarray, v_token: np.ndarray):
        """
        Append one token into one sequence's paged cache.

        k_token: [H, Dk]
        v_token: [H, Dk]
        """
        self._ensure_writable_block(sequence_id)
        state = self.sequences[sequence_id]
        block = state.sequence_blocks[-1]
        slot = block.append_token(k_token, v_token)

        record = {
            "token_index": state.total_tokens,
            "block_id": block.block_id,
            "slot": slot,
        }
        state.token_table.append(record)
        state.total_tokens += 1

        subsection("CACHE WRITE: APPEND ONE TOKEN INTO ONE SEQUENCE")
        print(f"Sequence id: {sequence_id}")
        print(f"Logical token index written: {record['token_index']}")
        print(f"Physical destination block: {record['block_id']}")
        print(f"Slot inside that block: {record['slot']}")
        print_tensor("K token being written", k_token, f"One token's K for sequence {sequence_id} across all heads", dim_names=["H", "Dk"])
        print_tensor("V token being written", v_token, f"One token's V for sequence {sequence_id} across all heads", dim_names=["H", "Dk"])

    def append_sequence(self, sequence_id: str, K_sequence: np.ndarray, V_sequence: np.ndarray):
        """
        Append a whole prompt during prefill.

        K_sequence: [H, S, Dk]
        V_sequence: [H, S, Dk]
        """
        S = K_sequence.shape[1]
        subsection("CACHE WRITE: PREFILL APPENDS PROMPT TOKENS FOR ONE SEQUENCE")
        print(f"Sequence id: {sequence_id}")
        print(f"Prompt token count for this sequence: {S}")
        for t in range(S):
            self.append_token(sequence_id, K_sequence[:, t, :], V_sequence[:, t, :])

    def dump_page_table(self, sequence_id: str = None):
        if sequence_id is not None:
            state = self.sequences[sequence_id]
            subsection("PAGE TABLE: ONE SEQUENCE")
            print(f"Sequence id: {sequence_id}")
            for row in state.token_table:
                print(f"Logical token {row['token_index']} -> physical block {row['block_id']}, slot {row['slot']}")
            return

        subsection("PAGE TABLE: ALL ACTIVE SEQUENCES")
        for seq_id, state in self.sequences.items():
            print(f"\nSequence id: {seq_id}")
            for row in state.token_table:
                print(f"  Logical token {row['token_index']} -> physical block {row['block_id']}, slot {row['slot']}")

    def dump_physical_blocks(self, sequence_id: str = None):
        if sequence_id is not None:
            state = self.sequences[sequence_id]
            subsection("PHYSICAL BLOCK CONTENTS FOR ONE SEQUENCE")
            print(f"Sequence id: {sequence_id}")
            for block in state.sequence_blocks:
                print(f"Physical block {block.block_id}: used_tokens={block.used_tokens}/{block.block_size}")
                print_tensor(f"Block {block.block_id} active K", block.active_k(), "Only used K slots for this sequence", dim_names=["H", "tokens_in_block", "Dk"])
                print_tensor(f"Block {block.block_id} active V", block.active_v(), "Only used V slots for this sequence", dim_names=["H", "tokens_in_block", "Dk"])
            return

        subsection("PHYSICAL BLOCK CONTENTS FOR ALL ACTIVE SEQUENCES")
        for seq_id, state in self.sequences.items():
            print(f"\nSequence id: {seq_id}")
            for block in state.sequence_blocks:
                print(f"Physical block {block.block_id}: used_tokens={block.used_tokens}/{block.block_size}")
                print_tensor(f"Block {block.block_id} active K", block.active_k(), f"Active K for sequence {seq_id}", dim_names=["H", "tokens_in_block", "Dk"])
                print_tensor(f"Block {block.block_id} active V", block.active_v(), f"Active V for sequence {seq_id}", dim_names=["H", "tokens_in_block", "Dk"])

    def gather_active_kv(self, sequence_id: str):
        """
        Gather active K/V for one sequence only.

        Returns:
            K_gathered: [H, T, Dk]
            V_gathered: [H, T, Dk]

        Even though physical blocks are shared by the allocator across all sequences,
        this gather only walks the block list of the requested sequence.
        This is how sequence isolation is honored without padding.
        """
        state = self.sequences[sequence_id]
        subsection("PAGE WALK: GATHER K/V FOR ONE SEQUENCE")
        print(f"Sequence id: {sequence_id}")
        k_parts = []
        v_parts = []

        for block in state.sequence_blocks:
            print(f"Reading physical block {block.block_id} during gather for sequence {sequence_id}")
            k_parts.append(block.active_k())
            v_parts.append(block.active_v())

        if not k_parts:
            raise RuntimeError(f"Cannot gather KV from empty cache for sequence {sequence_id}")

        K_gathered = np.concatenate(k_parts, axis=1)
        V_gathered = np.concatenate(v_parts, axis=1)
        print_tensor("K gathered from all blocks", K_gathered, f"All cached keys for sequence {sequence_id} in logical token order", dim_names=["H", "T", "Dk"])
        print_tensor("V gathered from all blocks", V_gathered, f"All cached values for sequence {sequence_id} in logical token order", dim_names=["H", "T", "Dk"])
        return K_gathered, V_gathered

    def active_sequence_ids(self):
        return list(self.sequences.keys())

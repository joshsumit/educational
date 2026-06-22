
import numpy as np
from decoder import EducationalPagedAttentionDecoder
from tensor_utils import banner, print_tensor


def demo():
    np.random.seed(42)

    banner("DEMO SETUP: MULTI-SEQUENCE, NO PADDING, CONTINUOUS / INFLIGHT BATCHING")
    print("This updated demo now supports MULTIPLE USERS / SEQUENCES.")
    print("Important properties:")
    print("- different users can have different prompt lengths")
    print("- no padding is used")
    print("- all users share one global physical block allocator")
    print("- each user has an independent logical KV history")
    print("- scheduler rounds can include different active users")
    print("- a new user can arrive while old users are still decoding")

    decoder = EducationalPagedAttentionDecoder(
        d_model=8,
        n_heads=2,
        block_size=2,
        num_blocks=32,
    )

    # Initial prompts with different lengths and no padding.
    prompts = {
        "user_A": np.random.randn(1, 4, 8).astype(np.float32),
        "user_B": np.random.randn(1, 2, 8).astype(np.float32),
    }

    banner("INITIAL PREFILL FOR EXISTING USERS")
    for seq_id, prompt in prompts.items():
        print_tensor(
            f"Prompt input for {seq_id}",
            prompt,
            f"Prompt hidden states for {seq_id}; note prompt length can differ by user",
            dim_names=["B", "S", "D"],
        )
    decoder.prefill_batch_no_padding(prompts)

    # Round 0: both users decode one token.
    round0 = {
        "user_A": np.random.randn(1, 1, 8).astype(np.float32),
        "user_B": np.random.randn(1, 1, 8).astype(np.float32),
    }
    banner("ROUND 0 INPUT TOKENS")
    for seq_id, tok in round0.items():
        print_tensor(
            f"Decode token for {seq_id} in round 0",
            tok,
            f"One new decode token for {seq_id}",
            dim_names=["B", "S=1", "D"],
        )
    decoder.decode_round_no_padding(round_id=0, decode_inputs=round0)

    # New user arrives while earlier users are still active.
    late_prompt = np.random.randn(1, 3, 8).astype(np.float32)
    banner("A NEW USER ARRIVES WHILE OTHERS ARE STILL ACTIVE")
    print_tensor(
        "Prompt input for user_C",
        late_prompt,
        "New request enters the runtime after earlier users already started decoding",
        dim_names=["B", "S", "D"],
    )
    decoder.prefill_sequence("user_C", late_prompt)

    # Round 1: all three users are active.
    round1 = {
        "user_A": np.random.randn(1, 1, 8).astype(np.float32),
        "user_B": np.random.randn(1, 1, 8).astype(np.float32),
        "user_C": np.random.randn(1, 1, 8).astype(np.float32),
    }
    banner("ROUND 1 INPUT TOKENS")
    for seq_id, tok in round1.items():
        print_tensor(
            f"Decode token for {seq_id} in round 1",
            tok,
            f"One new decode token for {seq_id}",
            dim_names=["B", "S=1", "D"],
        )
    decoder.decode_round_no_padding(round_id=1, decode_inputs=round1)

    # Round 2: user_B is assumed complete, user_A and user_C continue.
    round2 = {
        "user_A": np.random.randn(1, 1, 8).astype(np.float32),
        "user_C": np.random.randn(1, 1, 8).astype(np.float32),
    }
    banner("ROUND 2 INPUT TOKENS")
    for seq_id, tok in round2.items():
        print_tensor(
            f"Decode token for {seq_id} in round 2",
            tok,
            f"One new decode token for {seq_id}; user_B is omitted to simulate a completed request",
            dim_names=["B", "S=1", "D"],
        )
    decoder.decode_round_no_padding(round_id=2, decode_inputs=round2)

    banner("FINAL GLOBAL CACHE VIEW")
    decoder.cache.dump_page_table()
    decoder.cache.dump_physical_blocks()


if __name__ == "__main__":
    demo()


import numpy as np
from decoder import EducationalPagedAttentionDecoder
from tensor_utils import banner, print_tensor


def demo():
    np.random.seed(42)

    banner("DEMO SETUP: TRUE BATCHED QUERY ROWS WITH RAGGED PER-SEQUENCE KV")
    print("This demo keeps the existing educational paged-attention architecture, but now also shows a REAL batched decode query tensor.")
    print("Key points:")
    print("- multiple users have different prompt lengths")
    print("- no padding is used for prompt or KV histories")
    print("- every active decode sequence contributes one row to Q_batch")
    print("- therefore Q_batch can have B > 1")
    print("- but each row still gathers only its own per-sequence KV cache")
    print("- this demonstrates the row mapping idea: row 0 -> user_A, row 1 -> user_B, ...")

    decoder = EducationalPagedAttentionDecoder(
        d_model=8,
        n_heads=2,
        block_size=2,
        num_blocks=48,
    )

    # ----------------------------------------------------------------------
    # STEP 1: Prefill four users with different prompt lengths and no padding.
    # ----------------------------------------------------------------------
    prompts = {
        "user_A": np.random.randn(1, 4, 8).astype(np.float32),
        "user_B": np.random.randn(1, 2, 8).astype(np.float32),
        "user_C": np.random.randn(1, 3, 8).astype(np.float32),
        "user_D": np.random.randn(1, 5, 8).astype(np.float32),
    }

    banner("INITIAL PREFILL FOR FOUR DIFFERENT USERS")
    for seq_id, prompt in prompts.items():
        print_tensor(
            f"Prompt input for {seq_id}",
            prompt,
            f"Prompt hidden states for {seq_id}; note prompt length differs by user and no padding is used.",
            dim_names=["B_local", "S", "D"],
        )

    decoder.prefill_batch_no_padding(prompts)

    # ----------------------------------------------------------------------
    # STEP 2: One scheduler round where all four users are active.
    # This is the place where we want to visibly support:
    #   row 0 -> user_A
    #   row 1 -> user_B
    #   row 2 -> user_C
    #   row 3 -> user_D
    # so the batched decode query tensor has B = 4.
    # ----------------------------------------------------------------------
    decode_round_0 = {
        "user_A": np.random.randn(1, 1, 8).astype(np.float32),
        "user_B": np.random.randn(1, 1, 8).astype(np.float32),
        "user_C": np.random.randn(1, 1, 8).astype(np.float32),
        "user_D": np.random.randn(1, 1, 8).astype(np.float32),
    }

    banner("ROUND 0 INPUT TOKENS FOR A FOUR-ROW BATCHED DECODE")
    for seq_id, tok in decode_round_0.items():
        print_tensor(
            f"Decode token for {seq_id} in round 0",
            tok,
            f"One new decode token for {seq_id}. Every active sequence contributes one row to the upcoming Q_batch.",
            dim_names=["B_local", "S=1", "D"],
        )

    batched_out, row_outputs, row_sequence_ids = decoder.decode_round_ragged_batch(round_id=0, decode_inputs=decode_round_0)

    banner("SUMMARY AFTER ROUND 0")
    print("This is the exact row ownership of the batched output tensor:")
    for row_idx, seq_id in enumerate(row_sequence_ids):
        print(f"row {row_idx} -> {seq_id}")

    print_tensor(
        "Batched output after round 0",
        batched_out,
        "The batch dimension here is real. Because four active sequences participated, B_active = 4.",
        dim_names=["B_active", "S=1", "D"],
    )

    # ----------------------------------------------------------------------
    # STEP 3: Show that later rounds can have a different serving batch size.
    # Here user_B is omitted and only A/C/D remain active, so Q_batch will have B = 3.
    # ----------------------------------------------------------------------
    decode_round_1 = {
        "user_A": np.random.randn(1, 1, 8).astype(np.float32),
        "user_C": np.random.randn(1, 1, 8).astype(np.float32),
        "user_D": np.random.randn(1, 1, 8).astype(np.float32),
    }

    banner("ROUND 1 INPUT TOKENS FOR A THREE-ROW BATCHED DECODE")
    for seq_id, tok in decode_round_1.items():
        print_tensor(
            f"Decode token for {seq_id} in round 1",
            tok,
            f"One new decode token for {seq_id}. user_B is omitted in this round, so the next Q_batch will have only three rows.",
            dim_names=["B_local", "S=1", "D"],
        )

    batched_out_1, row_outputs_1, row_sequence_ids_1 = decoder.decode_round_ragged_batch(round_id=1, decode_inputs=decode_round_1)

    banner("SUMMARY AFTER ROUND 1")
    print("This round has fewer active sequences, so B_active changed.")
    for row_idx, seq_id in enumerate(row_sequence_ids_1):
        print(f"row {row_idx} -> {seq_id}")
    print_tensor(
        "Batched output after round 1",
        batched_out_1,
        "Here B_active = 3 because only three sequences were scheduled in this round.",
        dim_names=["B_active", "S=1", "D"],
    )


if __name__ == "__main__":
    demo()

import numpy as np
from decoder import EducationalPagedAttentionDecoder
from tensor_utils import banner, print_tensor


def demo():
    np.random.seed(42)

    banner("DEMO SETUP")
    print("This demo uses tiny dimensions so tensor values stay readable.")
    print("Configuration:")
    print("- Batch size B = 1")
    print("- Prompt sequence length S = 4")
    print("- Model hidden dimension D = 8")
    print("- Number of heads H = 2")
    print("- Per-head hidden dimension Dk = 4")
    print("- Physical block size = 2 tokens per block")

    decoder = EducationalPagedAttentionDecoder(
        d_model=8,
        n_heads=2,
        block_size=2,
        num_blocks=16,
    )

    # Prompt tokens for prefill.
    prompt = np.random.randn(1, 4, 8).astype(np.float32)
    print_tensor(
        "Prompt input",
        prompt,
        "This is the prompt hidden-state tensor that enters prefill",
        dim_names=["B", "S", "D"],
    )

    prefill_out = decoder.prefill(prompt)
    print_tensor(
        "Prefill output",
        prefill_out,
        "Attention output of the prompt positions after prefill",
        dim_names=["B", "S", "D"],
    )

    # Now decode several new tokens one by one.
    for step in range(3):
        new_token = np.random.randn(1, 1, 8).astype(np.float32)
        print_tensor(
            f"Decode input token at step {step}",
            new_token,
            "Single token hidden state entering autoregressive decode",
            dim_names=["B", "S=1", "D"],
        )
        decode_out = decoder.decode_step(new_token, step_idx=step)
        print_tensor(
            f"Decode output at step {step}",
            decode_out,
            "Attention output after this decode step",
            dim_names=["B", "S=1", "D"],
        )


if __name__ == "__main__":
    demo()

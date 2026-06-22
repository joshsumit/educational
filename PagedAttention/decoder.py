
import numpy as np
from attention import scaled_dot_product_attention
from paged_kv_cache import MultiSequencePagedKVCache
from tensor_utils import banner, subsection, print_tensor, print_matrix_multiply, print_tensor_stats


class EducationalPagedAttentionDecoder:
    """
    One decoder layer focused on attention and KV cache behavior.

    This updated version supports MULTIPLE ACTIVE SEQUENCES with NO PADDING.
    That makes it suitable for educational continuous batching / inflight batching.

    Key design choice:
    - the physical allocator is global
    - the cache state is per sequence/user
    - attention for one sequence gathers only that sequence's K/V history
    """

    def __init__(self, d_model: int = 8, n_heads: int = 2, block_size: int = 2, num_blocks: int = 64):
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.block_size = block_size

        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

        # Shared cache manager across many active sequences.
        self.cache = MultiSequencePagedKVCache(n_heads=n_heads, d_k=self.d_k, block_size=block_size, num_blocks=num_blocks)

    def project_qkv(self, x: np.ndarray):
        subsection("LINEAR PROJECTIONS: BUILD Q, K, V")
        print_tensor("Input to projection", x, "Input hidden states before Q/K/V projection", dim_names=["B", "S", "D"])

        print_matrix_multiply("Q projection", left_shape=x.shape, right_shape=self.W_q.shape, out_shape=x.shape, explanation="Each token hidden vector is projected into query space")
        Q_raw = np.matmul(x, self.W_q)
        print_tensor("Q raw", Q_raw, "Queries before head split", dim_names=["B", "S", "D"])
        print_tensor_stats("Q raw", Q_raw)

        print_matrix_multiply("K projection", left_shape=x.shape, right_shape=self.W_k.shape, out_shape=x.shape, explanation="Each token hidden vector is projected into key space")
        K_raw = np.matmul(x, self.W_k)
        print_tensor("K raw", K_raw, "Keys before head split", dim_names=["B", "S", "D"])
        print_tensor_stats("K raw", K_raw)

        print_matrix_multiply("V projection", left_shape=x.shape, right_shape=self.W_v.shape, out_shape=x.shape, explanation="Each token hidden vector is projected into value space")
        V_raw = np.matmul(x, self.W_v)
        print_tensor("V raw", V_raw, "Values before head split", dim_names=["B", "S", "D"])
        print_tensor_stats("V raw", V_raw)

        return Q_raw, K_raw, V_raw

    def split_heads(self, x_raw: np.ndarray, tensor_name: str):
        B, S, D = x_raw.shape

        subsection(f"HEAD SPLIT FOR {tensor_name}")
        print(f"Before any shape change, {tensor_name} is still [B, S, D].")
        print_tensor(f"{tensor_name} before split", x_raw, f"{tensor_name} before introducing heads", dim_names=["B", "S", "D"])

        reshaped = x_raw.reshape(B, S, self.n_heads, self.d_k)
        print("\nShape change explanation:")
        print("- We are splitting the last hidden dimension D into H heads and Dk per head.")
        print("- This changes [B, S, D] into [B, S, H, Dk].")
        print_tensor(f"{tensor_name} after reshape", reshaped, f"{tensor_name} with explicit head dimension, before transpose", dim_names=["B", "S", "H", "Dk"])

        transposed = reshaped.transpose(0, 2, 1, 3)
        print("\nShape change explanation:")
        print("- We now move H before S.")
        print("- This changes [B, S, H, Dk] into [B, H, S, Dk].")
        print("- This layout is convenient for per-head attention kernels.")
        print_tensor(f"{tensor_name} after transpose", transposed, f"{tensor_name} laid out per head for attention", dim_names=["B", "H", "S", "Dk"])

        return transposed

    def merge_heads(self, context: np.ndarray):
        subsection("MERGE HEADS AFTER ATTENTION")
        print_tensor("Context before merge", context, "Attention output per head", dim_names=["B", "H", "S", "Dk"])

        transposed = context.transpose(0, 2, 1, 3)
        print("\nShape change explanation:")
        print("- Move sequence dimension back in front of heads.")
        print("- [B, H, S, Dk] becomes [B, S, H, Dk].")
        print_tensor("Context after transpose", transposed, "Sequence-major layout before final reshape", dim_names=["B", "S", "H", "Dk"])

        B, S, H, Dk = transposed.shape
        merged = transposed.reshape(B, S, H * Dk)
        print("\nShape change explanation:")
        print("- Collapse H and Dk back into the original hidden size D.")
        print("- [B, S, H, Dk] becomes [B, S, D].")
        print_tensor("Context after reshape", merged, "All heads concatenated back together", dim_names=["B", "S", "D"])
        return merged

    def output_projection(self, merged_context: np.ndarray):
        subsection("OUTPUT PROJECTION AFTER HEAD MERGE")
        print_matrix_multiply("Output projection", left_shape=merged_context.shape, right_shape=self.W_o.shape, out_shape=merged_context.shape, explanation="Project merged multi-head output back into model hidden space")
        out = np.matmul(merged_context, self.W_o)
        print_tensor("Attention output after W_o", out, "Final attention sublayer output", dim_names=["B", "S", "D"])
        return out

    def prefill_sequence(self, sequence_id: str, x_prompt: np.ndarray):
        """
        Prefill exactly one sequence.
        x_prompt shape: [1, S, D] where S can differ across users.

        No padding is required because we process each arriving prompt individually,
        while still sharing the same global allocator across all active sequences.
        """
        banner(f"PREFILL PHASE FOR SEQUENCE {sequence_id}: PROCESS PROMPT AND POPULATE ITS PAGED KV CACHE")
        Q_raw, K_raw, V_raw = self.project_qkv(x_prompt)
        Q = self.split_heads(Q_raw, f"Q_{sequence_id}")
        K = self.split_heads(K_raw, f"K_{sequence_id}")
        V = self.split_heads(V_raw, f"V_{sequence_id}")

        self.cache.append_sequence(sequence_id, K[0], V[0])
        self.cache.dump_page_table(sequence_id)
        self.cache.dump_physical_blocks(sequence_id)

        K_gathered, V_gathered = self.cache.gather_active_kv(sequence_id)
        K_batched = K_gathered[None, :, :, :]
        V_batched = V_gathered[None, :, :, :]

        context, scores, weights = scaled_dot_product_attention(
            Q=Q,
            K=K_batched,
            V=V_batched,
            use_causal_mask=True,
            phase_name=f"PREFILL FOR SEQUENCE {sequence_id}",
        )

        merged = self.merge_heads(context)
        out = self.output_projection(merged)
        banner(f"PREFILL COMPLETE FOR SEQUENCE {sequence_id}")
        return out

    def decode_step(self, sequence_id: str, x_token: np.ndarray, step_idx: int):
        """
        Decode one new token for one sequence.
        x_token shape: [1, 1, D]

        This method is suitable for inflight batching schedulers:
        a scheduler can call decode_step for whichever sequence(s) are active in this round.
        """
        banner(f"DECODE STEP {step_idx} FOR SEQUENCE {sequence_id}: PROCESS ONE NEW TOKEN")
        Q_raw, K_raw, V_raw = self.project_qkv(x_token)
        Q = self.split_heads(Q_raw, f"Q_decode_{sequence_id}")
        K_new = self.split_heads(K_raw, f"K_decode_{sequence_id}")
        V_new = self.split_heads(V_raw, f"V_decode_{sequence_id}")

        self.cache.append_token(sequence_id, K_new[0, :, 0, :], V_new[0, :, 0, :])
        self.cache.dump_page_table(sequence_id)
        self.cache.dump_physical_blocks(sequence_id)

        K_gathered, V_gathered = self.cache.gather_active_kv(sequence_id)
        K_batched = K_gathered[None, :, :, :]
        V_batched = V_gathered[None, :, :, :]

        context, scores, weights = scaled_dot_product_attention(
            Q=Q,
            K=K_batched,
            V=V_batched,
            use_causal_mask=False,
            phase_name=f"DECODE STEP {step_idx} FOR SEQUENCE {sequence_id}",
        )

        merged = self.merge_heads(context)
        out = self.output_projection(merged)
        banner(f"DECODE STEP {step_idx} COMPLETE FOR SEQUENCE {sequence_id}")
        return out

    def prefill_batch_no_padding(self, sequence_inputs: dict):
        """
        sequence_inputs: dict[str, np.ndarray]
            Each value is [1, S_i, D].

        Educational version of batch prefill without padding.
        We iterate over variable-length prompts and prefill them one by one,
        but all sequences become resident in the same shared runtime after that.
        """
        banner("NO-PADDING MULTI-SEQUENCE PREFILL")
        outputs = {}
        for sequence_id, x_prompt in sequence_inputs.items():
            print(f"\nScheduler prefill dispatch -> sequence_id={sequence_id}, prompt_len={x_prompt.shape[1]}")
            outputs[sequence_id] = self.prefill_sequence(sequence_id, x_prompt)
        return outputs

    def decode_round_no_padding(self, round_id: int, decode_inputs: dict):
        """
        decode_inputs: dict[str, np.ndarray]
            Each value is [1, 1, D].

        Educational version of one inflight batching round.
        Only active sequences included in decode_inputs participate in this round.
        Sequences absent from this round are simply not scheduled.
        """
        banner(f"INFLIGHT / CONTINUOUS BATCHING ROUND {round_id}")
        outputs = {}
        print("Active sequences in this scheduler round:")
        for sequence_id in decode_inputs:
            print(f"- {sequence_id}")
        for sequence_id, x_token in decode_inputs.items():
            outputs[sequence_id] = self.decode_step(sequence_id, x_token, step_idx=round_id)
        return outputs

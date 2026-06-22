import numpy as np
from attention import scaled_dot_product_attention
from paged_kv_cache import PagedKVCache
from tensor_utils import banner, subsection, print_tensor, print_matrix_multiply, print_tensor_stats


class EducationalPagedAttentionDecoder:
    """
    One decoder layer focused on attention and KV cache behavior.

    This is intentionally small and heavily instrumented.
    The goal is understanding, not performance.
    """

    def __init__(self, d_model: int = 8, n_heads: int = 2, block_size: int = 2, num_blocks: int = 32):
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.block_size = block_size

        # Separate Q/K/V/O matrices so the reader can follow each operation individually.
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

        self.cache = PagedKVCache(n_heads=n_heads, d_k=self.d_k, block_size=block_size, num_blocks=num_blocks)

    def project_qkv(self, x: np.ndarray):
        """
        x: [B, S, D]

        Q_raw/K_raw/V_raw stay in the same rank and same visible shape [B, S, D].
        Heads do NOT exist yet at this point.
        Heads appear only after reshape.
        """
        subsection("LINEAR PROJECTIONS: BUILD Q, K, V")
        print_tensor("Input to projection", x, "Input hidden states before Q/K/V projection", dim_names=["B", "S", "D"])

        print_matrix_multiply(
            "Q projection",
            left_shape=x.shape,
            right_shape=self.W_q.shape,
            out_shape=x.shape,
            explanation="Each token hidden vector is projected into query space",
        )
        Q_raw = np.matmul(x, self.W_q)
        print_tensor("Q raw", Q_raw, "Queries before head split", dim_names=["B", "S", "D"])
        print_tensor_stats("Q raw", Q_raw)

        print_matrix_multiply(
            "K projection",
            left_shape=x.shape,
            right_shape=self.W_k.shape,
            out_shape=x.shape,
            explanation="Each token hidden vector is projected into key space",
        )
        K_raw = np.matmul(x, self.W_k)
        print_tensor("K raw", K_raw, "Keys before head split", dim_names=["B", "S", "D"])
        print_tensor_stats("K raw", K_raw)

        print_matrix_multiply(
            "V projection",
            left_shape=x.shape,
            right_shape=self.W_v.shape,
            out_shape=x.shape,
            explanation="Each token hidden vector is projected into value space",
        )
        V_raw = np.matmul(x, self.W_v)
        print_tensor("V raw", V_raw, "Values before head split", dim_names=["B", "S", "D"])
        print_tensor_stats("V raw", V_raw)

        return Q_raw, K_raw, V_raw

    def split_heads(self, x_raw: np.ndarray, tensor_name: str):
        """
        Shape journey:
        1. Start from [B, S, D]
        2. Reshape to [B, S, H, Dk]
        3. Transpose to [B, H, S, Dk]

        Why transpose?
        Because attention is computed independently for each head,
        so most implementations want the head dimension before sequence.
        """
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
        """
        Reverse of split_heads:
        [B, H, S, Dk] -> [B, S, H, Dk] -> [B, S, D]
        """
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
        print_matrix_multiply(
            "Output projection",
            left_shape=merged_context.shape,
            right_shape=self.W_o.shape,
            out_shape=merged_context.shape,
            explanation="Project merged multi-head output back into model hidden space",
        )
        out = np.matmul(merged_context, self.W_o)
        print_tensor("Attention output after W_o", out, "Final attention sublayer output", dim_names=["B", "S", "D"])
        return out

    def prefill(self, x_prompt: np.ndarray):
        """
        Prefill processes the full prompt sequence.

        Crucially, prefill also WRITES prompt K/V into the cache.
        Then attention can be computed against that context.
        """
        banner("PREFILL PHASE: PROCESS PROMPT AND POPULATE PAGED KV CACHE")
        Q_raw, K_raw, V_raw = self.project_qkv(x_prompt)

        # Shape changes happen explicitly and are narrated.
        Q = self.split_heads(Q_raw, "Q")
        K = self.split_heads(K_raw, "K")
        V = self.split_heads(V_raw, "V")

        # Write prompt K/V into paged cache.
        self.cache.append_sequence(K[0], V[0])
        self.cache.dump_page_table()
        self.cache.dump_physical_blocks()

        # Gather back from blocks to emulate paged-attention style access.
        K_gathered, V_gathered = self.cache.gather_active_kv()

        # Attention function expects batch dimension, so we add batch axis back.
        K_batched = K_gathered[None, :, :, :]
        V_batched = V_gathered[None, :, :, :]

        context, scores, weights = scaled_dot_product_attention(
            Q=Q,
            K=K_batched,
            V=V_batched,
            use_causal_mask=True,
            phase_name="PREFILL",
        )

        merged = self.merge_heads(context)
        out = self.output_projection(merged)
        banner("PREFILL COMPLETE")
        return out

    def decode_step(self, x_token: np.ndarray, step_idx: int):
        """
        Decode processes one new token at a time.

        The new token creates a new Q, K, V.
        The new K/V are appended to the cache.
        Then attention uses the new Q against the entire gathered cached K/V.
        """
        banner(f"DECODE STEP {step_idx}: PROCESS ONE NEW TOKEN AND ATTEND TO FULL CACHE")
        Q_raw, K_raw, V_raw = self.project_qkv(x_token)

        # Each of these starts as [B, 1, D] because decode handles one token.
        Q = self.split_heads(Q_raw, "Q_decode")
        K_new = self.split_heads(K_raw, "K_decode")
        V_new = self.split_heads(V_raw, "V_decode")

        # Append only the single new token's K/V into cache.
        self.cache.append_token(K_new[0, :, 0, :], V_new[0, :, 0, :])
        self.cache.dump_page_table()
        self.cache.dump_physical_blocks()

        # Gather all K/V that now exist: prompt + previous decoded + current token.
        K_gathered, V_gathered = self.cache.gather_active_kv()
        K_batched = K_gathered[None, :, :, :]
        V_batched = V_gathered[None, :, :, :]

        context, scores, weights = scaled_dot_product_attention(
            Q=Q,
            K=K_batched,
            V=V_batched,
            use_causal_mask=False,
            phase_name=f"DECODE STEP {step_idx}",
        )

        merged = self.merge_heads(context)
        out = self.output_projection(merged)
        banner(f"DECODE STEP {step_idx} COMPLETE")
        return out


import numpy as np
from attention import scaled_dot_product_attention
from paged_kv_cache import MultiSequencePagedKVCache
from tensor_utils import banner, subsection, print_tensor, print_matrix_multiply, print_tensor_stats


class EducationalPagedAttentionDecoder:
    """
    Educational decoder focused on:
    - Q / K / V projection
    - head split / transpose
    - paged KV caching
    - multi-sequence serving without padding
    - continuous / inflight batching intuition

    IMPORTANT DESIGN CHOICE
    -----------------------
    This implementation supports two batching views:

    1. Serving batch:
       Multiple sequences can be active in one scheduler round.

    2. Tensor batch:
       During the new ragged batched decode path, all active sequences contribute
       one query row each, so Q becomes a real batched tensor with shape:
           [B_active, H, 1, Dk]

       However, K/V remain ragged per row because each sequence can have a
       different context length T_i and we explicitly avoid padding.

    That means:
    - row 0 may belong to user_A
    - row 1 may belong to user_B
    - row 2 may belong to user_C
    - row 3 may belong to user_D

    but each row still gathers only its own sequence's KV history.
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

    def _print_row_mapping(self, row_sequence_ids):
        subsection("BATCH ROW MAPPING")
        print("This mapping tells us which active sequence owns each row in the batched decode query tensor.")
        for row_idx, sequence_id in enumerate(row_sequence_ids):
            print(f"row {row_idx} -> {sequence_id}")
        print(f"Therefore the batched query tensor has B = {len(row_sequence_ids)} rows in this scheduler round.")

    def prefill_sequence(self, sequence_id: str, x_prompt: np.ndarray):
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

    def prefill_batch_no_padding(self, sequence_inputs: dict):
        banner("NO-PADDING MULTI-SEQUENCE PREFILL")
        outputs = {}
        for sequence_id, x_prompt in sequence_inputs.items():
            print(f"\nScheduler prefill dispatch -> sequence_id={sequence_id}, prompt_len={x_prompt.shape[1]}")
            outputs[sequence_id] = self.prefill_sequence(sequence_id, x_prompt)
        return outputs

    def decode_step(self, sequence_id: str, x_token: np.ndarray, step_idx: int):
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

    def decode_round_no_padding(self, round_id: int, decode_inputs: dict):
        banner(f"INFLIGHT / CONTINUOUS BATCHING ROUND {round_id}")
        outputs = {}
        print("Active sequences in this scheduler round:")
        for sequence_id in decode_inputs:
            print(f"- {sequence_id}")
        for sequence_id, x_token in decode_inputs.items():
            outputs[sequence_id] = self.decode_step(sequence_id, x_token, step_idx=round_id)
        return outputs

    def decode_round_ragged_batch(self, round_id: int, decode_inputs: dict):
        """
        Educational ragged batched decode.

        This method shows a real batched Q tensor with B_active > 1 while still
        avoiding padding on the context side.

        Why only Q is densely batched:
        - every active sequence contributes exactly one decode token
        - so every query row has the same query length S = 1
        - therefore Q rows can be stacked into one tensor: [B_active, H, 1, Dk]

        Why K/V stay ragged:
        - different active sequences can have different historical context lengths T_i
        - we explicitly avoid padding those contexts
        - so K/V are gathered independently for each row / sequence

        This mirrors the important serving intuition:
        - same execution batch
        - different logical attention domains per row
        """
        banner(f"RAGGED BATCHED DECODE ROUND {round_id}")
        print("This path creates one real batched query tensor with B_active > 1.")
        print("Each row corresponds to one active sequence in this scheduler round.")
        print("However, each row still gathers only its own per-sequence KV cache. No cross-sequence attention is allowed.")

        row_sequence_ids = list(decode_inputs.keys())
        B_active = len(row_sequence_ids)
        self._print_row_mapping(row_sequence_ids)

        q_rows = []
        ragged_k = []
        ragged_v = []

        # ----------------------------------------------------------
        # STEP 1: Build one query row per active sequence.
        # ----------------------------------------------------------
        for row_idx, sequence_id in enumerate(row_sequence_ids):
            x_token = decode_inputs[sequence_id]

            subsection(f"BUILD QUERY ROW {row_idx} FOR SEQUENCE {sequence_id}")
            print_tensor(
                f"Input decode token for row {row_idx}",
                x_token,
                f"This one-token input belongs to sequence {sequence_id}. Because decode processes one new token, the shape is [1, 1, D].",
                dim_names=["B_local", "S=1", "D"],
            )

            Q_raw, K_raw, V_raw = self.project_qkv(x_token)
            Q = self.split_heads(Q_raw, f"Q_decode_{sequence_id}")
            K_new = self.split_heads(K_raw, f"K_decode_{sequence_id}")
            V_new = self.split_heads(V_raw, f"V_decode_{sequence_id}")

            # Append current token's K/V into this sequence's cache before gather.
            self.cache.append_token(sequence_id, K_new[0, :, 0, :], V_new[0, :, 0, :])
            self.cache.dump_page_table(sequence_id)

            # Gather only this sequence's history. This is the key isolation property.
            K_gathered, V_gathered = self.cache.gather_active_kv(sequence_id)
            ragged_k.append(K_gathered)
            ragged_v.append(V_gathered)

            # Q has shape [1, H, 1, Dk]. Keep it as one batch row for now.
            q_rows.append(Q)

        # ----------------------------------------------------------
        # STEP 2: Stack all per-sequence Q rows into one batched Q.
        # ----------------------------------------------------------
        subsection("STACK ALL QUERY ROWS INTO ONE REAL BATCHED QUERY TENSOR")
        print("Each active sequence contributes exactly one query row, so these rows can be stacked along batch dimension.")
        print("Before stack, each Q row has shape [1, H, 1, Dk].")
        print(f"There are B_active = {B_active} rows in this scheduler round.")

        q_squeezed = [q[0] for q in q_rows]  # each becomes [H, 1, Dk]
        Q_batch = np.stack(q_squeezed, axis=0)  # [B_active, H, 1, Dk]
        print_tensor(
            "Batched decode query tensor Q_batch",
            Q_batch,
            "A real batched query tensor. Row 0 belongs to the first active sequence, row 1 to the second, and so on.",
            dim_names=["B_active", "H", "S=1", "Dk"],
        )
        print("Interpretation:")
        print("- B_active is the number of active sequences in this scheduler round")
        print("- each row is one different sequence / request")
        print("- all rows are batched together for execution efficiency")
        print("- but each row must still attend only to its own KV history")

        # ----------------------------------------------------------
        # STEP 3: Run attention row-by-row because K/V are ragged.
        # ----------------------------------------------------------
        context_rows = []
        row_outputs = {}

        for row_idx, sequence_id in enumerate(row_sequence_ids):
            subsection(f"ATTENTION FOR BATCH ROW {row_idx} -> SEQUENCE {sequence_id}")
            print(f"Batch row {row_idx} belongs to sequence {sequence_id}.")
            print("This row will use:")
            print("- Query = only row's own Q")
            print("- Context K/V = only this sequence's gathered cache")
            print("Therefore it cannot see any other row's history.")

            Q_row = Q_batch[row_idx:row_idx + 1]                # [1, H, 1, Dk]
            K_row = ragged_k[row_idx][None, :, :, :]            # [1, H, T_i, Dk]
            V_row = ragged_v[row_idx][None, :, :, :]            # [1, H, T_i, Dk]

            print_tensor(
                f"Q_row for batch row {row_idx}",
                Q_row,
                f"This is the query tensor for batch row {row_idx}, which corresponds to sequence {sequence_id}.",
                dim_names=["B_row=1", "H", "S=1", "Dk"],
            )
            print_tensor(
                f"K_row for batch row {row_idx}",
                K_row,
                f"This row's gathered key context for sequence {sequence_id}. Context length T may differ from other rows.",
                dim_names=["B_row=1", "H", "T_i", "Dk"],
            )
            print_tensor(
                f"V_row for batch row {row_idx}",
                V_row,
                f"This row's gathered value context for sequence {sequence_id}. Context length T may differ from other rows.",
                dim_names=["B_row=1", "H", "T_i", "Dk"],
            )

            context_row, scores_row, weights_row = scaled_dot_product_attention(
                Q=Q_row,
                K=K_row,
                V=V_row,
                use_causal_mask=False,
                phase_name=f"BATCH ROW {row_idx} / SEQUENCE {sequence_id} / ROUND {round_id}",
            )
            context_rows.append(context_row)

        # ----------------------------------------------------------
        # STEP 4: Rebuild a batched context tensor after per-row attention.
        # ----------------------------------------------------------
        subsection("REBUILD ONE BATCHED CONTEXT TENSOR FROM ALL ROW RESULTS")
        print("Each row-level attention call returned context of shape [1, H, 1, Dk].")
        print("Now we concatenate those row results along the batch dimension.")
        context_batch = np.concatenate(context_rows, axis=0)  # [B_active, H, 1, Dk]
        print_tensor(
            "Batched context tensor",
            context_batch,
            "This tensor contains attention outputs for all active rows in the same batch order as Q_batch.",
            dim_names=["B_active", "H", "S=1", "Dk"],
        )

        # ----------------------------------------------------------
        # STEP 5: Merge heads and apply one batched output projection.
        # ----------------------------------------------------------
        merged = self.merge_heads(context_batch)      # [B_active, 1, D]
        out = self.output_projection(merged)          # [B_active, 1, D]

        subsection("FINAL BATCHED OUTPUT ROW MAPPING")
        for row_idx, sequence_id in enumerate(row_sequence_ids):
            print(f"output row {row_idx} -> {sequence_id}")
            row_outputs[sequence_id] = out[row_idx:row_idx + 1]

        print_tensor(
            "Final batched decode output",
            out,
            "The output tensor now has a true batch dimension B_active. Row ownership is described in the mapping above.",
            dim_names=["B_active", "S=1", "D"],
        )

        banner(f"RAGGED BATCHED DECODE ROUND {round_id} COMPLETE")
        return out, row_outputs, row_sequence_ids

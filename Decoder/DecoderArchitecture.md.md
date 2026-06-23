## End-to-End Shape Tracing with KV Cache

Here is the precise, shape-by-shape breakdown of how a Decoder-Only model splits its execution into two distinct phases: the **Prefill Phase** (processing the prompt) and the **Decode Loop** (generating next tokens using KV Caching).

Let:
* $B$ = Batch size
* $p$ = Prompt length (number of input tokens)
* $h$ = Hidden dimension ($d_{model}$)
* $v$ = Vocabulary size
* $num\_heads$ = Number of attention heads
* $d_{head}$ = Dimension per head (where $num\_heads \times d_{head} = h$)
* $t$ = The current generation step index ($t = 1, 2, 3 \dots$)

---

### Phase 1: The Prefill Phase (Prompt Processing)
This happens **once** at the very beginning. The model processes the entire prompt of length $p$ simultaneously to populate the initial KV cache.

#### Step 1: Input Embedding & Positional Encoding
* **Input IDs Shape:** $(B, p)$
* **Embedding Lookup + RoPE:** Maps token IDs to vectors.
* **Output Shape:** $(B, p, h)$

#### Step 2: Inside the Layer — Core Attention & Cache Population
The tensor $X$ with shape $(B, p, h)$ enters the layer.

1. **Linear Projections:** Compute $Q, K, V$ matrices using weights $W_q, W_k, W_v$.
   * $Q, K, V$ **Shapes:** $(B, p, h)$
2. **Multi-Head Splitting & Transposition:**
   * Reshape and permute to isolate heads: $(B, num\_heads, p, d_{head})$
3. **CRITICAL STEP — Populate KV Cache:** The $K$ and $V$ tensors generated from the prompt are saved directly into memory.
   * **Stored Cache Shape:** $K_{cache}$ and $V_{cache}$ both have the shape $(B, num\_heads, p, d_{head})$.
4. **Causal Attention Execution:**
   * **MatMul ($QK^T$):** $(B, num\_heads, p, d_{head}) \times (B, num\_heads, d_{head}, p) \rightarrow (B, num\_heads, p, p)$
   * **Apply Causal Mask & Softmax:** Shape remains $(B, num\_heads, p, p)$.
   * **MatMul with V:** $(B, num\_heads, p, p) \times (B, num\_heads, p, d_{head}) \rightarrow (B, num\_heads, p, d_{head})$
5. **Concat & FFN Projection:**
   * Reshape back to hidden dimension: $(B, p, h)$
   * Feed-Forward network transformations bring it out of the layer with the final shape $(B, p, h)$.

#### Step 3: First Token Generation
* **Slice Last Token Hidden State:** Extract only the last prompt position: $(B, p, h) \rightarrow (B, 1, h)$
* **LM Head Projection:** $(B, 1, h) \times (h, v) \rightarrow (B, 1, v)$ (Logits for the very first generated token).
* **Sampling:** Yields 1 new token ID.

---

### Phase 2: The Decode Loop (Next-Token Generation with KV Cache)
Now, the autoregressive loop begins. For every new token generated, **the sequence length being processed is always exactly 1 ($s = 1$)**, completely bypassing redundant matrix multiplications for past tokens.
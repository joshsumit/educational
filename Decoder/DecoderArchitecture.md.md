# Decoder-Only Transformer: End-to-End Shape Tracing with KV Cache

This section traces every tensor shape through a modern decoder-only Transformer (GPT, LLaMA, Mistral, Qwen, Gemma, DeepSeek, etc.) and explains how KV caching makes autoregressive generation efficient.

---

# Symbols

Let:

- `B` = Batch size
- `P` = Prompt length
- `D` = Hidden dimension (`d_model`)
- `Vocab` = Vocabulary size
- `Dh` = Head dimension
- `L` = Number of decoder layers
- `t` = Current decode step

Additional symbols for modern attention variants:

- `Hq` = Number of Query heads
- `Hkv` = Number of Key/Value heads
- `G` = Query-to-KV head ratio

Relationships:

```text
D = Hq × Dh
G = Hq / Hkv
```

Example:

```text
D = 4096
Hq = 32
Hkv = 8
Dh = 128
G = 32 / 8 = 4

4096 = 32 × 128
```

---

# High-Level Execution

A decoder-only model executes in two phases.

## Phase 1: Prefill

The entire prompt is processed once.

Purpose:

1. Compute hidden states for all prompt tokens.
2. Create Keys and Values for every prompt position.
3. Populate the KV Cache.
4. Predict the first output token.

Computational complexity:

```text
O(P²)
```

because each prompt token attends to all previous tokens.

---

## Phase 2: Decode Loop

Generate one token at a time.

At every generation step:

```text
Sequence Length = 1
```

Only the newly generated token is processed.

All previously computed Keys and Values are reused from cache.

Computational complexity per step:

```text
O(P + t)
```

instead of recomputing the entire sequence.

---

# A Modern Decoder Layer

Most modern LLMs use a Pre-Norm architecture.

```text
Input X → RMSNorm → Self Attention → Residual Add → RMSNorm → SwiGLU FFN → Residual Add → Output
```

The following sections trace tensor shapes through this layer.

---

# PHASE 1 — PREFILL

---

## Step 1: Input Token IDs

Input:

```text
(B, P)
```

Example:

```text
(2, 128)
```

---

## Step 2: Embedding Lookup

Embedding table:

```text
(Vocab, D)
```

Lookup converts token IDs into vectors.

```text
(B, P) → (B, P, D)
```

Output:

```text
X = (B, P, D)
```

Example:

```text
(2, 128, 4096)
```

---

## Step 3: Positional Information

Modern decoder models typically use:

```text
RoPE (Rotary Positional Embeddings)
```

Important:

RoPE is not added to embeddings.

Instead, it is applied later to Queries and Keys.

Shape remains:

```text
(B, P, D)
```

---

# Self-Attention Block

---

## Step 4: RMSNorm

Input:

```text
(B, P, D)
```

Output:

```text
(B, P, D)
```

Shape does not change.

---

## Step 5: QKV Linear Projections

Weights:

```text
Wq = (D, Hq × Dh)
Wk = (D, Hkv × Dh)
Wv = (D, Hkv × Dh)
```

Compute:

```text
Q = XWq
K = XWk
V = XWv
```

Shapes:

### Queries

```text
(B, P, Hq × Dh)
```

### Keys

```text
(B, P, Hkv × Dh)
```

### Values

```text
(B, P, Hkv × Dh)
```

---

## Step 6: Split Into Heads

Queries:

```text
(B, P, Hq × Dh) → (B, P, Hq, Dh)
```

Keys:

```text
(B, P, Hkv × Dh) → (B, P, Hkv, Dh)
```

Values:

```text
(B, P, Hkv × Dh) → (B, P, Hkv, Dh)
```

---

## Step 7: Transpose Head Dimension Forward

Queries:

```text
(B, Hq, P, Dh)
```

Keys:

```text
(B, Hkv, P, Dh)
```

Values:

```text
(B, Hkv, P, Dh)
```

---

## Step 8: Apply RoPE

RoPE rotates Query and Key vectors.

Input:

```text
Q = (B, Hq, P, Dh)
K = (B, Hkv, P, Dh)
```

Output:

```text
Q = (B, Hq, P, Dh)
K = (B, Hkv, P, Dh)
```

Shape remains unchanged.

---

## Step 9: Populate KV Cache

Generated Keys and Values are stored.

### Key Cache

```text
(B, Hkv, P, Dh)
```

### Value Cache

```text
(B, Hkv, P, Dh)
```

Because modern decoder models often use GQA, only the KV heads are stored in cache.

This significantly reduces KV cache memory usage compared to storing one Key and Value head for every Query head.

---

## Step 10: Attention Score Computation

For clarity, prefill attention is easiest to understand in the case where attention is expressed with aligned Query and Key head groups.

Query shape:

```text
(B, Hq, P, Dh)
```

Key shape in memory:

```text
(B, Hkv, P, Dh)
```

Conceptually, each KV head is shared across multiple Query heads.

For explanatory purposes, we can view the Keys as logically expanded across Query groups:

```text
(B, Hkv, P, Dh) → (B, Hq, P, Dh)
```

Modern implementations usually avoid physically duplicating these tensors in memory.

Before matrix multiplication, the last two dimensions of Keys are transposed:

```text
(B, Hq, P, Dh) → (B, Hq, Dh, P)
```

Matrix multiplication:

```text
(B, Hq, P, Dh) × (B, Hq, Dh, P) → (B, Hq, P, P)
```

Result:

```text
Attention Scores = (B, Hq, P, P)
```

---

## Step 11: Scaling

Apply:

```text
Scores / √Dh
```

Shape:

```text
(B, Hq, P, P)
```

---

## Step 12: Causal Mask

Future positions are masked.

Shape remains:

```text
(B, Hq, P, P)
```

---

## Step 13: Softmax

Convert scores into probabilities.

```text
(B, Hq, P, P)
```

---

## Step 14: Attention × Values

Values are conceptually aligned with Query heads in the same way as Keys.

```text
(B, Hq, P, P) × (B, Hq, P, Dh) → (B, Hq, P, Dh)
```

Context output:

```text
(B, Hq, P, Dh)
```

---

## Step 15: Concatenate Heads

Transpose:

```text
(B, Hq, P, Dh) → (B, P, Hq, Dh)
```

Flatten heads:

```text
(B, P, Hq, Dh) → (B, P, D)
```

because:

```text
D = Hq × Dh
```

---

## Step 16: Output Projection

Output projection matrix:

```text
Wo = (D, D)
```

Computation:

```text
(B, P, D) × (D, D) → (B, P, D)
```

---

## Step 17: Residual Connection

Attention output is added to the block input.

```text
(B, P, D) + (B, P, D) → (B, P, D)
```

---

# Feed Forward Network (Modern SwiGLU)

Most modern decoder models no longer use the original Transformer MLP.

Instead they use a gated feed-forward network called SwiGLU.

Input:

```text
(B, P, D)
```

Weight matrices:

```text
W_gate = (D, Di)
W_up   = (D, Di)
W_down = (Di, D)
```

Where:

```text
Di ≈ 8D/3
```

Example:

```text
D = 4096
Di = 11008
```

---

## Step 18: RMSNorm

Input and output:

```text
(B, P, D)
```

---

## Step 19: Parallel Projections

Gate path:

```text
(B, P, D) → (B, P, Di)
```

Up path:

```text
(B, P, D) → (B, P, Di)
```

Outputs:

```text
Gate = (B, P, Di)
Up   = (B, P, Di)
```

---

## Step 20: SwiGLU Activation

Apply:

```text
SiLU(Gate) ⊙ Up
```

Result:

```text
(B, P, Di)
```

---

## Step 21: Down Projection

```text
(B, P, Di) × (Di, D) → (B, P, D)
```

---

## Step 22: Residual Connection

```text
(B, P, D) + (B, P, D) → (B, P, D)
```

Output of decoder layer:

```text
(B, P, D)
```

This becomes the input to the next layer.

---

# Final Layer Output

After all decoder layers:

```text
(B, P, D)
```

---

## Step 23: Extract Last Position

Only the final token position is needed for next-token prediction.

```text
(B, P, D) → (B, D)
```

---

## Step 24: LM Head Projection

Vocabulary projection:

```text
Wlm = (D, Vocab)
```

Computation:

```text
(B, D) × (D, Vocab) → (B, Vocab)
```

Output:

```text
Logits = (B, Vocab)
```

Example:

```text
(2, 32000)
```

---

## Step 25: Sampling

Apply:

```text
Softmax
ArgMax
Top-K
Top-P
Temperature
```

Output:

```text
(B, 1)
```

The first generated token is produced.

---

# PHASE 2 — DECODE LOOP

At this point the KV Cache already contains information from the prompt.

Current cache:

```text
(B, Hkv, P + t, Dh)
```

---

## Step 1: Process Only One New Token

Input:

```text
(B, 1)
```

Embedding:

```text
(B, 1) → (B, 1, D)
```

---

## Compute Q, K, V for the New Token

Using the new token hidden state (B, 1, D) and the learned projection matrices Wq, Wk, and Wv:

Queries:

```text
h = (B, 1, D)

Wq = (D, Hq × Dh)
Wk = (D, Hkv × Dh)
Wv = (D, Hkv × Dh)

Q = hWq
K = hWk
V = hWv

(B, 1, D) × (D, Hq × Dh)  → (B, 1, Hq × Dh)
(B, 1, D) × (D, Hkv × Dh) → (B, 1, Hkv × Dh)
(B, 1, D) × (D, Hkv × Dh) → (B, 1, Hkv × Dh)

Reshape and transpose:

Q → (B, Hq, 1, Dh)
K → (B, Hkv, 1, Dh)
V → (B, Hkv, 1, Dh)

```

---

## Step 3: Append To Cache

Previous cache:

```text
(B, Hkv, P + t, Dh)
```

Append new token:

```text
(B, Hkv, P + t, Dh) → (B, Hkv, P + t + 1, Dh)
```

Both Key and Value caches grow by one position.

---

## Step 4: Decode Attention

Query shape:

```text
(B, Hq, 1, Dh)
```

Key cache shape in memory:

```text
(B, Hkv, P + t + 1, Dh)
```

### GQA Expansion

Queries outnumber KV heads.

Example:

```text
Hq = 32
Hkv = 8
G = 4
```

Each KV head services multiple Query heads.

For explanatory purposes, Keys can be viewed as logically broadcast across Query groups:

```text
(B, Hkv, P + t + 1, Dh) → (B, Hq, P + t + 1, Dh)
```

Modern implementations typically avoid physically duplicating the tensors.

### Explicit Key Transpose

The Key tensor is stored in memory as:

```text
(B, Hq, P + t + 1, Dh)
```

Before matrix multiplication, the last two dimensions are transposed:

```text
(B, Hq, P + t + 1, Dh) → (B, Hq, Dh, P + t + 1)
```

### MatMul

```text
(B, Hq, 1, Dh) × (B, Hq, Dh, P + t + 1) → (B, Hq, 1, P + t + 1)
```

Result:

```text
Attention Scores = (B, Hq, 1, P + t + 1)
```

Only a single Query token performs attention against the entire cached sequence.

This is the primary optimization enabled by KV caching.

---

## Step 5: Attention Output

Values are conceptually aligned with Query heads in the same way as Keys.

```text
(B, Hq, 1, P + t + 1) × (B, Hq, P + t + 1, Dh) → (B, Hq, 1, Dh)
```

Continue through:

```text
Output Projection → Residual Connection → RMSNorm → SwiGLU FFN → Residual Connection → LM Head → Sampling
```

to generate the next token.

---

# GQA vs MHA vs MQA

## Multi-Head Attention (MHA)

```text
Q Heads = 32
K Heads = 32
V Heads = 32
```

Cache:

```text
(B, 32, SeqLen, Dh)
```

Largest memory footprint.

---

## Multi-Query Attention (MQA)

```text
Q Heads = 32
K Heads = 1
V Heads = 1
```

Cache:

```text
(B, 1, SeqLen, Dh)
```

Smallest memory footprint.

---

## Grouped Query Attention (GQA)

```text
Q Heads = 32
KV Heads = 8
```

Cache:

```text
(B, 8, SeqLen, Dh)
```

Most modern LLMs use GQA because it significantly reduces KV cache memory while maintaining quality.

---

# KV Cache Memory Growth

Approximate KV cache memory:

```text
KV_Memory ≈ 2 × L × Hkv × Dh × SeqLen × B × BytesPerValue
```

Where:

```text
2 = Keys + Values
```

Observations:

- Increasing context length increases cache size linearly.
- Increasing batch size increases cache size linearly.
- Increasing model size increases cache size linearly.
- Long-context inference is frequently constrained by KV cache memory rather than model weights.

---

# Why KV Cache Matters

Without KV Cache:

```text
Step 1: Process P tokens
Step 2: Process P + 1 tokens
Step 3: Process P + 2 tokens
...
```

The entire sequence is repeatedly recomputed.

---

With KV Cache:

```text
Prompt: Process P tokens once
Then: Process only 1 token at each generation step
```

The model reuses all previously computed Keys and Values.

This dramatically reduces latency and is the primary reason modern LLM inference is practical.

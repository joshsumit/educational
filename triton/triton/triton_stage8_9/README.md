# Triton From Zero To Expert — Stage 8 and Stage 9

This is the fifth downloadable implementation wave.

It adds the attention path that builds on the earlier matmul, masking, softmax, and normalization stages.

## Included stages

```text
stage08_attention_basics/
  00_attention_shapes.py
  01_qk_matmul_reference.py
  02_causal_mask_reference.py
  03_attention_reference.py
  04_qk_matmul_triton.py
  05_causal_softmax_triton.py
  06_attention_two_pass_triton.py
  07_attention_interview_notes.py

stage09_online_flashattention/
  00_online_softmax_derivation.py
  01_online_softmax_reference.py
  02_flashattention_v1_reference.py
  03_flashattention_v1_triton.py
  04_flashattention_memory_model.py
  05_flashattention_interview_notes.py
```

## Why this matters

These two stages move the repo from matmul kernels into Transformer kernels:

- QK^T score computation
- scaled dot-product attention
- causal masking
- softmax over attention scores
- two-pass attention kernels
- online softmax
- FlashAttention v1-style tiled attention
- memory traffic comparison
- interview-ready explanations

## Design rule

Every serious file includes:

```text
NumPy reference
CPU program-style simulation
real guarded Triton kernel where appropriate
wrapper
smoke test
interview/performance comments
```

## Important note

`stage08_attention_basics/06_attention_two_pass_triton.py` is intentionally a teaching implementation:

```text
QK Triton kernel -> materialized score matrix -> causal softmax Triton kernel -> PV matmul
```

It is useful for learning, but not memory optimal.

`stage09_online_flashattention/03_flashattention_v1_triton.py` introduces the memory-efficient direction:

```text
stream K/V tiles
maintain running max and denominator
avoid materializing full attention matrix
```

## Run CPU-safe tests

```bash
pip install -r requirements.txt
python run_all_smoke_tests.py
```

## Optional GPU/Triton execution

```bash
pip install torch triton
python -m stage08_attention_basics.04_qk_matmul_triton
python -m stage08_attention_basics.05_causal_softmax_triton
python -m stage08_attention_basics.06_attention_two_pass_triton
python -m stage09_online_flashattention.03_flashattention_v1_triton
```

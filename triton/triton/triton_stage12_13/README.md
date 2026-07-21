# Triton From Zero To Expert — Stage 12 and Stage 13

This is the seventh downloadable implementation wave.

It adds:

```text
Stage 12 — Quantized Matmul and KV Quantization
Stage 13 — Profiling and Benchmarking Discipline
```

These stages are high ROI for NVIDIA, AMD, Qualcomm, TensorRT-LLM, compiler/runtime, and edge-inference interviews.

## Included

```text
stage12_quantization_kernels/
  00_quantization_concepts.py
  01_int8_quantization_reference.py
  02_int8_matmul_reference.py
  03_int8_matmul_triton.py
  04_int4_packing_reference.py
  05_weight_only_int4_reference.py
  06_weight_only_int4_triton_skeleton.py
  07_kv_cache_quantization.py
  08_fp8_block_scaled_reference.py
  09_quantization_interview_notes.py

stage13_profiling_benchmarking/
  00_benchmarking_rules.py
  01_timer_harness.py
  02_gpu_timer_triton_stub.py
  03_roofline_model.py
  04_matmul_roofline.py
  05_attention_roofline.py
  06_decode_bandwidth_model.py
  07_profile_report_template.py
  08_profiling_interview_notes.py
```

## Run CPU-safe tests

```bash
pip install -r requirements.txt
python run_all_smoke_tests.py
```

Optional GPU/Triton execution:

```bash
pip install torch triton
python -m stage12_quantization_kernels.03_int8_matmul_triton
python -m stage12_quantization_kernels.06_weight_only_int4_triton_skeleton
python -m stage13_profiling_benchmarking.02_gpu_timer_triton_stub
```

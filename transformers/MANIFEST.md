# Repository Manifest

Actual implementation files included in this transformer-only repo:

- `README.md`
- `requirements.txt`
- `run_all_smoke_tests.py`
- `stage00_tensor_layout/__init__.py`
- `stage00_tensor_layout/tensor_layouts.py`
- `stage01_attention_basics/__init__.py`
- `stage01_attention_basics/attention_pipeline.py`
- `stage01_attention_basics/qk_matmul.py`
- `stage02_flash_attention/__init__.py`
- `stage02_flash_attention/flash_attention_reference.py`
- `stage02_flash_attention/flashattention_v2_partitioning.py`
- `stage02_flash_attention/online_softmax.py`
- `stage03_positional_embeddings/__init__.py`
- `stage03_positional_embeddings/rope_alibi.py`
- `stage04_kv_cache/__init__.py`
- `stage04_kv_cache/kv_cache.py`
- `stage05_paged_attention/__init__.py`
- `stage05_paged_attention/paged_kv_cache.py`
- `stage06_decode_runtime/__init__.py`
- `stage06_decode_runtime/decode_attention.py`
- `stage06_decode_runtime/sampling_and_scheduler.py`
- `stage07_parallelism/__init__.py`
- `stage07_parallelism/ring_attention.py`
- `stage07_parallelism/tensor_parallel_attention.py`
- `stage08_moe_runtime/__init__.py`
- `stage08_moe_runtime/moe_runtime.py`
- `stage09_backward/__init__.py`
- `stage09_backward/attention_backward.py`
- `stage10_performance/__init__.py`
- `stage10_performance/roofline_models.py`
- `stage11_end_to_end_runtime/__init__.py`
- `stage11_end_to_end_runtime/mini_llm_runtime.py`

"""
Runs tiny CPU smoke tests across the repo.

Dependency note:
    The implementation files use PyTorch because the target interview/runtime ecosystem
    usually discusses tensor shapes, reference parity, and fused kernels in PyTorch terms.
    Install dependencies first:
        pip install -r requirements.txt
"""
import importlib
import sys

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        'PyTorch is required for smoke tests. Install it with: pip install -r requirements.txt'
    ) from exc

TEST_MODULES = [
    'stage00_tensor_layout.tensor_layouts',
    'stage01_attention_basics.qk_matmul',
    'stage01_attention_basics.attention_pipeline',
    'stage02_flash_attention.online_softmax',
    'stage02_flash_attention.flash_attention_reference',
    'stage02_flash_attention.flashattention_v2_partitioning',
    'stage03_positional_embeddings.rope_alibi',
    'stage04_kv_cache.kv_cache',
    'stage05_paged_attention.paged_kv_cache',
    'stage06_decode_runtime.decode_attention',
    'stage06_decode_runtime.sampling_and_scheduler',
    'stage07_parallelism.tensor_parallel_attention',
    'stage07_parallelism.ring_attention',
    'stage08_moe_runtime.moe_runtime',
    'stage09_backward.attention_backward',
    'stage10_performance.roofline_models',
    'stage11_end_to_end_runtime.mini_llm_runtime',
]

for name in TEST_MODULES:
    m = importlib.import_module(name)
    if hasattr(m, 'smoke_test'):
        print(f'[RUN] {name}.smoke_test()')
        m.smoke_test()
print('[OK] all smoke tests passed')

from __future__ import annotations
import importlib

MODULES = [
    'stage06_matmul.00_matmul_shapes_and_strides',
    'stage06_matmul.01_naive_matmul_reference',
    'stage06_matmul.02_tiled_matmul_reference',
    'stage06_matmul.03_naive_matmul_triton',
    'stage06_matmul.04_tiled_matmul_triton',
    'stage06_matmul.05_grouped_matmul_triton',
    'stage06_matmul.06_matmul_interview_notes',
    'stage07_autotuning_batched.00_autotune_config_model',
    'stage07_autotuning_batched.01_autotuned_matmul_triton',
    'stage07_autotuning_batched.02_batched_matmul_reference',
    'stage07_autotuning_batched.03_batched_matmul_triton',
    'stage07_autotuning_batched.04_precision_and_accumulation',
    'stage07_autotuning_batched.05_matmul_benchmark_harness',
]

for name in MODULES:
    module = importlib.import_module(name)
    if hasattr(module, 'smoke_test'):
        print(f'[RUN] {name}.smoke_test()')
        module.smoke_test()
print('[OK] Stage 6 and Stage 7 CPU smoke tests passed')

from __future__ import annotations
import importlib

MODULES = [
    'stage04_reductions.00_sum_reduction_triton',
    'stage04_reductions.01_max_reduction_triton',
    'stage04_reductions.02_row_softmax_triton',
    'stage04_reductions.03_masked_softmax_triton',
    'stage04_reductions.04_reduction_interview_notes',
    'stage05_normalization.00_layernorm_triton',
    'stage05_normalization.01_rmsnorm_triton',
    'stage05_normalization.02_fused_rmsnorm_residual_triton',
    'stage05_normalization.03_normalization_memory_model',
]

for name in MODULES:
    module = importlib.import_module(name)
    if hasattr(module, 'smoke_test'):
        print(f'[RUN] {name}.smoke_test()')
        module.smoke_test()
print('[OK] Stage 4 and Stage 5 CPU smoke tests passed')

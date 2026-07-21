from __future__ import annotations
import importlib

MODULES = [
    'stage02_memory_and_masks.00_pointer_arithmetic',
    'stage02_memory_and_masks.01_masked_load_store',
    'stage02_memory_and_masks.02_strides_and_layouts',
    'stage02_memory_and_masks.03_coalescing_patterns',
    'stage03_elementwise.00_vector_add_triton',
    'stage03_elementwise.01_relu_triton',
    'stage03_elementwise.02_silu_triton',
    'stage03_elementwise.03_gelu_triton',
    'stage03_elementwise.04_scale_bias_triton',
    'stage03_elementwise.05_fused_affine_relu_triton',
]

for name in MODULES:
    module = importlib.import_module(name)
    if hasattr(module, 'smoke_test'):
        print(f'[RUN] {name}.smoke_test()')
        module.smoke_test()
print('[OK] Stage 2 and Stage 3 CPU smoke tests passed')

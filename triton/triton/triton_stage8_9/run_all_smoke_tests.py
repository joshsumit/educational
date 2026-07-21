from __future__ import annotations
import importlib

MODULES = [
    'stage08_attention_basics.00_attention_shapes',
    'stage08_attention_basics.01_qk_matmul_reference',
    'stage08_attention_basics.02_causal_mask_reference',
    'stage08_attention_basics.03_attention_reference',
    'stage08_attention_basics.04_qk_matmul_triton',
    'stage08_attention_basics.05_causal_softmax_triton',
    'stage08_attention_basics.06_attention_two_pass_triton',
    'stage08_attention_basics.07_attention_interview_notes',
    'stage09_online_flashattention.00_online_softmax_derivation',
    'stage09_online_flashattention.01_online_softmax_reference',
    'stage09_online_flashattention.02_flashattention_v1_reference',
    'stage09_online_flashattention.03_flashattention_v1_triton',
    'stage09_online_flashattention.04_flashattention_memory_model',
    'stage09_online_flashattention.05_flashattention_interview_notes',
]

for name in MODULES:
    module = importlib.import_module(name)
    if hasattr(module, 'smoke_test'):
        print(f'[RUN] {name}.smoke_test()')
        module.smoke_test()
print('[OK] Stage 8 and Stage 9 CPU smoke tests passed')

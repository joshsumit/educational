from __future__ import annotations
import importlib

MODULES = [
    'stage10_decode_kv_cache.00_prefill_vs_decode',
    'stage10_decode_kv_cache.01_kv_cache_layouts',
    'stage10_decode_kv_cache.02_decode_attention_reference',
    'stage10_decode_kv_cache.03_decode_attention_triton',
    'stage10_decode_kv_cache.04_multihead_decode_reference',
    'stage10_decode_kv_cache.05_kv_cache_memory_model',
    'stage10_decode_kv_cache.06_decode_interview_notes',
    'stage11_paged_attention.00_block_table_basics',
    'stage11_paged_attention.01_paged_kv_cache_layout',
    'stage11_paged_attention.02_paged_attention_reference',
    'stage11_paged_attention.03_paged_attention_triton_skeleton',
    'stage11_paged_attention.04_slot_mapping_and_append',
    'stage11_paged_attention.05_prefix_cache_simulator',
    'stage11_paged_attention.06_paged_attention_memory_model',
    'stage11_paged_attention.07_paged_attention_interview_notes',
]

for name in MODULES:
    module = importlib.import_module(name)
    if hasattr(module, 'smoke_test'):
        print(f'[RUN] {name}.smoke_test()')
        module.smoke_test()
print('[OK] Stage 10 and Stage 11 CPU smoke tests passed')

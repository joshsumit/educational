from __future__ import annotations
import importlib

MODULES = [
    'stage12_quantization_kernels.00_quantization_concepts',
    'stage12_quantization_kernels.01_int8_quantization_reference',
    'stage12_quantization_kernels.02_int8_matmul_reference',
    'stage12_quantization_kernels.03_int8_matmul_triton',
    'stage12_quantization_kernels.04_int4_packing_reference',
    'stage12_quantization_kernels.05_weight_only_int4_reference',
    'stage12_quantization_kernels.06_weight_only_int4_triton_skeleton',
    'stage12_quantization_kernels.07_kv_cache_quantization',
    'stage12_quantization_kernels.08_fp8_block_scaled_reference',
    'stage12_quantization_kernels.09_quantization_interview_notes',
    'stage13_profiling_benchmarking.00_benchmarking_rules',
    'stage13_profiling_benchmarking.01_timer_harness',
    'stage13_profiling_benchmarking.02_gpu_timer_triton_stub',
    'stage13_profiling_benchmarking.03_roofline_model',
    'stage13_profiling_benchmarking.04_matmul_roofline',
    'stage13_profiling_benchmarking.05_attention_roofline',
    'stage13_profiling_benchmarking.06_decode_bandwidth_model',
    'stage13_profiling_benchmarking.07_profile_report_template',
    'stage13_profiling_benchmarking.08_profiling_interview_notes',
]

for name in MODULES:
    module = importlib.import_module(name)
    if hasattr(module, 'smoke_test'):
        print(f'[RUN] {name}.smoke_test()')
        module.smoke_test()
print('[OK] Stage 12 and Stage 13 CPU smoke tests passed')

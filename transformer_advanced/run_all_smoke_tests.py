"""Run small CPU smoke tests across V2 advanced modules."""
import importlib

MODULES = [
    'stage00_cuda_execution_model.warp_block_memory',
    'stage01_tensorcore_programming.mma_sync_walkthrough',
    'stage01_tensorcore_programming.wmma_fragments',
    'stage01_tensorcore_programming.ldmatrix_explained',
    'stage02_async_memory_pipeline.cp_async_pipeline',
    'stage02_async_memory_pipeline.double_buffering',
    'stage03_cutlass_gemm_tiling.cutlass_style_gemm_tiling',
    'stage03_cutlass_gemm_tiling.splitk_gemm_reduction',
    'stage04_triton_attention_sim.triton_attention_kernel_simulation',
    'stage05_flashdecode.flashdecode_reference',
    'stage06_paged_scheduler.paged_attention_scheduler_deep_dive',
    'stage07_speculative_decoding.speculative_decoding_runtime',
    'stage08_quantized_runtime_kernels.int8_weight_only_gemm',
    'stage08_quantized_runtime_kernels.block_scaled_matmul',
    'stage09_end_to_end_advanced_runtime.advanced_runtime_simulator',
]

for name in MODULES:
    module = importlib.import_module(name)
    if hasattr(module, 'smoke_test'):
        print(f'[RUN] {name}.smoke_test()')
        module.smoke_test()
print('[OK] V2 advanced smoke tests passed')

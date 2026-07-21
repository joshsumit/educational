import importlib
MODULES = [
 'stage00_environment.optional_imports',
 'stage01_programming_model.program_id_grid',
 'stage01_programming_model.block_vectors',
 'stage02_memory_and_masks.pointer_arithmetic',
 'stage02_memory_and_masks.masked_load_store',
 'stage03_elementwise.vector_add',
 'stage03_elementwise.fused_affine_relu',
 'stage04_reductions.block_reductions',
 'stage04_reductions.softmax_layernorm',
 'stage05_matmul.matmul_reference_tiling',
 'stage06_fusions.fused_bias_gelu',
 'stage07_autotuning.autotune_simulator',
 'stage08_debugging.parity_testing',
 'stage09_transformer_kernels.qk_attention_pipeline',
 'stage09_transformer_kernels.decode_attention_kernel',
 'stage10_project_capstones.mini_triton_kernel_library',
]
for name in MODULES:
    m=importlib.import_module(name)
    if hasattr(m,'smoke_test'):
        print(f'[RUN] {name}.smoke_test()')
        m.smoke_test()
print('[OK] Triton from zero smoke tests passed')

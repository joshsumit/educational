"""Run all CPU-safe smoke tests for Stage 0 and Stage 1.

This runner intentionally does not require Triton or a GPU. The goal of the first
downloadable wave is to make the execution model executable everywhere.
"""
from __future__ import annotations

import importlib

MODULES = [
    'stage00_environment.01_gpu_execution_model',
    'stage00_environment.02_triton_vs_cuda',
    'stage00_environment.03_optional_imports',
    'stage00_environment.04_first_kernel_anatomy',
    'stage00_environment.05_common_triton_primitives',
    'stage01_programming_model.00_program_ids',
    'stage01_programming_model.01_block_vectors',
    'stage01_programming_model.02_masks',
    'stage01_programming_model.03_grid_mapping',
    'stage01_programming_model.04_2d_grids',
    'stage01_programming_model.05_3d_grids',
    'stage01_programming_model.06_program_ordering',
]

def main() -> None:
    for name in MODULES:
        module = importlib.import_module(name)
        if hasattr(module, 'smoke_test'):
            print(f'[RUN] {name}.smoke_test()')
            module.smoke_test()
    print('[OK] Stage 0 and Stage 1 smoke tests passed')

if __name__ == '__main__':
    main()

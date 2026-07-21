"""Real Triton vector-add kernel. Requires Triton + Torch + GPU to run."""
try:
    import torch
    import triton
    import triton.language as tl
except Exception:  # repo remains importable CPU-only
    torch = triton = tl = None

if tl is not None:
    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, x + y, mask=mask)

def run_vector_add(x, y, block=1024):
    if triton is None:
        raise RuntimeError('Triton/Torch not available')
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), block),)
    vector_add_kernel[grid](x, y, out, x.numel(), BLOCK=block)
    return out

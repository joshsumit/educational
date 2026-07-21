"""Real Triton matmul kernel skeleton with grouped program ordering."""
try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = triton = tl = None

if tl is not None:
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M:tl.constexpr, N:tl.constexpr, K:tl.constexpr,
                      stride_am:tl.constexpr, stride_ak:tl.constexpr, stride_bk:tl.constexpr, stride_bn:tl.constexpr,
                      stride_cm:tl.constexpr, stride_cn:tl.constexpr,
                      BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr, BLOCK_K:tl.constexpr, GROUP_M:tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + offs_m[:,None]*stride_am + (k0+offs_k[None,:])*stride_ak, mask=(offs_m[:,None]<M) & (k0+offs_k[None,:]<K), other=0.0)
            b = tl.load(b_ptr + (k0+offs_k[:,None])*stride_bk + offs_n[None,:]*stride_bn, mask=(k0+offs_k[:,None]<K) & (offs_n[None,:]<N), other=0.0)
            acc += tl.dot(a,b)
        tl.store(c_ptr + offs_m[:,None]*stride_cm + offs_n[None,:]*stride_cn, acc, mask=(offs_m[:,None]<M) & (offs_n[None,:]<N))

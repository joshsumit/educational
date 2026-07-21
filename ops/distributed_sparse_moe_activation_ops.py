from typing import Callable

import torch


# -----------------------------
# Tier 8: Distributed training primitives
# -----------------------------

def ring_allreduce(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    # Ring AllReduce simulation: every rank receives the summed tensor.
    total = torch.zeros_like(tensors[0])
    for t in tensors:
        total = total + t
    return [total.clone() for _ in tensors]


def tree_allreduce(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    # Tree AllReduce simulation by recursive pairwise sum.
    work = [t.clone() for t in tensors]
    while len(work) > 1:
        nxt = []
        for i in range(0, len(work), 2):
            if i + 1 < len(work):
                nxt.append(work[i] + work[i + 1])
            else:
                nxt.append(work[i])
        work = nxt
    root = work[0]
    return [root.clone() for _ in tensors]


def reduce_scatter(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    # ReduceScatter simulation: sum tensors then shard result across ranks on dim 0.
    total = torch.zeros_like(tensors[0])
    for t in tensors:
        total += t
    return list(torch.chunk(total, len(tensors), dim=0))


def all_gather(shards: list[torch.Tensor]) -> list[torch.Tensor]:
    # AllGather simulation: concat all shards and return full tensor to every rank.
    full = torch.cat(shards, dim=0)
    return [full.clone() for _ in shards]


def broadcast(src: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    # Broadcast simulation from one source tensor.
    return [src.clone() for _ in range(world_size)]


def all_to_all(rank_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    # AllToAll simulation.
    # Each rank input is split into world_size chunks along dim 0 and exchanged.
    world = len(rank_inputs)
    chunks_per_rank = [list(torch.chunk(t, world, dim=0)) for t in rank_inputs]
    outputs = []
    for dst in range(world):
        gathered = [chunks_per_rank[src][dst] for src in range(world)]
        outputs.append(torch.cat(gathered, dim=0))
    return outputs


def data_parallel_simulation(gradients_per_rank: list[torch.Tensor]) -> list[torch.Tensor]:
    # Data parallel gradient averaging simulation.
    summed = ring_allreduce(gradients_per_rank)[0]
    avg = summed / len(gradients_per_rank)
    return [avg.clone() for _ in gradients_per_rank]


def tensor_parallel_simulation(x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
    # Tensor parallel simulation: split output channels across rank-local weight shards.
    # x shape [B, D], each weight shard [D, D_out_rank].
    parts = [x @ w for w in weights]
    return torch.cat(parts, dim=-1)


def pipeline_parallel_simulation(x: torch.Tensor, stages: list[Callable[[torch.Tensor], torch.Tensor]]) -> torch.Tensor:
    # Pipeline parallel forward simulation.
    y = x
    for stage in stages:
        y = stage(y)
    return y


def onef_oneb_schedule(num_microbatches: int, num_stages: int) -> list[tuple[int, int, str]]:
    # 1F1B scheduling simulation trace.
    # Returns tuples: (time, stage, action) where action is Fwd or Bwd.
    timeline: list[tuple[int, int, str]] = []
    t = 0

    # Warmup forwards.
    for mb in range(min(num_stages, num_microbatches)):
        for s in range(mb + 1):
            timeline.append((t, s, f"Fwd(mb={mb - s})"))
            t += 1

    # Simplified steady-state interleaving.
    for mb in range(num_microbatches):
        for s in reversed(range(num_stages)):
            timeline.append((t, s, f"Bwd(mb={mb})"))
            t += 1
    return timeline


def gpipe_schedule(num_microbatches: int, num_stages: int) -> list[tuple[int, int, str]]:
    # GPipe scheduling simulation: all forwards then all backwards.
    timeline: list[tuple[int, int, str]] = []
    t = 0

    for mb in range(num_microbatches):
        for s in range(num_stages):
            timeline.append((t, s, f"Fwd(mb={mb})"))
            t += 1

    for mb in reversed(range(num_microbatches)):
        for s in reversed(range(num_stages)):
            timeline.append((t, s, f"Bwd(mb={mb})"))
            t += 1
    return timeline


# -----------------------------
# Tier 9: Sparse computation
# -----------------------------

def coo_sparse_format(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # COO conversion: return (indices[2,nnz], values[nnz]) for 2D matrix.
    nz = dense.nonzero(as_tuple=False).transpose(0, 1)
    vals = dense[dense != 0]
    return nz, vals


def csr_sparse_format(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # CSR conversion: row_ptr, col_idx, values.
    if dense.dim() != 2:
        raise ValueError("CSR conversion expects a 2D matrix")

    rows, cols = dense.shape
    row_ptr = [0]
    col_idx = []
    vals = []

    nnz = 0
    for r in range(rows):
        for c in range(cols):
            v = dense[r, c]
            if float(v) != 0.0:
                col_idx.append(c)
                vals.append(v)
                nnz += 1
        row_ptr.append(nnz)

    return (
        torch.tensor(row_ptr, dtype=torch.int64, device=dense.device),
        torch.tensor(col_idx, dtype=torch.int64, device=dense.device),
        torch.tensor(vals, dtype=dense.dtype, device=dense.device),
    )


def sparse_matrix_multiply(a_sparse: torch.Tensor, b_sparse: torch.Tensor) -> torch.Tensor:
    # Sparse-sparse matmul by densifying second operand for simplicity.
    return torch.sparse.mm(a_sparse, b_sparse.to_dense()).to_sparse()


def sparse_dense_gemm(a_sparse: torch.Tensor, b_dense: torch.Tensor) -> torch.Tensor:
    # Sparse-dense GEMM path.
    return torch.sparse.mm(a_sparse, b_dense)


def block_sparse_gemm(a: torch.Tensor, b: torch.Tensor, block_mask: torch.Tensor, block_size: int) -> torch.Tensor:
    # Block-sparse GEMM simulation.
    # block_mask shape [Mb, Nb] controls whether output blocks are computed.
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("Inner dimensions mismatch")

    mb = (m + block_size - 1) // block_size
    nb = (n + block_size - 1) // block_size
    if block_mask.shape != (mb, nb):
        raise ValueError("block_mask shape mismatch")

    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for bi in range(mb):
        for bj in range(nb):
            if int(block_mask[bi, bj]) == 0:
                continue
            i0 = bi * block_size
            j0 = bj * block_size
            i1 = min(i0 + block_size, m)
            j1 = min(j0 + block_size, n)
            out[i0:i1, j0:j1] = a[i0:i1] @ b[:, j0:j1]
    return out


# -----------------------------
# Tier 10: Mixture-of-Experts
# -----------------------------

def token_routing(router_logits: torch.Tensor) -> torch.Tensor:
    # Token routing probabilities for [T, E] logits.
    return torch.softmax(router_logits, dim=-1)


def top1_routing(router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Top-1 expert assignment.
    probs = torch.softmax(router_logits, dim=-1)
    score, expert = torch.max(probs, dim=-1)
    return expert, score


def top2_routing(router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Top-2 expert assignment.
    probs = torch.softmax(router_logits, dim=-1)
    score, expert = torch.topk(probs, k=2, dim=-1)
    return expert, score


def expert_capacity_handling(expert_ids: torch.Tensor, scores: torch.Tensor, num_experts: int, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Enforce expert capacity by keeping highest-score tokens per expert.
    keep_mask = torch.zeros_like(expert_ids, dtype=torch.bool)

    for e in range(num_experts):
        idx = torch.where(expert_ids == e)[0]
        if idx.numel() == 0:
            continue
        s = scores[idx]
        top = torch.argsort(s, descending=True)[:capacity]
        keep_mask[idx[top]] = True

    return expert_ids[keep_mask], scores[keep_mask]


def load_balancing_loss_reference(router_probs: torch.Tensor) -> torch.Tensor:
    # Simple load-balancing loss reference.
    # Encourage equal token probability mass across experts.
    expert_mean = torch.mean(router_probs, dim=0)
    uniform = torch.full_like(expert_mean, 1.0 / expert_mean.numel())
    return torch.sum((expert_mean - uniform) ** 2)


def dispatch_tokens_to_experts(tokens: torch.Tensor, expert_ids: torch.Tensor, num_experts: int) -> list[torch.Tensor]:
    # Dispatch token vectors [T, D] into per-expert lists.
    expert_batches = []
    for e in range(num_experts):
        mask = expert_ids == e
        expert_batches.append(tokens[mask])
    return expert_batches


def combine_expert_outputs(expert_outputs: list[torch.Tensor], expert_ids: torch.Tensor, num_tokens: int, hidden_dim: int) -> torch.Tensor:
    # Combine expert outputs back to token order.
    out = torch.zeros((num_tokens, hidden_dim), dtype=expert_outputs[0].dtype, device=expert_outputs[0].device)
    offsets = [0 for _ in range(len(expert_outputs))]

    for t in range(num_tokens):
        e = int(expert_ids[t])
        out[t] = expert_outputs[e][offsets[e]]
        offsets[e] += 1
    return out


# -----------------------------
# Tier 11: Missing activation functions
# -----------------------------

def relu(x: torch.Tensor) -> torch.Tensor:
    # ReLU activation.
    return torch.clamp(x, min=0)


def relu6(x: torch.Tensor) -> torch.Tensor:
    # ReLU6 activation.
    return torch.clamp(x, min=0, max=6)


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    # LeakyReLU activation.
    return torch.where(x >= 0, x, x * negative_slope)


def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    # ELU activation.
    return torch.where(x >= 0, x, alpha * (torch.exp(x) - 1.0))


def selu(x: torch.Tensor) -> torch.Tensor:
    # SELU activation constants.
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    return scale * elu(x, alpha=alpha)


def mish(x: torch.Tensor) -> torch.Tensor:
    # Mish activation: x * tanh(softplus(x)).
    return x * torch.tanh(torch.nn.functional.softplus(x))


def hard_swish(x: torch.Tensor) -> torch.Tensor:
    # HardSwish activation: x * relu6(x+3)/6.
    return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # HardSigmoid activation.
    return torch.clamp((x + 3.0) / 6.0, min=0.0, max=1.0)


def softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    # Softplus activation.
    return torch.nn.functional.softplus(x, beta=beta, threshold=threshold)


# -----------------------------
# Practice Questions
# -----------------------------
# Beginner:
# 1) What is the difference between ring_allreduce and tree_allreduce?
# 2) Why are sparse formats like COO/CSR useful for memory savings?
# 3) How does top-1 routing differ from top-2 routing in MoE?
# Intermediate:
# 4) Compare data parallel, tensor parallel, and pipeline parallel tradeoffs.
# 5) How does expert capacity handling prevent out-of-memory failures?
# 6) Why can block sparse GEMM improve throughput for structured sparsity?
# Advanced:
# 7) How would you design an all_to_all schedule to minimize network hot spots?
# 8) What failure modes cause expert collapse, and how does load balancing help?
# 9) Compare activation choices (SELU/Mish/HardSwish) for training stability and inference speed.
# Expert:
# 10) Propose a hybrid parallelism plan for trillion-parameter MoE training.
# 11) How would you overlap communication and compute in reduce_scatter + all_gather steps?
# 12) Design benchmarks for sparse and MoE kernels with realistic token routing skew.

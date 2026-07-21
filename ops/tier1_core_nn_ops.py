import math
import torch
import torch.nn.functional as F


# -----------------------------
# Convolution operators
# -----------------------------

def conv1d_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    # Naive Conv1D reference for learning.
    # Shapes: x[B, Cin, L], weight[Cout, Cin/groups, K].
    bsz, c_in, l_in = x.shape
    c_out, c_per_group, k = weight.shape

    if c_in % groups != 0 or c_out % groups != 0:
        raise ValueError("Invalid groups for Conv1D")

    x_pad = F.pad(x, (padding, padding))
    l_out = (l_in + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    out = torch.zeros((bsz, c_out, l_out), dtype=x.dtype, device=x.device)

    in_group = c_in // groups
    out_group = c_out // groups

    for b in range(bsz):
        for g in range(groups):
            in_start = g * in_group
            out_start = g * out_group
            for oc in range(out_group):
                oc_global = out_start + oc
                for i in range(l_out):
                    acc = 0.0
                    for ic in range(c_per_group):
                        ic_global = in_start + ic
                        for kk in range(k):
                            in_idx = i * stride + kk * dilation
                            acc += float(x_pad[b, ic_global, in_idx]) * float(weight[oc_global, ic, kk])
                    if bias is not None:
                        acc += float(bias[oc_global])
                    out[b, oc_global, i] = acc
    return out


def conv1d_optimized(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    # Optimized path delegates to highly tuned backend kernels.
    return F.conv1d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


def conv2d_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    # Naive Conv2D reference for learning loop order and shape math.
    # Shapes: x[B, Cin, H, W], weight[Cout, Cin/groups, Kh, Kw].
    bsz, c_in, h_in, w_in = x.shape
    c_out, c_per_group, kh, kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    if c_in % groups != 0 or c_out % groups != 0:
        raise ValueError("Invalid groups for Conv2D")

    x_pad = F.pad(x, (pw, pw, ph, ph))
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = torch.zeros((bsz, c_out, h_out, w_out), dtype=x.dtype, device=x.device)

    in_group = c_in // groups
    out_group = c_out // groups

    for b in range(bsz):
        for g in range(groups):
            in_start = g * in_group
            out_start = g * out_group
            for oc in range(out_group):
                oc_global = out_start + oc
                for oh in range(h_out):
                    for ow in range(w_out):
                        acc = 0.0
                        for ic in range(c_per_group):
                            ic_global = in_start + ic
                            for kh_i in range(kh):
                                for kw_i in range(kw):
                                    ih = oh * sh + kh_i * dh
                                    iw = ow * sw + kw_i * dw
                                    acc += float(x_pad[b, ic_global, ih, iw]) * float(weight[oc_global, ic, kh_i, kw_i])
                        if bias is not None:
                            acc += float(bias[oc_global])
                        out[b, oc_global, oh, ow] = acc
    return out


def conv2d_optimized(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    # Optimized path through backend kernel libraries (cuDNN/oneDNN/etc.).
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


def conv3d_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> torch.Tensor:
    # Naive Conv3D reference.
    # Shapes: x[B, Cin, D, H, W], weight[Cout, Cin/groups, Kd, Kh, Kw].
    bsz, c_in, d_in, h_in, w_in = x.shape
    c_out, c_per_group, kd, kh, kw = weight.shape
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation

    if c_in % groups != 0 or c_out % groups != 0:
        raise ValueError("Invalid groups for Conv3D")

    x_pad = F.pad(x, (pw, pw, ph, ph, pd, pd))
    d_out = (d_in + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    out = torch.zeros((bsz, c_out, d_out, h_out, w_out), dtype=x.dtype, device=x.device)
    in_group = c_in // groups
    out_group = c_out // groups

    for b in range(bsz):
        for g in range(groups):
            in_start = g * in_group
            out_start = g * out_group
            for oc in range(out_group):
                oc_global = out_start + oc
                for od in range(d_out):
                    for oh in range(h_out):
                        for ow in range(w_out):
                            acc = 0.0
                            for ic in range(c_per_group):
                                ic_global = in_start + ic
                                for kd_i in range(kd):
                                    for kh_i in range(kh):
                                        for kw_i in range(kw):
                                            id_i = od * sd + kd_i * dd
                                            ih_i = oh * sh + kh_i * dh
                                            iw_i = ow * sw + kw_i * dw
                                            acc += float(x_pad[b, ic_global, id_i, ih_i, iw_i]) * float(
                                                weight[oc_global, ic, kd_i, kh_i, kw_i]
                                            )
                            if bias is not None:
                                acc += float(bias[oc_global])
                            out[b, oc_global, od, oh, ow] = acc
    return out


def conv3d_optimized(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> torch.Tensor:
    # Optimized path through backend Conv3D kernel.
    return F.conv3d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


def strided_conv2d_naive(x: torch.Tensor, weight: torch.Tensor, stride: tuple[int, int]) -> torch.Tensor:
    # Explicit strided Conv2D helper using naive kernel.
    return conv2d_naive(x, weight, stride=stride)


def dilated_conv2d_naive(x: torch.Tensor, weight: torch.Tensor, dilation: tuple[int, int]) -> torch.Tensor:
    # Explicit dilated Conv2D helper using naive kernel.
    return conv2d_naive(x, weight, dilation=dilation)


def grouped_conv2d_naive(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    # Explicit grouped Conv2D helper using naive kernel.
    return conv2d_naive(x, weight, groups=groups)


def depthwise_conv2d_naive(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    # Depthwise Conv2D is grouped convolution with groups == in_channels.
    return conv2d_naive(x, weight, bias=bias, groups=x.shape[1])


def depthwise_conv2d_optimized(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    # Optimized depthwise Conv2D uses backend grouped conv implementation.
    return conv2d_optimized(x, weight, bias=bias, groups=x.shape[1])


def depthwise_separable_conv2d_naive(
    x: torch.Tensor,
    depthwise_weight: torch.Tensor,
    pointwise_weight: torch.Tensor,
    pointwise_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # Depthwise-separable Conv2D = depthwise conv then 1x1 pointwise conv.
    dw = depthwise_conv2d_naive(x, depthwise_weight)
    return conv2d_naive(dw, pointwise_weight, bias=pointwise_bias)


def depthwise_separable_conv2d_optimized(
    x: torch.Tensor,
    depthwise_weight: torch.Tensor,
    pointwise_weight: torch.Tensor,
    pointwise_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # Optimized depthwise-separable Conv2D via optimized depthwise + optimized 1x1 conv.
    dw = depthwise_conv2d_optimized(x, depthwise_weight)
    return conv2d_optimized(dw, pointwise_weight, bias=pointwise_bias)


def conv_transpose2d_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    # Naive transposed conv (deconvolution) implemented by scattering input contributions.
    # Shapes: x[B, Cin, H, W], weight[Cin, Cout, Kh, Kw].
    bsz, c_in, h_in, w_in = x.shape
    c_in_w, c_out, kh, kw = weight.shape
    if c_in_w != c_in:
        raise ValueError("weight first dim must match input channels")

    sh, sw = stride
    ph, pw = padding

    h_out = (h_in - 1) * sh - 2 * ph + kh
    w_out = (w_in - 1) * sw - 2 * pw + kw
    out = torch.zeros((bsz, c_out, h_out, w_out), dtype=x.dtype, device=x.device)

    for b in range(bsz):
        for ic in range(c_in):
            for ih in range(h_in):
                for iw in range(w_in):
                    in_val = float(x[b, ic, ih, iw])
                    base_h = ih * sh - ph
                    base_w = iw * sw - pw
                    for oc in range(c_out):
                        for kh_i in range(kh):
                            for kw_i in range(kw):
                                oh = base_h + kh_i
                                ow = base_w + kw_i
                                if 0 <= oh < h_out and 0 <= ow < w_out:
                                    out[b, oc, oh, ow] += in_val * float(weight[ic, oc, kh_i, kw_i])

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def conv_transpose2d_optimized(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    # Optimized transposed conv path.
    return F.conv_transpose2d(x, weight, bias=bias, stride=stride, padding=padding)


def im2col_naive(
    x: torch.Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    # Naive im2col for Conv2D lowering.
    bsz, c_in, h_in, w_in = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    x_pad = F.pad(x, (pw, pw, ph, ph))
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    cols = torch.zeros((bsz, c_in * kh * kw, h_out * w_out), dtype=x.dtype, device=x.device)
    for b in range(bsz):
        col_idx = 0
        for oh in range(h_out):
            for ow in range(w_out):
                row_idx = 0
                for ic in range(c_in):
                    for kh_i in range(kh):
                        for kw_i in range(kw):
                            ih = oh * sh + kh_i * dh
                            iw = ow * sw + kw_i * dw
                            cols[b, row_idx, col_idx] = x_pad[b, ic, ih, iw]
                            row_idx += 1
                col_idx += 1
    return cols


def im2col_optimized(
    x: torch.Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    # Optimized lowering via torch.unfold.
    return F.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)


def col2im_naive(
    cols: torch.Tensor,
    output_size: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    channels: int = 1,
) -> torch.Tensor:
    # Naive col2im inverse lowering (overlap-add).
    bsz, _, n_cols = cols.shape
    h_out, w_out = output_size
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    h_pad = h_out + 2 * ph
    w_pad = w_out + 2 * pw
    x_pad = torch.zeros((bsz, channels, h_pad, w_pad), dtype=cols.dtype, device=cols.device)

    out_h = (h_out + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (w_out + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if out_h * out_w != n_cols:
        raise ValueError("Column count mismatch for output size/kernel/stride/padding/dilation")

    for b in range(bsz):
        col_idx = 0
        for oh in range(out_h):
            for ow in range(out_w):
                row_idx = 0
                for ic in range(channels):
                    for kh_i in range(kh):
                        for kw_i in range(kw):
                            ih = oh * sh + kh_i * dh
                            iw = ow * sw + kw_i * dw
                            x_pad[b, ic, ih, iw] += cols[b, row_idx, col_idx]
                            row_idx += 1
                col_idx += 1

    return x_pad[:, :, ph : ph + h_out, pw : pw + w_out]


def col2im_optimized(
    cols: torch.Tensor,
    output_size: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    # Optimized fold path via torch.fold.
    return F.fold(cols, output_size=output_size, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)


def winograd_conv2d_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # Winograd reference placeholder for educational parity.
    # For correctness-first baseline, delegate to Conv2D.
    # The real Winograd path would transform input/kernel/output tiles.
    return conv2d_optimized(x, weight, bias=bias)


# -----------------------------
# Pooling operators
# -----------------------------

def maxpool1d_naive(x: torch.Tensor, kernel_size: int, stride: int | None = None) -> torch.Tensor:
    # Naive MaxPool1D.
    if stride is None:
        stride = kernel_size
    bsz, channels, length = x.shape
    out_len = (length - kernel_size) // stride + 1
    out = torch.empty((bsz, channels, out_len), dtype=x.dtype, device=x.device)

    for b in range(bsz):
        for c in range(channels):
            for i in range(out_len):
                start = i * stride
                out[b, c, i] = torch.max(x[b, c, start : start + kernel_size])
    return out


def maxpool1d_optimized(x: torch.Tensor, kernel_size: int, stride: int | None = None) -> torch.Tensor:
    # Optimized MaxPool1D path.
    return F.max_pool1d(x, kernel_size=kernel_size, stride=stride)


def maxpool2d_naive(x: torch.Tensor, kernel_size: tuple[int, int], stride: tuple[int, int] | None = None) -> torch.Tensor:
    # Naive MaxPool2D.
    if stride is None:
        stride = kernel_size
    kh, kw = kernel_size
    sh, sw = stride
    bsz, channels, h_in, w_in = x.shape
    h_out = (h_in - kh) // sh + 1
    w_out = (w_in - kw) // sw + 1
    out = torch.empty((bsz, channels, h_out, w_out), dtype=x.dtype, device=x.device)

    for b in range(bsz):
        for c in range(channels):
            for oh in range(h_out):
                for ow in range(w_out):
                    hs = oh * sh
                    ws = ow * sw
                    out[b, c, oh, ow] = torch.max(x[b, c, hs : hs + kh, ws : ws + kw])
    return out


def maxpool2d_optimized(x: torch.Tensor, kernel_size: tuple[int, int], stride: tuple[int, int] | None = None) -> torch.Tensor:
    # Optimized MaxPool2D path.
    return F.max_pool2d(x, kernel_size=kernel_size, stride=stride)


def avgpool1d_naive(x: torch.Tensor, kernel_size: int, stride: int | None = None) -> torch.Tensor:
    # Naive AvgPool1D.
    if stride is None:
        stride = kernel_size
    bsz, channels, length = x.shape
    out_len = (length - kernel_size) // stride + 1
    out = torch.empty((bsz, channels, out_len), dtype=x.dtype, device=x.device)

    for b in range(bsz):
        for c in range(channels):
            for i in range(out_len):
                start = i * stride
                out[b, c, i] = torch.mean(x[b, c, start : start + kernel_size])
    return out


def avgpool1d_optimized(x: torch.Tensor, kernel_size: int, stride: int | None = None) -> torch.Tensor:
    # Optimized AvgPool1D path.
    return F.avg_pool1d(x, kernel_size=kernel_size, stride=stride)


def avgpool2d_naive(x: torch.Tensor, kernel_size: tuple[int, int], stride: tuple[int, int] | None = None) -> torch.Tensor:
    # Naive AvgPool2D.
    if stride is None:
        stride = kernel_size
    kh, kw = kernel_size
    sh, sw = stride
    bsz, channels, h_in, w_in = x.shape
    h_out = (h_in - kh) // sh + 1
    w_out = (w_in - kw) // sw + 1
    out = torch.empty((bsz, channels, h_out, w_out), dtype=x.dtype, device=x.device)

    for b in range(bsz):
        for c in range(channels):
            for oh in range(h_out):
                for ow in range(w_out):
                    hs = oh * sh
                    ws = ow * sw
                    out[b, c, oh, ow] = torch.mean(x[b, c, hs : hs + kh, ws : ws + kw])
    return out


def avgpool2d_optimized(x: torch.Tensor, kernel_size: tuple[int, int], stride: tuple[int, int] | None = None) -> torch.Tensor:
    # Optimized AvgPool2D path.
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)


def global_avg_pool2d_naive(x: torch.Tensor) -> torch.Tensor:
    # Naive global average pooling for [B, C, H, W] -> [B, C, 1, 1].
    bsz, channels, _, _ = x.shape
    out = torch.zeros((bsz, channels, 1, 1), dtype=x.dtype, device=x.device)
    for b in range(bsz):
        for c in range(channels):
            out[b, c, 0, 0] = torch.mean(x[b, c])
    return out


def global_avg_pool2d_optimized(x: torch.Tensor) -> torch.Tensor:
    # Optimized global average pooling via adaptive pool to 1x1.
    return F.adaptive_avg_pool2d(x, output_size=(1, 1))


def adaptive_avg_pool2d_naive(x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    # Naive adaptive average pooling by mapping each output bin to an input range.
    bsz, channels, h_in, w_in = x.shape
    h_out, w_out = output_size
    out = torch.empty((bsz, channels, h_out, w_out), dtype=x.dtype, device=x.device)

    for oh in range(h_out):
        h_start = int(math.floor(oh * h_in / h_out))
        h_end = int(math.ceil((oh + 1) * h_in / h_out))
        for ow in range(w_out):
            w_start = int(math.floor(ow * w_in / w_out))
            w_end = int(math.ceil((ow + 1) * w_in / w_out))
            region = x[:, :, h_start:h_end, w_start:w_end]
            out[:, :, oh, ow] = torch.mean(region, dim=(-1, -2))
    return out


def adaptive_avg_pool2d_optimized(x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    # Optimized adaptive average pooling path.
    return F.adaptive_avg_pool2d(x, output_size=output_size)


# -----------------------------
# Embedding operators
# -----------------------------

def embedding_lookup_naive(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # Naive embedding lookup with explicit element gather.
    # weight[V, D], indices[*] -> output[* , D].
    flat = indices.reshape(-1)
    out = torch.empty((flat.shape[0], weight.shape[1]), dtype=weight.dtype, device=weight.device)
    for i in range(flat.shape[0]):
        out[i] = weight[int(flat[i])]
    return out.reshape(*indices.shape, weight.shape[1])


def embedding_lookup_optimized(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # Optimized embedding lookup via advanced indexing.
    return weight[indices]


def batched_embedding_lookup_naive(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # Batched embedding lookup reference.
    # weight[B, V, D], indices[B, T] -> output[B, T, D].
    bsz, _, d = weight.shape
    t = indices.shape[1]
    out = torch.empty((bsz, t, d), dtype=weight.dtype, device=weight.device)
    for b in range(bsz):
        for i in range(t):
            out[b, i] = weight[b, int(indices[b, i])]
    return out


def batched_embedding_lookup_optimized(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # Optimized batched embedding lookup using torch.gather.
    bsz, _, d = weight.shape
    idx = indices.unsqueeze(-1).expand(bsz, indices.shape[1], d)
    return torch.gather(weight, dim=1, index=idx)


def embedding_bag_naive(
    weight: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    mode: str = "sum",
) -> torch.Tensor:
    # Naive embedding bag.
    # offsets define segment starts in indices.
    n_bags = offsets.shape[0]
    out = torch.zeros((n_bags, weight.shape[1]), dtype=weight.dtype, device=weight.device)

    for b in range(n_bags):
        start = int(offsets[b])
        end = int(offsets[b + 1]) if b + 1 < n_bags else int(indices.shape[0])
        bag = weight[indices[start:end]]

        if bag.shape[0] == 0:
            continue
        if mode == "sum":
            out[b] = torch.sum(bag, dim=0)
        elif mode == "mean":
            out[b] = torch.mean(bag, dim=0)
        elif mode == "max":
            out[b] = torch.max(bag, dim=0).values
        else:
            raise ValueError("mode must be one of: sum, mean, max")
    return out


def embedding_bag_optimized(
    weight: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    mode: str = "sum",
) -> torch.Tensor:
    # Optimized embedding bag path via torch.nn.functional.embedding_bag.
    return F.embedding_bag(indices, weight, offsets, mode=mode, include_last_offset=False)


def conv2d_backward_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_out: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # Conv2D backward reference using autograd.
    # Returns gradients for input, weight, and optional bias.
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    b_ref = bias.detach().clone().requires_grad_(True) if bias is not None else None

    y = F.conv2d(x_ref, w_ref, bias=b_ref, stride=stride, padding=padding, dilation=dilation, groups=groups)
    y.backward(grad_out)

    grad_x = x_ref.grad if x_ref.grad is not None else torch.zeros_like(x_ref)
    grad_w = w_ref.grad if w_ref.grad is not None else torch.zeros_like(w_ref)
    grad_b = b_ref.grad if b_ref is not None and b_ref.grad is not None else None
    return grad_x, grad_w, grad_b


# -----------------------------
# Practice Questions
# -----------------------------
# Beginner:
# 1) Derive Conv2D output size from input size, kernel, stride, padding, and dilation.
# 2) What is the intuition behind depthwise-separable convolution?
# 3) Why do pooling layers reduce overfitting and compute cost?
# Intermediate:
# 4) How does im2col transform convolution into GEMM, and what is its memory overhead?
# 5) Compare grouped convolution and depthwise convolution in parameter count.
# 6) How would you choose adaptive pooling output size for variable-resolution inputs?
# Advanced:
# 7) Explain transposed convolution artifacts and how stride/kernel choices affect them.
# 8) Why can col2im require overlap accumulation?
# 9) Outline Winograd transform stages and where numerical error increases.
# Expert:
# 10) Design a fused conv+bias+activation kernel with tiling strategy.
# 11) Discuss cache and shared-memory blocking choices for 3D convolution.
# 12) How would you profile whether a convolution is memory-bound or compute-bound?

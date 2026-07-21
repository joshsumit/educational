from __future__ import annotations

# This file contains the small mathematical building blocks used by the later
# quantization stages. Keep it simple: no kernels, no model code, no calibration
# policy. Only ranges, scales, zero-points, and error metrics live here.

import torch

from .constants import EPS


def integer_range(num_bits: int, signed: bool = True) -> tuple[int, int]:
    """Return the representable integer range for a quantized format.

    Examples:
        signed INT8   -> [-128, 127]
        unsigned INT8 -> [0, 255]
        signed INT4   -> [-8, 7]
        unsigned INT4 -> [0, 15]

    Args:
        num_bits: Number of bits used by the integer representation.
        signed: Whether the integer format has both negative and positive values.

    Returns:
        A tuple ``(qmin, qmax)``.
    """
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")

    if signed:
        # Two's-complement signed range:
        #   min = -2^(bits - 1)
        #   max =  2^(bits - 1) - 1
        return -(1 << (num_bits - 1)), (1 << (num_bits - 1)) - 1

    # Unsigned integer range:
    #   min = 0
    #   max = 2^bits - 1
    return 0, (1 << num_bits) - 1


def symmetric_scale(x: torch.Tensor, num_bits: int = 8, signed: bool = True) -> torch.Tensor:
    """Compute the scale for symmetric quantization.

    Symmetric quantization fixes zero-point to 0 and maps the largest absolute
    floating-point value to the largest representable integer magnitude.

    Formula:
        scale = max(abs(x)) / max(abs(qmin), abs(qmax))

    For signed INT8 this is usually:
        scale = max(abs(x)) / 128

    Some production systems prefer 127 instead of 128 to keep the positive and
    negative ranges balanced around zero. This implementation uses the exact
    two's-complement range magnitude returned by ``integer_range``.

    Args:
        x: Floating-point tensor to be quantized later.
        num_bits: Number of integer bits.
        signed: Whether the target integer format is signed.

    Returns:
        A scalar tensor containing the quantization scale.
    """
    qmin, qmax = integer_range(num_bits, signed)

    # For signed types, the negative range has one extra value in two's-complement
    # formats. Example: INT8 is [-128, 127]. max(abs(qmin), abs(qmax)) is 128.
    # For unsigned types, qmax is the full positive range.
    denom = max(abs(qmin), abs(qmax)) if signed else qmax

    # Convert to float32 before reductions so integer or lower-precision inputs
    # do not affect scale computation.
    max_abs = torch.max(torch.abs(x.to(torch.float32)))

    # EPS prevents division-by-zero later when all values in x are zero.
    return torch.clamp(max_abs / float(denom), min=EPS)


def asymmetric_scale_zero_point(
    x: torch.Tensor,
    num_bits: int = 8,
    signed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scale and zero-point for affine/asymmetric quantization.

    Asymmetric quantization maps the real floating-point range ``[xmin, xmax]``
    onto the integer range ``[qmin, qmax]``.

    Formula:
        scale = (xmax - xmin) / (qmax - qmin)
        zero_point = round(qmin - xmin / scale)

    The zero-point is the integer value that should represent real value 0.0.
    Because it must be stored as an integer, it is rounded and clamped.

    Args:
        x: Floating-point tensor to analyze.
        num_bits: Number of integer bits.
        signed: Whether the target integer format is signed.

    Returns:
        ``(scale, zero_point)`` where scale is float tensor and zero_point is int32 tensor.
    """
    qmin, qmax = integer_range(num_bits, signed)
    xf = x.to(torch.float32)
    xmin, xmax = torch.min(xf), torch.max(xf)

    # Degenerate case: every value is equal. There is no meaningful dynamic
    # range, so return a safe scale. Using zero_point=0 keeps the mapping simple.
    if torch.isclose(xmax, xmin):
        return (
            torch.tensor(1.0, device=x.device),
            torch.tensor(0 if signed else qmin, dtype=torch.int32, device=x.device),
        )

    # Map real range width to integer range width.
    scale = torch.clamp((xmax - xmin) / float(qmax - qmin), min=EPS)

    # Choose an integer zero-point so that real 0.0 is represented as closely as
    # possible. Clamp because rounding can push it just outside the legal range.
    zp = torch.round(qmin - xmin / scale).clamp(qmin, qmax).to(torch.int32)

    return scale, zp


def quantization_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Return elementwise reconstruction error after quantize/dequantize.

    ``x`` is the original tensor.
    ``x_hat`` is the reconstructed/dequantized tensor.

    Error convention used here:
        error = reconstructed - original
    """
    return x_hat.to(torch.float32) - x.to(torch.float32)


def mse_quantization_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Return mean squared quantization error.

    MSE is useful when comparing quantization schemes because it penalizes large
    reconstruction errors more strongly than small errors.
    """
    e = quantization_error(x, x_hat)
    return torch.mean(e * e)


def max_abs_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Return the largest absolute reconstruction error.

    This is useful for spotting worst-case error caused by clipping, saturation,
    or a poor scale choice.
    """
    return torch.max(torch.abs(quantization_error(x, x_hat)))


def sqnr_db(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Return signal-to-quantization-noise ratio in decibels.

    Formula:
        SQNR(dB) = 10 * log10(signal_power / noise_power)

    Higher SQNR means the reconstructed tensor is closer to the original tensor.
    """
    signal = torch.sum(x.to(torch.float32) ** 2)
    noise = torch.sum((x.to(torch.float32) - x_hat.to(torch.float32)) ** 2)

    # Clamp both terms so all-zero inputs or perfect reconstruction do not create
    # divide-by-zero or log-of-zero numerical issues.
    return 10.0 * torch.log10(torch.clamp(signal, min=EPS) / torch.clamp(noise, min=EPS))


def saturation_rate(q: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """Return fraction of quantized values that hit the integer limits.

    A high saturation rate often means the chosen calibration range is too narrow.
    In that case, many real values were clipped to qmin or qmax before storage.

    Args:
        q: Quantized integer tensor.
        qmin: Minimum representable integer value.
        qmax: Maximum representable integer value.

    Returns:
        Scalar tensor in [0, 1]. Example: 0.05 means 5% saturated values.
    """
    return ((q == qmin) | (q == qmax)).to(torch.float32).mean()

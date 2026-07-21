from __future__ import annotations
import math
import torch
from .stage04_low_bit_quantization import pack_int4, unpack_int4


def int8_dot_product(a: torch.Tensor, b: torch.Tensor, a_zero_point: int = 0, b_zero_point: int = 0) -> torch.Tensor:
    if a.numel() != b.numel():
        raise ValueError("dot product inputs must have same number of elements")
    return torch.sum((a.flatten().to(torch.int32) - a_zero_point) * (b.flatten().to(torch.int32) - b_zero_point))


def dp4a_reference(a4: torch.Tensor, b4: torch.Tensor) -> torch.Tensor:
    if a4.numel() != 4 or b4.numel() != 4:
        raise ValueError("dp4a_reference expects 4 int8 values per input")
    return torch.sum(a4.to(torch.int32) * b4.to(torch.int32))


def int4_packing_layout(q: torch.Tensor, signed: bool = True):
    packed = pack_int4(q, signed=signed)
    return packed, {"original_shape": tuple(q.shape), "values_per_byte": 2, "signed": signed}


def int4_unpacking_layout(packed: torch.Tensor, metadata: dict) -> torch.Tensor:
    vals = unpack_int4(packed, math.prod(metadata["original_shape"]), signed=metadata.get("signed", True))
    return vals.view(metadata["original_shape"])


def tensorcore_int8_tile_layout(a: torch.Tensor, tile_m: int = 16, tile_k: int = 32) -> torch.Tensor:
    m, k = a.shape
    pm, pk = (-m) % tile_m, (-k) % tile_k
    ap = torch.nn.functional.pad(a, (0, pk, 0, pm))
    return ap.view((m+pm)//tile_m, tile_m, (k+pk)//tile_k, tile_k).permute(0, 2, 1, 3).contiguous()


def tensorcore_fp8_tile_layout(a: torch.Tensor, tile_m: int = 16, tile_k: int = 16) -> torch.Tensor:
    return tensorcore_int8_tile_layout(a, tile_m, tile_k)


def memory_footprint_bytes(num_elements: int, bits_per_element: int, metadata_bytes: int = 0) -> int:
    return (num_elements * bits_per_element + 7) // 8 + metadata_bytes


def memory_bandwidth_estimation(num_elements: int, src_bits: int = 32, dst_bits: int = 8) -> dict:
    src = memory_footprint_bytes(num_elements, src_bits)
    dst = memory_footprint_bytes(num_elements, dst_bits)
    return {"source_bytes": src, "quantized_bytes": dst, "compression_ratio": src / max(dst, 1)}


def cache_footprint_analysis(shape: tuple[int, ...], bits_per_element: int, cache_bytes: int = 256 * 1024) -> dict:
    n = math.prod(shape)
    b = memory_footprint_bytes(n, bits_per_element)
    return {"elements": n, "bytes": b, "fits_cache": b <= cache_bytes, "cache_bytes": cache_bytes}


def systolic_array_gemm_reference(a: torch.Tensor, b: torch.Tensor, tile: int = 16) -> torch.Tensor:
    m, k = a.shape; k2, n = b.shape
    if k != k2:
        raise ValueError("inner dimensions must match")
    out = torch.zeros((m, n), dtype=torch.int32, device=a.device)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            for kk in range(0, k, tile):
                out[i:i+tile, j:j+tile] += a[i:i+tile, kk:kk+tile].to(torch.int32) @ b[kk:kk+tile, j:j+tile].to(torch.int32)
    return out

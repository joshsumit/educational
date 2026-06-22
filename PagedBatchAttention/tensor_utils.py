
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=160)


def banner(title: str):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def subsection(title: str):
    print("\n" + "-" * 120)
    print(title)
    print("-" * 120)


def explain_shape(shape, names):
    if len(shape) != len(names):
        return f"shape={shape}"
    return ", ".join([f"{n}={v}" for n, v in zip(names, shape)])


def print_tensor(name: str, x, meaning: str, dim_names=None):
    print(f"\n{name}")
    print(f"What this tensor is: {meaning}")
    if hasattr(x, "shape"):
        print(f"Raw shape: {x.shape}")
        if dim_names is not None:
            print(f"Dimension meaning: [{', '.join(dim_names)}]")
            print(f"Expanded shape meaning: {explain_shape(x.shape, dim_names)}")
    print("Values:")
    print(x)


def print_matrix_multiply(name: str, left_shape, right_shape, out_shape, explanation: str):
    print(f"\n{name}")
    print(f"Operation: {left_shape} x {right_shape} -> {out_shape}")
    print(f"Why: {explanation}")


def tensor_bytes(x):
    return int(x.size * x.itemsize)


def print_tensor_stats(name: str, x):
    print(f"Tensor stats for {name}: elements={x.size}, dtype={x.dtype}, bytes={tensor_bytes(x)}")

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function with Numba.

    Args:
    ----
        fn: The function to be compiled.
        **kwargs: Additional arguments for Numba's njit.

    Returns:
    -------
        The JIT-compiled function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if np.array_equal(in_strides, out_strides) and np.array_equal(
            in_shape, out_shape
        ):
            for i in prange(
                len(out)
            ):  # Parallel loop over all elements in the output tensor.
                out[i] = fn(
                    in_storage[i]
                )  # Directly apply the function `fn` to the input element.

        else:
            for i in prange(
                len(out)
            ):  # Parallel loop over all elements in the output tensor.
                # Create index arrays for navigating the output and input tensors.
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
                in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)

                # Convert flat output index into a multidimensional index.
                to_index(i, out_shape, out_index)

                # Compute the corresponding input index using broadcasting rules.
                broadcast_index(out_index, out_shape, in_shape, in_index)

                # Translate multidimensional indices to positions in flat storage.
                o = index_to_position(out_index, out_strides)  # Output position.
                j = index_to_position(in_index, in_strides)  # Input position.

                # Apply the function `fn` to the input element and store the result in the output.
                out[o] = fn(in_storage[j])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if (
            np.array_equal(a_strides, b_strides)
            and np.array_equal(a_strides, out_strides)
            and np.array_equal(a_shape, b_shape)
            and np.array_equal(a_shape, out_shape)
        ):
            for i in prange(len(out)):  # Parallel loop over all output elements.
                out[i] = fn(a_storage[i], b_storage[i])  # Apply function directly.

        else:
            for i in prange(len(out)):  # Parallel loop over all output elements.
                # Create index arrays for output and input tensors.
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
                a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
                b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)

                # Convert flat output index into a multidimensional index.
                to_index(i, out_shape, out_index)

                # Compute corresponding indices for A and B using broadcasting rules.
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)

                # Convert multidimensional indices to positions in flat storage.
                o = index_to_position(out_index, out_strides)
                j = index_to_position(a_index, a_strides)
                k = index_to_position(b_index, b_strides)

                # Apply the function `fn` and store the result in the output tensor.
                out[o] = fn(a_storage[j], b_storage[k])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # Parallel loop over all output positions.
        for i in prange(len(out)):
            out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
            # Convert the flat output index `i` to a multi-dimensional index.
            to_index(i, out_shape, out_index)

            # Compute the output position in the flat storage.
            o = index_to_position(out_index, out_strides)

            # Compute the input position corresponding to the output position.
            j = index_to_position(out_index, a_strides)

            # Initialize the accumulator with the current value in the output tensor.
            acc = out[o]

            # Compute the stride step along the reduction dimension.
            step = a_strides[reduce_dim]

            # Iterate over the reduction dimension and accumulate results.
            for _ in range(a_shape[reduce_dim]):
                acc = fn(acc, a_storage[j])  # Apply the reduction function.
                j += step  # Move to the next element in the reduction dimension.

            # Store the accumulated result back into the output tensor.
            out[o] = acc

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Ensure dimensions are compatible for matrix multiplication.
    assert a_shape[-1] == b_shape[-2]

    # Outer loop over the batch dimension (parallelized).
    for batch in prange(out_shape[0]):
        # Loop over the rows (j) and columns (i) of the output matrix.
        for i in range(out_shape[-1]):  # Column index in the output matrix.
            for j in range(out_shape[-2]):  # Row index in the output matrix.
                # Compute the starting positions for the current row of `a` and column of `b`.
                a_pos = batch * a_batch_stride + j * a_strides[-2]
                b_pos = batch * b_batch_stride + i * b_strides[-1]

                # Initialize the accumulator for the dot product.
                acc = 0.0

                # Reduction loop over the shared dimension.
                for _ in range(a_shape[-1]):
                    acc += (
                        a_storage[a_pos] * b_storage[b_pos]
                    )  # Dot product accumulation.
                    a_pos += a_strides[
                        -1
                    ]  # Move to the next element in the current row of `a`.
                    b_pos += b_strides[
                        -2
                    ]  # Move to the next element in the current column of `b`.

                # Compute the position in the output tensor and store the result.
                o = j * out_strides[-2] + i * out_strides[-1] + batch * out_strides[0]
                out[o] = acc  # Store the accumulated value.


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None

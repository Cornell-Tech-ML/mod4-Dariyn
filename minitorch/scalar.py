from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    def __eq__(self, other: ScalarLike) -> "Scalar":
        """Return self == other as a Scalar."""
        return EQ.apply(self, other)

    def __lt__(self, other: ScalarLike) -> "Scalar":
        """Return self < other as a Scalar."""
        return LT.apply(self, other)

    def __add__(self, other: ScalarLike) -> "Scalar":
        """Return self + other as a Scalar."""
        return Add.apply(self, other)

    def __neg__(self) -> "Scalar":
        """Return -self as a Scalar."""
        return Neg.apply(self)

    def log(self) -> "Scalar":
        """Return log(self) as a Scalar."""
        return Log.apply(self)

    def exp(self) -> "Scalar":
        """Return exp(self) as a Scalar."""
        return Exp.apply(self)

    def inv(self) -> "Scalar":
        """Return 1/self as a Scalar."""
        return Inv.apply(self)

    def sigmoid(self) -> "Scalar":
        """Return sigmoid(self) as a Scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> "Scalar":
        """Return relu(self) as a Scalar."""
        return ReLU.apply(self)

    def __sub__(self, other: ScalarLike) -> "Scalar":
        """Return self - other as a Scalar."""
        return Add.apply(self, Neg.apply(other))

    def __rsub__(self, other: ScalarLike) -> "Scalar":
        """Return other - self as a Scalar."""
        return Add.apply(Neg.apply(self), other)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Return True if this variable is constant (no history)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Return the parent variables of this variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute the derivative of the output with respect to each input variable.

        This method is used to compute the derivative of the output of a function with respect to its input variables.
        It recursively applies the chain rule to each operation in the computation history of the variable, starting from the output.
        The method returns an iterable of tuples, where each tuple contains a variable and its corresponding derivative.

        Args:
        ----
            d_output (Any): The derivative of the output with respect to which the chain rule is applied.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple contains a variable and its corresponding derivative.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        local_grads = h.last_fn._backward(h.ctx, d_output)

        # Yield each input along with its corresponding gradient
        for inp, grad in zip(h.inputs, local_grads):
            yield inp, grad

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters
    ----------
    f : callable
        A function that takes n scalar values and returns a single scalar value.
        This function will be tested for correct autodifferentiation.
    *scalars : Scalar
        Variable number of Scalar objects to be used as inputs to function f.
        These are the points at which the derivative will be checked.

    Returns
    -------
    None
        Function asserts if the autodiff derivative doesn't match
        the numerical derivative within a tolerance of 1e-2.

    Examples
    --------
    >>> def f(x, y):
    ...     return x * y
    >>> x, y = Scalar(1.0), Scalar(2.0)
    >>> derivative_check(f, x, y)

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )

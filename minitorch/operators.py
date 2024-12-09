"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Get the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close in value."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Exponential function."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg."""
    return y / (x + EPS)


def inv(x: float) -> float:
    """Calculates the reciprocal."""
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : Negate all elements in a list using map
# - addLists : Add corresponding elements from two lists using zipWith
# - sum: Sum all elements in a list using reduce
# - prod: Calculate the product of all elements in a list using reduce


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float], arr: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable."""
    return [fn(x) for x in arr]


def zipWith(
    fn: Callable[[float, float], float], arr1: Iterable[float], arr2: Iterable[float]
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function."""
    result = []

    for x, y in zip(arr1, arr2):
        result.append(fn(x, y))
    return result


def reduce(
    fn: Callable[[float, float], float], arr: Iterable[float], init: float
) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function."""
    result = init
    for x in arr:
        result = fn(result, x)
    return result


def negList(arr: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list."""
    return map(lambda x: -x, arr)


def addLists(arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith."""
    return zipWith(add, arr1, arr2)


def sum(arr: Iterable[float]) -> float:
    """Sum all elements in a list."""
    return reduce(add, arr, 0.0)


def prod(arr: Iterable[float]) -> float:
    """Calculate the product of all elements in a list."""
    return reduce(mul, arr, 1.0)

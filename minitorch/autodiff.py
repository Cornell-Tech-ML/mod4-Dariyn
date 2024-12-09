from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique id of the variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns True if the variable is a leaf, no last_fn in history."""
        ...

    def is_constant(self) -> bool:
        """Returns True if the variable is a constant, no history."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents (inputs) of the variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Implements the chain rule for the variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    seen = set()
    order: List[Variable] = []

    def visit(var: Variable) -> None:
        """Depth-first search from the variable."""
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
    variable: The right-most variable.
    deriv: Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    As illustrated in the graph for the above example, each of the red arrows represents a constructed derivative which eventually passed to
    d_out in the chain rule. Starting from the rightmost arrow, which is passed in as an argument, backpropagate should run the following algorithm:

    1. Call topological sort to get an ordered queue
    2. Create a dictionary of Scalars and current derivatives
    3. For each node in backward order, pull a completed Scalar and derivative from the queue:
        a. if the Scalar is a leaf, add its final derivative (accumulate_derivative) and loop to (1)
        b. if the Scalar is not a leaf,
            a. call .chain_rule on the last function with d_out
            b. loop through all the Scalars+derivative produced by the chain rule
            c. accumulate derivatives for the Scalar in a dictionary

    Final note: only leaf Scalars should ever have non-None .derivative value. All intermediate Scalars should only keep their current derivative values in the dictionary. This is a bit annoying, but it follows the behavior of PyTorch.

    """
    # Call topological sort to get an ordered queue
    queue = topological_sort(variable)

    # Create a dictionary of Scalars and current derivatives
    derivatives = {variable.unique_id: deriv}

    # Traverse the graph in topological order
    for var in queue:
        deriv = derivatives[var.unique_id]

        # If node is a leaf, add the final derivative
        if var.is_leaf():
            var.accumulate_derivative(deriv)

        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] += d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors."""
        return self.saved_values

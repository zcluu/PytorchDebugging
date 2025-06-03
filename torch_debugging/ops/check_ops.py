import torch

import numpy as np

from torch_debugging.utils.ops import binary_assert, message_prefix

__all__ = [
    "assert_non_negative",
    "assert_equal",
    "assert_not_equal",
    "assert_less",
    "assert_less_equal",
    "assert_greater",
    "assert_greater_equal",
    "assert_rank_at_least",
]


def assert_non_negative(x, data=None, summarize=None, message=None, name=None):
    message = message_prefix(message)

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    x_name = name or "x"

    if data is None:
        data = [
            f"{message}Condition x >= 0 did not hold element-wise:",
            f"{x_name} =",
            x,
        ]
    zero = torch.tensor(0, dtype=x.dtype, device=x.device)
    return assert_less_equal(
        zero,
        x,
        data=data,
        summarize=summarize,
        message=None,
        name="assert_non_negative",
    )


def assert_equal(x, y, data=None, summarize=None, message=None, name=None):
    return binary_assert(
        "==", "assert_equal", torch.eq, np.equal, x, y, data, summarize, message, name
    )


def assert_not_equal(x, y, **kwargs):
    return binary_assert(
        "!=", "assert_not_equal", torch.ne, np.not_equal, x, y, **kwargs
    )


def assert_less(x, y, **kwargs):
    return binary_assert("<", "assert_less", torch.lt, np.less, x, y, **kwargs)


def assert_less_equal(x, y, **kwargs):
    return binary_assert(
        "<=", "assert_less_equal", torch.le, np.less_equal, x, y, **kwargs
    )


def assert_greater(x, y, **kwargs):
    return binary_assert(">", "assert_greater", torch.gt, np.greater, x, y, **kwargs)


def assert_greater_equal(x, y, **kwargs):
    return binary_assert(
        ">=", "assert_greater_equal", torch.ge, np.greater_equal, x, y, **kwargs
    )


def assert_rank_at_least(x, rank, data=None, summarize=3, message=None, name=None):
    message = message_prefix(message)

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if not isinstance(rank, int):
        raise TypeError("`rank` must be an int in PyTorch version.")

    actual_rank = x.dim()

    if actual_rank < rank:
        raise ValueError(
            f"{message}Tensor must have rank at least {rank}. "
            f"Received rank {actual_rank}, shape {tuple(x.shape)}."
        )

    if data is None:
        data = [
            f"{message}Tensor must have rank at least {rank}.",
            f"Actual rank: {actual_rank}",
            f"Shape: {tuple(x.shape)}",
            f"Tensor (summary): {x.flatten()[:summarize].tolist()}",
        ]

    if actual_rank < rank:
        raise AssertionError("\n".join(str(d) for d in data))

    return

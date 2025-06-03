import torch
import numpy as np

from .tensors import get_static_value


def message_prefix(message):
    if message is None:
        return ""
    else:
        return "%s.  " % message


def binary_assert(
    sym,
    opname,
    op_func,
    static_func,
    x,
    y,
    data=None,
    summarize=3,
    message=None,
    name=None,
):
    # Type checks
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError(f"x and y must be torch.Tensor, got {type(x)} and {type(y)}")

    x_static = get_static_value(x)
    y_static = get_static_value(y)

    if x_static is not None and y_static is not None:
        condition_static = static_func(x_static, y_static)
        if not np.all(condition_static):
            msg = [f"Static check failed for `{opname}`: x {sym} y"]
            if message is not None:
                msg.insert(0, message)
            raise AssertionError("\n".join(msg))

    try:
        result = op_func(x, y)
    except RuntimeError as e:
        raise AssertionError(f"{opname} failed to apply: {e}")

    if torch.all(result):
        return

    if summarize is None:
        summarize = 3
    elif summarize < 0:
        summarize = x.numel()

    failed_indices = (result == False).nonzero(as_tuple=False)
    failed_x = x[failed_indices[:, 0]]
    failed_y = y[failed_indices[:, 0]]

    msg = []

    if message is not None:
        msg.append(message)

    msg.append(f"Assertion `{opname}` failed: x {sym} y did not hold element-wise.")
    msg.append(f"x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")

    msg.append(f"First {summarize} mismatches:")
    for i in range(min(summarize, failed_x.shape[0])):
        msg.append(f"  x = {failed_x[i].item()}, y = {failed_y[i].item()}")

    if data:
        msg.append("Additional data:")
        msg.extend([str(d) for d in data])

    raise AssertionError("\n".join(msg))

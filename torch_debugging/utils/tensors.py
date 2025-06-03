from typing import Optional
import torch
import numpy as np


def get_static_value(tensor) -> Optional[np.ndarray]:
    """
    Attempt to extract static value from tensor.
    """
    if isinstance(tensor, torch.Tensor):
        try:
            return tensor.detach().cpu().numpy()
        except:
            return None
    return None

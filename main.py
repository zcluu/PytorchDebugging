import torch

from torch_debugging.ops import *

assert_equal(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
assert_non_negative(torch.tensor([1, 2, 3]))
assert_rank_at_least(torch.randn(3, 4, 5), 3, message="Rank too low.")

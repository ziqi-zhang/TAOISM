from collections import namedtuple

import torch.distributed as dist
import numpy as np
import torch
import random
import os
from pdb import set_trace as st

def torch_sync():
    dist.barrier()


def get_prod(x):
    return np.prod(list(x))


def get_numpy(x):
    return x.cpu().detach().numpy()


def reject_outliers(data, m=2):
    return data[abs(data - torch.mean(data)) < m * torch.std(data)]


def compare_expected_actual(expected, actual, show_where_err=False, get_relative=False, verbose=False, show_values=False):
    def purify(x):
        # return torch.tensor(x)
        res = x
        # if not (isinstance(x, torch.Tensor) or isinstance(x, torch.Variable)):
        if not (isinstance(x, torch.Tensor) ):
            res = torch.tensor(x)
            # return x.detach().numpy()
        return res.type(torch.float).to("cpu")
    expected = purify(expected)
    actual = purify(actual)

    if show_values:
        print("expected:", expected[0, 0])
        print("actual:", actual[0, 0])

    avg_abs_diff = torch.mean(torch.abs(expected - actual)).item()
    res = avg_abs_diff

    if show_where_err:
        show_indices = torch.abs(expected - actual) / torch.abs(expected) > 0.5
        # show_indices = (expected != actual)
        print("error indices: ", np.where(show_indices.cpu()))
        print("expected values:", expected[show_indices])
        print("difference:", (expected - actual)[show_indices])

    if get_relative:
        tmp_expected, tmp_actual = expected[expected != 0], actual[expected != 0]
        relative_diff = torch.abs(tmp_expected - tmp_actual) / torch.abs(tmp_expected)
        relative_avg_diff = torch.mean(torch.abs(tmp_actual - tmp_expected)) / torch.mean(torch.abs(tmp_expected))
        Error = namedtuple("Error", ("AvgAbsDiff", "RelAvgDiff", "AvgRelDiff", "StdRelDiff"))
        res = Error(avg_abs_diff, relative_avg_diff.item(), torch.mean(relative_diff).item(), torch.std(relative_diff).item())

    if verbose:
        print(res)

    return res

def seed_torch(seed=123):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

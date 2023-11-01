#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import torch

from python.global_config import SecretConfig
from python.enclave_interfaces import GlobalParam

device = torch.device("cuda:0")


class GlobalCppExtension(object):
    __instance = None

    @staticmethod
    def get_instance():
        if GlobalCppExtension.__instance is None:
            GlobalCppExtension()
        return GlobalCppExtension.__instance

    def __init__(self):
        GlobalCppExtension.__instance = self
        from torch.utils.cpp_extension import load
        self.conv2d_cudnn = load(name="conv2d_backward", sources=["./conv2d_extension/conv2d_backward.cpp"],
                                 verbose=True)

    @staticmethod
    def get_conv2d_cudnn():
        return GlobalCppExtension.get_instance().conv2d_cudnn


def union_dicts(*dicts):
    return dict(i for dct in dicts for i in dct.items())


def calc_conv2d_output_shape(x_shape, w_shape, padding):
    batch_size = x_shape[0]
    n_output_channel = w_shape[0]
    img_hw = x_shape[3]
    filter_hw = w_shape[3]
    output_hw = img_hw + 2 * padding - (filter_hw - 1)
    return [batch_size, n_output_channel, output_hw, output_hw]

def calc_conv2d_output_shape_stride(x_shape, w_shape, padding, stride):
    batch_size = x_shape[0]
    n_output_channel = w_shape[0]
    img_hw = x_shape[3]
    filter_hw = w_shape[3]
    assert padding == int(filter_hw / 2)
    output_hw = int( (img_hw + 2 * padding - filter_hw)/stride +1 )
    return [batch_size, n_output_channel, output_hw, output_hw]


def calc_shape_conv2d_weight(dy, x):
    padding = 1
    batch_size, in_chan, img_hw, _ = x.size()
    _, out_chan, out_hw, __ = dy.size()
    filter_hw = img_hw - out_hw + 2 * padding + 1
    return [out_chan, in_chan, filter_hw, filter_hw]


def mod_on_cpu(x):
    return x.fmod_(SecretConfig.PrimeLimit).add_(SecretConfig.PrimeLimit).fmod_(SecretConfig.PrimeLimit)
    return torch.from_numpy(np.fmod(x.numpy().astype(np.uint16) + PRIME_LIMIT, PRIME_LIMIT).astype(np.int16))


def mod_on_gpu(x):
    x.fmod_(SecretConfig.PrimeLimit).add_(SecretConfig.PrimeLimit).fmod_(SecretConfig.PrimeLimit)


# The range is [-p//2 - 1, p//2)
def mod_move_down(x):
    x = x.fmod(SecretConfig.PrimeLimit).add(SecretConfig.PrimeLimit).fmod(SecretConfig.PrimeLimit)
    p = SecretConfig.PrimeLimit
    return torch.where(x >= p // 2, x - p, x)


def move_down(x):
    p = SecretConfig.PrimeLimit
    np_x = x.numpy()
    np_x[np_x >= p // 2] -= p
    np_x[np_x < -p // 2] += p


def quantize(x, src):
    over, scale = src
    c = over * scale
    return torch.clamp((x * c).round(), -scale + 1, scale - 1)


def dequantize(x, src1, src2, dst):
    # return x
    over1, scale1 = src1
    over2, scale2 = src2
    total = over1 * scale1 * over2 * scale2
    return x / total


def find_max_expand(x):
    dim_size = len(list(x.size()))
    XMax = torch.max(torch.abs(x.contiguous().view(x.size()[0], -1)), 1, keepdim=True)[0]
    Shift = XMax
    return torch.clamp(Shift.reshape([x.size()[0]] + [1] * (dim_size - 1)), 1e-6, float('inf'))


def rescale(x, scale):
    return quantize(x / find_max_expand(x), scale)


def get_random_uniform(upper_bound, size):
    return torch.from_numpy(np.random.uniform(0, upper_bound - 1, size=size)).type(torch.int32).type(
        SecretConfig.dtypeForCpuOp)


def generate_unquantized_tensor(enum, shape):
    over, scale = GlobalParam.get_for_enum(enum)
    t = get_random_uniform(2 * scale - 2, shape).type(torch.int).type(SecretConfig.dtypeForCpuOp) - (scale - 1)
    t /= (over * scale)
    return t


def modest_magnitude(w):
    if isinstance(w, torch.autograd.Variable):
        w = w.detach()
    if isinstance(w, torch.Tensor):
        w = w.numpy()
    n_bins = 100
    n_modest = 3
    sort = np.sort(np.abs(w.reshape(-1)))
    indices = np.linspace(0, sort.size - 1, n_bins).astype(int)
    low_idx, high_idx = n_modest, n_bins - n_modest - 1
    while sort[indices[low_idx]] == 0 and low_idx < high_idx:
        low_idx += 1
    return np.log2(sort[indices[low_idx]]), np.log2(sort[indices[high_idx]])

import os
import sys
from pdb import set_trace as st
import numpy as np
import torch
from torch import optim, nn
import torch.distributed as dist

from python.common_net import register_layer, register_weight_layer, get_layer_weight, get_layer_input, \
    get_layer_weight_grad, get_layer_output, get_layer_output_grad, get_layer_input_grad
from python.enclave_interfaces import GlobalTensor
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.sgx_net import init_communicate, warming_up_cuda, SecretNeuralNetwork, SgdOptimizer
from python.utils.basic_utils import ExecutionModeOptions

from python.utils.torch_utils import compare_expected_actual, seed_torch

device_cuda = torch.device("cuda:0")
torch.set_printoptions(precision=10)

def test_BN(sid=0, master_addr=0, master_port=0, is_compare=False):

    batch_size = 2
    n_img_channel = 256
    img_hw = 32

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape, ExecutionModeOptions.Enclave)
    input = torch.rand(x_shape)
    # input.zero_()
    # input += 1

    test_layer = SecretBatchNorm2dLayer(sid, f"TestNorm", ExecutionModeOptions.Enclave)

    output_layer = SecretOutputLayer(sid, "OutputLayer", ExecutionModeOptions.Enclave, inference=True)
    layers = [input_layer, test_layer, output_layer]
    
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)
    
    input_layer.StoreInEnclave = False

    plain_module = nn.BatchNorm2d(n_img_channel)
    plain_module.eval()
    plain_module.weight.requires_grad = False
    plain_module.bias.requires_grad = False
    plain_module.weight.normal_()
    plain_module.bias.normal_()
    plain_module.running_mean.normal_()
    plain_module.running_var += 3
    # print(
    #     f"Weight {plain_module.weight}, bias {plain_module.bias}, mean {plain_module.running_mean}, var {plain_module.running_var}"
    # )
    test_layer.inject_params(plain_module)

    plain_output = plain_module(input)

    input_layer.set_input(input)
    secret_nn.forward()
    secret_nn.plain_forward()

    test_layer.transfer_enclave_to_cpu("output")
    secret_output = test_layer.get_cpu("output")
    final_input = output_layer.get_cpu("input")
    test_layer.show_plain_error_forward()

if __name__ == "__main__":
    # sys.stdout = Logger()

    seed_torch(123)
    test_BN()
    # test_linear()

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
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.output import SecretOutputLayer
from python.layers.relu import SecretReLULayer
from python.sgx_net import init_communicate, warming_up_cuda, SecretNeuralNetwork, SgdOptimizer
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.sgx_conv_base import SGXConvBase
from python.utils.basic_utils import ExecutionModeOptions

from python.utils.logger_utils import Logger
from python.quantize_net import NetQ
from python.test_sgx_net import argparser_distributed, marshal_process, load_cifar10, seed_torch
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel, NamedTimer
from python.utils.torch_utils import compare_expected_actual

device_cuda = torch.device("cuda:0")
torch.set_printoptions(precision=10)
def compare_layer_member(layer: SGXLinearBase, layer_name: str,
                         extract_func , member_name: str, save_path=None) -> None:
    print(member_name)
    layer.make_sure_cpu_is_latest(member_name)
    compare_expected_actual(extract_func(layer_name), layer.get_cpu(member_name), get_relative=True, verbose=True)
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("Directory ", save_path, " Created ")
        else:
            print("Directory ", save_path, " already exists")

        torch.save(extract_func(layer_name), os.path.join(save_path, member_name + "_expected"))
        torch.save(layer.get_cpu(member_name), os.path.join(save_path, member_name + "_actual"))


def compare_layer(layer: SGXLinearBase, layer_name: str, save_path=None) -> None:
    print("comparing with layer in expected NN :", layer_name)
    compare_name_function = [("input", get_layer_input), ("output", get_layer_output),
                             ("DerOutput", get_layer_output_grad), ]
    if layer_name != "conv1":
        compare_name_function.append(("DerInput", get_layer_input_grad))
    for member_name, extract_func in compare_name_function:
        compare_layer_member(layer, layer_name, extract_func, member_name, save_path=save_path)

def compare_weight_layer(layer: SGXLinearBase, layer_name: str, save_path=None) -> None:
    compare_layer(layer, layer_name, save_path)
    compare_name_function = [("weight", get_layer_weight), ("DerWeight", get_layer_weight_grad) ]
    for member_name, extract_func in compare_name_function:
        compare_layer_member(layer, layer_name, extract_func, member_name, save_path=save_path)



def test_maxpool(
    batch_size, img_hw, input_c, output_c,
    kernel, padding, stride, set_values_to_one=True,
    sid=0
):
    print("="*20, "TestConv", "="*20)
    print(
        f"batch {batch_size}, img_hw {img_hw}, input_c {input_c}, output_c {output_c}, " +
        f"kernel {kernel}, padding {padding}, stride {stride}"
    )

    x_shape = [batch_size, input_c, img_hw, img_hw]

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape, ExecutionModeOptions.Enclave)
    input = torch.rand(x_shape)
    # print("Python Input: ", input)
    num_elem = torch.numel(input)
    # input = torch.arange(num_elem).reshape(x_shape).float() + 1
    if set_values_to_one:
        input.zero_()
        input += 1
    # print("input: ", input)

    test_layer = SecretMaxpool2dLayer(
        sid, f"TestSGXMaxpool", ExecutionModeOptions.Enclave,
        kernel, stride, padding, 
    )
    

    output_layer = SecretOutputLayer(sid, "OutputLayer", ExecutionModeOptions.Enclave, inference=True)
    layers = [input_layer, test_layer, output_layer]
    
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)
    
    input_layer.StoreInEnclave = False

    plain_module = nn.MaxPool2d(
        kernel_size=kernel, stride=stride, padding=padding
    )

    input_layer.set_input(input)
    secret_nn.forward()
    secret_nn.plain_forward()

    test_layer.show_plain_error()

    module_output = output_layer.get_cpu("input")

    output = plain_module(input)

    diff = (module_output-output).abs().max()

    print("Diff: ", diff)
    GlobalTensor.destroy()


if __name__ == "__main__":
    # sys.stdout = Logger()

    seed_torch(123)

    # kernel, padding, stride = 3, 1, 2 for  student model
    test_maxpool(
        batch_size=64, img_hw=8, input_c=1024, output_c=1024,
        kernel=3, padding=1, stride=2, set_values_to_one=False,
    )
    test_maxpool(
        batch_size=1, img_hw=8, input_c=16, output_c=16,
        kernel=5, padding=2, stride=4, set_values_to_one=False,
    )

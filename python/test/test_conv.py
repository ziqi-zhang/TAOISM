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

import sys
import pdb
from pdb import set_trace as st

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
        from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def test_conv(
    batch_size, img_hw, input_c, output_c,
    kernel, padding, stride, bias=False, set_values_to_one=False,
    sid=0
):
    print("="*20, "TestConv", "="*20)
    print(
        f"batch {batch_size}, img_hw {img_hw}, input_c {input_c}, output_c {output_c}, " +
        f"kernel {kernel}, padding {padding}, stride {stride}"
    )
# def test_conv(
#     bias=False, set_values_to_one=True,
#     sid=0
# ):
    
    # batch_size = 128
    # input_c = 3
    # output_c = 64
    # img_hw = 224
    # kernel, padding, stride = 7, 3, 2

    # batch_size = 128
    # input_c = 512
    # output_c = 512
    # img_hw = 7
    # kernel, padding, stride = 3, 1, 1

    x_shape = [batch_size, input_c, img_hw, img_hw]

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape, ExecutionModeOptions.Enclave )
    input = torch.rand(x_shape)
    num_elem = torch.numel(input)
    # input = torch.arange(num_elem).reshape(x_shape).float() + 1
    if set_values_to_one:
        input.zero_()
        input += 0.5
    # print("input: ", input)

    test_layer = SGXConvBase(
        sid, f"TestSGXConv", ExecutionModeOptions.Enclave,
        output_c, kernel, stride, padding, is_enclave_mode=True, 
        bias=bias
    )
    

    output_layer = SecretOutputLayer(sid, "OutputLayer", ExecutionModeOptions.Enclave, inference=True)
    layers = [input_layer, test_layer, output_layer]
    
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)
    
    input_layer.StoreInEnclave = False

    plain_module = nn.Conv2d(
        input_c, output_c, kernel, stride, padding, 
        bias=bias
    )
    # num_elem = torch.numel(plain_module.weight.data)
    # plain_module.weight.data = torch.arange(num_elem).reshape(plain_module.weight.shape).float() 
    # print("Pytorch weight \n", plain_module.weight.data)
    
    if set_values_to_one:
        # out * in * h * w
        plain_module.weight.data.zero_()
        plain_module.weight.data += 0.5
    
    
    # print("Bias: ", plain_module.bias.data[:10])
    test_layer.inject_params(plain_module)
    plain_output = plain_module(input)

    # # out * in * h * w
    # weight_pytorch_idxs = [
    #     [3,9,0,2], [43,7,2,1], [63,15,2,0]
    # ]
    # values = [plain_module.weight.data[idxs[0],idxs[1],idxs[2],idxs[3]] for idxs in weight_pytorch_idxs]
    # print(
    #     "Weight: ", plain_module.weight.shape, values
    # )
    # # in * w * h * out
    # weight_tf_idxs = [
    #     [9,2,0,3], [7,1,2,43], [15,0,2,63]
    # ]
    # values = [test_layer.get_cpu("weight")[idxs[0],idxs[1],idxs[2],idxs[3]] for idxs in weight_tf_idxs]
    # print(
    #     "TFWeight: ", test_layer.get_cpu("weight").shape, values
    # )

    input_layer.set_input(input)
    secret_nn.forward()
    secret_nn.plain_forward()
    secret_output = output_layer.get_cpu("input")

    test_layer.show_plain_error_forward()
    
    # nobias_linear = nn.Linear(input_size, output_size, bias=False)
    # nobias_linear.weight.data = plain_module.weight.data
    # nobias_output = nobias_linear(input)
    # print("Nobias output: ", nobias_output[0,:10])
    output = plain_module(input)

    # print("Secret output: ", secret_output[3,0,3,2])
    # print("Expected output: ", output[3,0,3,2])

    # path = "debug_input"
    # debug_input = []
    # with open(path, "r") as f:
    #     lines = f.readlines()
    #     for l in lines:
    #         decode_l = [ float(d.strip()) for d in l.strip().split(",")[:-1]]
    #         debug_input.append(np.array(decode_l))
    # debug_input = np.array(debug_input)

    # debug_input = torch.Tensor(debug_input).flatten()
    # weight = plain_module.weight.data
    # weight = weight.permute(0,2,3,1).contiguous().flatten()
    # print(weight.dot(debug_input))

    # base_r, base_c = 5, 1
    # raw_input = []
    # for r in range(3):
    #     for c in range(3):
    #         raw_input.append(input[3, :, base_r+r, base_c+c].numpy())
    # raw_input = np.array(raw_input)
    # np.set_printoptions(linewidth=64, precision=4)
    # print(f"Base [{base_r},{base_c}] Raw Input : ")
    # for i in range(9):
    #     input_str = f"Line {i}: "
    #     for j in range(5):
    #         input_str += f"{raw_input[i,j]:.2f}"
    #         input_str += ", "
    #     # input_str = np.array_repr(raw_input[i]).replace('\n', '').replace('\t','')
    #     print(input_str)
    # raw_input = torch.Tensor(raw_input).flatten()
    # weight = plain_module.weight.data
    # weight = weight.permute(0,2,3,1).contiguous().flatten()
    # print("Raw input result: ", weight.dot(raw_input))

    # base_r, base_c = 5, 3
    # raw_input = []
    # for r in range(3):
    #     for c in range(3):
    #         raw_input.append(input[3, :, base_r+r, base_c+c].numpy())
    # raw_input = np.array(raw_input)
    # np.set_printoptions(linewidth=64, precision=4)
    # print(f"Base [{base_r},{base_c}] Raw Input : ")
    # for i in range(9):
    #     input_str = f"Line {i}: "
    #     for j in range(5):
    #         input_str += f"{raw_input[i,j]:.2f}"
    #         input_str += ", "
    #     # input_str = np.array_repr(raw_input[i]).replace('\n', '').replace('\t','')
    #     print(input_str)
    # raw_input = torch.Tensor(raw_input).flatten()
    # weight = plain_module.weight.data
    # weight = weight.permute(0,2,3,1).contiguous().flatten()
    # print("Raw input result: ", weight.dot(raw_input))

    # print("Plain output: ", output.permute(0,2,3,1).squeeze().view(-1, output_c))
    # print("Plain input: ", input[1,:10])

    GlobalTensor.destroy()

def single():
    seed_torch(123)
    batch_size=128

    # layer1

    test_conv(
        batch_size=batch_size, img_hw=28, input_c=64, output_c=128,
        kernel=3, padding=1, stride=1
    )

def resnet18():
    seed_torch(123)
    batch_size=512

    # layer1

    # test_conv(
    #     batch_size=batch_size, img_hw=56, input_c=64, output_c=64,
    #     kernel=3, padding=1, stride=1
    # )

    # layer2
    input_img_hw, middle_img_hw = 56, 28
    input_c, middle_c = 64, 128

    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=2
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c,
    #     kernel=1, padding=0, stride=2
    # )

    # layer3
    input_img_hw, middle_img_hw = 28, 14
    input_c, middle_c = 128, 256

    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=2
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c,
    #     kernel=1, padding=0, stride=2
    # )

    # layer4
    input_img_hw, middle_img_hw = 14, 7
    input_c, middle_c = 256, 512

    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=2
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c,
    #     kernel=1, padding=0, stride=2
    # )

def resnet50():
    seed_torch(123)
    batch_size=512

    # layer1
    input_img_hw, middle_img_hw = 56, 28
    input_c, middle_c = 64, 64

    # test_conv(
    #     batch_size=batch_size, img_hw=56, input_c=64, output_c=64,
    #     kernel=1, padding=0, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=56, input_c=64, output_c=64,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=56, input_c=64, output_c=64*4,
    #     kernel=1, padding=0, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=56, input_c=64*4, output_c=64,
    #     kernel=1, padding=0, stride=1
    # )

    # layer2
    input_img_hw, middle_img_hw = 56, 28
    input_c, middle_c = 256, 128

    # block0
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c,
    #     kernel=1, padding=0, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=middle_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=2
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c, output_c=middle_c*4,
    #     kernel=1, padding=0, stride=1
    # )
    # downsample
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c*4,
    #     kernel=1, padding=0, stride=2
    # )
    # block1
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c*4, output_c=middle_c,
    #     kernel=1, padding=0, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=1
    # )


    # layer3
    input_img_hw, middle_img_hw = 28, 14
    input_c, middle_c = 512, 256

    # block0
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c,
    #     kernel=1, padding=0, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=middle_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=2
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c, output_c=middle_c*4,
    #     kernel=1, padding=0, stride=1
    # )
    # downsample
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c*4,
    #     kernel=1, padding=0, stride=2
    # )
    # block1
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c*4, output_c=middle_c,
    #     kernel=1, padding=0, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=1
    # )

    # layer4
    input_img_hw, middle_img_hw = 14, 7
    input_c, middle_c = 1024, 512

    # block0
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c,
    #     kernel=1, padding=0, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=middle_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=2
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c, output_c=middle_c*4,
    #     kernel=1, padding=0, stride=1
    # )
    # downsample
    # test_conv(
    #     batch_size=batch_size, img_hw=input_img_hw, input_c=input_c, output_c=middle_c*4,
    #     kernel=1, padding=0, stride=2
    # )
    # block1
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c*4, output_c=middle_c,
    #     kernel=1, padding=0, stride=1
    # )
    # test_conv(
    #     batch_size=batch_size, img_hw=middle_img_hw, input_c=middle_c, output_c=middle_c,
    #     kernel=3, padding=1, stride=1
    # )

if __name__ == "__main__":
    single()
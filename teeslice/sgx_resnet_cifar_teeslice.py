import os
import sys
from pdb import set_trace as st
import collections
import os.path as osp

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
from python.layers.avgpool2d import SecretAvgpool2dLayer
from python.layers.output import SecretOutputLayer
from python.layers.relu import SecretReLULayer
from python.sgx_net import init_communicate, warming_up_cuda, SecretNeuralNetwork, SgdOptimizer
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.identity import SecretIdentityLayer
from python.layers.add import SecretAddLayer
from python.layers.weighted_add import SecretWeightedAddLayer

from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions
import teeslice.student_resnet_cifar as resnet_cifar

from teeslice.utils import *
from .resnet import *

device_cuda = torch.device("cuda:0")
torch.set_printoptions(precision=10)

def inject_res_block(block, pretrained_block):
    if isinstance(block, SecretBasicBlock):
        block.conv1.inject_params(pretrained_block.conv1)
        block.bn1.inject_params(pretrained_block.bn1)
        block.conv2.inject_params(pretrained_block.conv2)
        block.bn2.inject_params(pretrained_block.bn2)
        if len(block.downsample_layers) > 1:
            block.downsample_layers[0].inject_params(pretrained_block.conv_res)
            block.downsample_layers[1].inject_params(pretrained_block.bn_res)
    elif isinstance(block, SecretBottleneckBlock):
        block.conv1.inject_params(pretrained_block.conv1)
        block.bn1.inject_params(pretrained_block.bn1)
        block.conv2.inject_params(pretrained_block.conv2)
        block.bn2.inject_params(pretrained_block.bn2)
        block.conv3.inject_params(pretrained_block.conv3)
        block.bn3.inject_params(pretrained_block.bn3)
        if len(block.downsample_layers) > 1:
            block.downsample_layers[0].inject_params(pretrained_block.conv_res)
            block.downsample_layers[1].inject_params(pretrained_block.bn_res)
    else:
        raise NotImplementedError

def inject_proxy_block(block, pretrained_block):
    if isinstance(block, SecretBasicProxy):
        block.conv.inject_params(pretrained_block.conv)
        block.bn.inject_params(pretrained_block.bn)
    elif isinstance(block, SecretBottleneckProxy):
        block.conv1.inject_params(pretrained_block.conv1)
        block.bn1.inject_params(pretrained_block.bn1)
        block.conv2.inject_params(pretrained_block.conv2)
        block.bn2.inject_params(pretrained_block.bn2)
    else:
        raise NotImplementedError

class SecretBasicBlock():
    expansion = 1

    def __init__(self, inplanes, planes, sid, name_prefix, stride=1, downsample_layers=None, EnclaveMode=ExecutionModeOptions.Enclave):
        super(SecretBasicBlock, self).__init__()
        self.conv1 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv1",
            n_output_channel=planes, filter_hw=3, stride=stride, padding=1, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn1 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn1", EnclaveMode=EnclaveMode,
        )
        self.relu1 = SecretReLULayer(sid, f"{name_prefix}.relu1", EnclaveMode=EnclaveMode)

        self.conv2 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv2",
            n_output_channel=planes, filter_hw=3, stride=1, padding=1, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn2 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn2", EnclaveMode=EnclaveMode, 
            manually_register_next=True, link_next=False,
        )
        # self.relu2 = SecretReLULayer(sid, f"{name_prefix}.relu2", EnclaveMode=EnclaveMode)

        self.add = SecretAddLayer(
            sid=sid, LayerName=f"{name_prefix}.add", EnclaveMode=EnclaveMode,
            manually_register_prev=True
        )
        self.add.register_prev_layer(self.bn2)
        self.bn2.register_next_layer(self.add)

        layers = [
            self.conv1, self.bn1, self.relu1, self.conv2, self.bn2,
        ]
        self.downsample_layers = downsample_layers
        if downsample_layers is not None:
            layers += self.downsample_layers
            self.add.register_prev_layer(self.downsample_layers[-1])
            self.downsample_layers[-1].register_next_layer(self.add)
        layers.append(self.add)
        # layers.append(self.relu2)
        self.layers = layers

        # layers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2]
        # self.layers = layers
    def last_layer(self):
        return self.add
        # return self.relu2

    def __str__(self):
        info = f"SecretBasicBlock\n"
        info += f"\t{self.conv1.LayerName}: {self.conv1}\n"
        info += f"\t{self.bn1.LayerName}: {self.bn1}\n"
        info += f"\t{self.relu1.LayerName}: {self.relu1}"
        info += f"\t{self.conv2.LayerName}: {self.conv2}\n"
        info += f"\t{self.bn2.LayerName}: {self.bn2}\n"
        if self.downsample_layers is not None:
            if len(self.downsample_layers) == 1:
                info += f"\t{self.downsample_layers[0].LayerName}: {self.downsample_layers[0]}\n"
            elif len(self.downsample_layers) == 2:
                info += f"\t{self.downsample_layers[0].LayerName}: {self.downsample_layers[0]}"
                info += f"\t{self.downsample_layers[1].LayerName}: {self.downsample_layers[1]}\n"
        info += f"\t{self.add.LayerName}: {self.add}\n"
        # info += f"\t{self.relu2.LayerName}: {self.relu2}\n"
        return info
    
    def __repr__(self):
        return self.__str__()

class SecretBasicProxy():
    def __init__(self, inplanes, planes, sid, name_prefix, stride=1, EnclaveMode=ExecutionModeOptions.Enclave):
        super(SecretBasicProxy, self).__init__()
        self.stride = stride
        if stride > 1:
            self.maxpool = SecretMaxpool2dLayer(
                sid=sid, LayerName=f"{name_prefix}.maxpool", 
                filter_hw=stride+1, stride=stride, padding=stride//2,
                EnclaveMode=EnclaveMode,
                manually_register_prev=True, 
            )
        self.conv = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv",
            n_output_channel=planes, filter_hw=1, stride=1, padding=0, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn", EnclaveMode=EnclaveMode,
            link_next=False
        )

    def last_layer(self):
        return self.bn
    def first_layer(self):
        if self.stride > 1:
            return self.maxpool
        else:
            return self.conv
    def layer_list(self):
        if self.stride > 1:
            return [self.maxpool, self.conv, self.bn]
        else:
            return [self.conv, self.bn]

    def __str__(self):
        info = f"SecretBasicProxy\n"
        info += f"\t{self.conv.LayerName}: {self.conv}\n"
        info += f"\t{self.bn.LayerName}: {self.bn}\n"
        return info
    def __repr__(self):
        return self.__str__()


class SecretBottleneckBlock():
    expansion = 4

    def __init__(self, inplanes, planes, sid, name_prefix, stride=1, downsample_layers=None, EnclaveMode=ExecutionModeOptions.Enclave):
        super(SecretBottleneckBlock, self).__init__()
        self.stride = stride

        self.conv1 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv1",
            n_output_channel=planes, filter_hw=1, stride=1, padding=0, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn1 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn1", EnclaveMode=EnclaveMode,
        )
        self.relu1 = SecretReLULayer(sid, f"{name_prefix}.relu1", EnclaveMode=EnclaveMode)

        self.conv2 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv2",
            n_output_channel=planes, filter_hw=3, stride=stride, padding=1, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn2 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn2", EnclaveMode=EnclaveMode,
        )
        self.relu2 = SecretReLULayer(sid, f"{name_prefix}.relu2", EnclaveMode=EnclaveMode)

        self.conv3 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv3",
            n_output_channel=planes * self.expansion, filter_hw=1, stride=1, padding=0, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn3 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn3", EnclaveMode=EnclaveMode,
            manually_register_next=True, link_next=False,
        )
        # self.relu3 = SecretReLULayer(sid, f"{name_prefix}.relu3", EnclaveMode=EnclaveMode)

        self.add = SecretAddLayer(
            sid=sid, LayerName=f"{name_prefix}.add", EnclaveMode=EnclaveMode,
            manually_register_prev=True
        )
        self.add.register_prev_layer(self.bn3)
        self.bn3.register_next_layer(self.add)

        layers = [
            self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2, self.conv3, self.bn3
        ]
        self.downsample_layers = downsample_layers
        if downsample_layers is not None:
            layers += self.downsample_layers
            self.add.register_prev_layer(self.downsample_layers[-1])
            self.downsample_layers[-1].register_next_layer(self.add)
        layers.append(self.add)
        # layers.append(self.relu3)

        # layers = [
        #     self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2, self.conv3, self.bn3
        # ]
        self.layers = layers
        
    def last_layer(self):
        return self.add
        # return self.relu3
    def __str__(self):
        info = f"SecretBottleneck\n"
        info += f"\t{self.conv1.LayerName}: {self.conv1}\n"
        info += f"\t{self.bn1.LayerName}: {self.bn1}\n"
        info += f"\t{self.relu1.LayerName}: {self.relu1}"
        info += f"\t{self.conv2.LayerName}: {self.conv2}\n"
        info += f"\t{self.bn2.LayerName}: {self.bn2}\n"
        info += f"\t{self.relu2.LayerName}: {self.relu2}"
        info += f"\t{self.conv3.LayerName}: {self.conv3}\n"
        info += f"\t{self.bn3.LayerName}: {self.bn3}\n"
        if self.downsample_layers is not None:
            if len(self.downsample_layers) == 1:
                info += f"\t{self.downsample_layers[0].LayerName}: {self.downsample_layers[0]}\n"
            elif len(self.downsample_layers) == 2:
                info += f"\t{self.downsample_layers[0].LayerName}: {self.downsample_layers[0]}"
                info += f"\t{self.downsample_layers[1].LayerName}: {self.downsample_layers[1]}\n"
        info += f"\t{self.add.LayerName}: {self.add}\n"
        info += f"\t{self.relu3.LayerName}: {self.relu3}"
        return info
    
    def __repr__(self):
        return self.__str__()
class SecretBottleneckProxy():
    def __init__(self, inplanes, planes, sid, name_prefix, stride=1, EnclaveMode=ExecutionModeOptions.Enclave):
        super(SecretBottleneckProxy, self).__init__()
        self.stride = stride
        mid_planes = max(planes//32, 16)
        if stride > 1:
            self.maxpool = SecretMaxpool2dLayer(
                sid=sid, LayerName=f"{name_prefix}.maxpool", 
                filter_hw=stride+1, stride=stride, padding=stride//2,
                EnclaveMode=EnclaveMode,
                manually_register_prev=True, 
            )
        self.conv1 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv1",
            n_output_channel=mid_planes, filter_hw=1, stride=1, padding=0, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn1 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn1", EnclaveMode=EnclaveMode,
        )
        self.relu = SecretReLULayer(sid, f"{name_prefix}.relu", EnclaveMode=EnclaveMode)
        self.conv2 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv2",
            n_output_channel=planes, filter_hw=1, stride=1, padding=0, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn2 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn2", EnclaveMode=EnclaveMode,
            link_next=False
        )

    def last_layer(self):
        return self.bn2
    def first_layer(self):
        if self.stride > 1:
            return self.maxpool
        else:
            return self.conv1
    def layer_list(self):
        if self.stride > 1:
            return [self.maxpool, self.conv1, self.bn1, self.relu, self.conv2, self.bn2]
        else:
            return [self.conv1, self.bn1, self.relu, self.conv2, self.bn2]

    def __str__(self):
        info = f"SecretBottleneckProxy\n"
        info += f"\t{self.conv1.LayerName}: {self.conv1}\n"
        info += f"\t{self.bn1.LayerName}: {self.bn1}\n"
        info += f"\t{self.relu.LayerName}: {self.relu}\n"
        info += f"\t{self.conv2.LayerName}: {self.conv2}\n"
        info += f"\t{self.bn2.LayerName}: {self.bn2}\n"
        return info
    def __repr__(self):
        return self.__str__()

class SecretteesliceBlock():
    def __init__(
        self, main, proxy_prior_layers, prior_feat_shapes, hw, planes, proxy_block, sid, 
        name_prefix, pretrained_block, EnclaveMode=ExecutionModeOptions.Enclave
    ):
        # print(name_prefix, pretrained_block.alphas().data.cpu().numpy(), pretrained_block.keep_flag)
        # if name_prefix == "Layer3.0":
        #     st()
        self.BlockName = name_prefix
        self.main = main
        self.proxies = []
        self.layers = [l for l in main.layers]
        self.keep_flag = pretrained_block.keep_flag
        if sum(self.keep_flag[1:]) > 0:
            for idx, (prior_layer, feat_shape, keep_flag) in enumerate(zip(proxy_prior_layers, prior_feat_shapes, pretrained_block.keep_flag[1:])):
                if keep_flag == 0:
                    continue
                inhw, inplanes = feat_shape
                stride = inhw // hw
                proxy = proxy_block(
                    inplanes, planes*main.expansion, sid, f"{name_prefix}.proxies.{idx}", stride, EnclaveMode=EnclaveMode
                )
                self.proxies.append(proxy)
                proxy.first_layer().register_prev_layer(prior_layer)
                self.layers += proxy.layer_list()
            
            # if 
            #     weighted_add_enclave_mode = EnclaveMode
            # else:
            #     weighted_add_enclave_mode = ExecutionModeOptions.GPU
            self.add = SecretWeightedAddLayer(
                sid=sid, LayerName=f"{name_prefix}.weighted_add", EnclaveMode=EnclaveMode,
                manually_register_prev=True, num_layers=sum(pretrained_block.keep_flag)
            )
            self.main.last_layer().register_next_layer(self.add)
            self.add.register_main_prev_layer(self.main.last_layer())
            for layer in self.proxies:
                self.add.register_prev_layer(layer.last_layer())
                layer.last_layer().register_next_layer(self.add)
            self.layers.append(self.add)

            self.relu = SecretReLULayer(sid, f"{name_prefix}.relu", EnclaveMode=EnclaveMode)
            self.layers.append(self.relu)
        else:
            weighted_add_enclave_mode = ExecutionModeOptions.GPU
            self.relu = SecretReLULayer(sid, f"{name_prefix}.relu", EnclaveMode=weighted_add_enclave_mode)
            self.layers.append(self.relu)

    
    def last_layer(self):
        return self.relu

    def __str__(self):
        info = f"SecretteesliceBlock\n"
        info += f"Main:\n"
        info += str(self.main)
        for idx, proxy in enumerate(self.proxies):
            info += f"Proxy {idx}:\n"
            info += str(proxy)
        info += f"{self.add.LayerName}: {self.add}\n"
        return info

    def inject_pretrained_weight(self, pretrained_block):
        inject_res_block(self.main, pretrained_block.main)
        if sum(self.keep_flag[1:]) > 0:
            for proxy, pretrained_proxy in zip(self.proxies, pretrained_block.proxies):
                inject_proxy_block(proxy, pretrained_proxy)
            masked_alphas = [
                alpha for alpha, flag in zip(pretrained_block.alphas(), pretrained_block.keep_flag) if flag == 1
            ]
            # print(f"Injected weighted add alpha {masked_alphas}")
            self.add.set_weights(masked_alphas)



class SecretNNStudentResNetSmall(nn.Module):

    def __init__(
        self, block, layers, pretrained_model, num_classes=1000, batch_size=64, sid=0, EnclaveMode=ExecutionModeOptions.Enclave,
        img_hw = 32, inplanes = 64, max_skip=3, 
    ):
        self.inplanes = inplanes
        self.batch_size = batch_size
        self.img_hw = img_hw
        self.cur_hw = img_hw
        self.sid = sid
        self.max_skip = max_skip
        super(SecretNNStudentResNetSmall, self).__init__()

        if block is SecretBasicBlock:
            self.proxy_block = SecretBasicProxy
        else:
            self.proxy_block = SecretBottleneckProxy

        self.input_shape = [batch_size, 3, self.img_hw, self.img_hw]
        self.input_layer = SecretInputLayer(sid, "InputLayer", self.input_shape, ExecutionModeOptions.CPU)
        self.input_layer.StoreInEnclave = False
        
        header_enclave_mode = ExecutionModeOptions.CPU if EnclaveMode is ExecutionModeOptions.Enclave else EnclaveMode
        self.conv1 = SGXConvBase(
            sid, f"conv1",header_enclave_mode, 
            self.inplanes, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        
        self.bn1 = SecretBatchNorm2dLayer(sid, f"bn1",EnclaveMode=header_enclave_mode)
        self.relu1 = SecretReLULayer(sid, f"relu1",header_enclave_mode)
        # self.maxpool = SecretMaxpool2dLayer(sid, f"maxpool1", filter_hw=3, stride=2, padding=1,EnclaveMode=header_enclave_mode)
        self.sgx_layers = [
            self.input_layer, self.conv1, self.bn1, self.relu1, 
            # self.input_layer, self.conv1, self.bn1
        ]
        # self.output_layer = SecretOutputLayer(sid, "OutputLayer", inference=True)
        # self.sgx_layers.append(self.output_layer)
        self.proxy_prior_layers = collections.deque(maxlen=self.max_skip)
        self.proxy_prior_layers.appendleft(self.relu1)
        self.proxy_prior_layer_feat_shape = collections.deque(maxlen=self.max_skip)
        self.proxy_prior_layer_feat_shape.appendleft((self.cur_hw, self.inplanes))

        self.layer_blocks = []
        pretrained_blocks = pretrained_model.layers[:layers[0]]
        self.layer1_blocks, self.layer1 = self._make_layer(
            block, 64, layers[0], sid=sid, prev_layer=self.relu1, name_prefix="Layer1", 
            pretrained_blocks=pretrained_blocks, stride=1, EnclaveMode=EnclaveMode
        )
        self.sgx_layers += self.layer1
        self.layer_blocks += self.layer1_blocks

        pretrained_blocks = pretrained_model.layers[sum(layers[:1]):sum(layers[:2])]
        self.layer2_blocks, self.layer2 = self._make_layer(
            block, 128, layers[1], sid=sid, prev_layer=self.layer1[-1], name_prefix="Layer2", 
            pretrained_blocks=pretrained_blocks, stride=2, EnclaveMode=EnclaveMode
        )
        self.sgx_layers += self.layer2
        self.layer_blocks += self.layer2_blocks

        pretrained_blocks = pretrained_model.layers[sum(layers[:2]):sum(layers[:3])]
        self.layer3_blocks, self.layer3 = self._make_layer(
            block, 256, layers[2], sid=sid, prev_layer=self.layer2[-1], name_prefix="Layer3", 
            pretrained_blocks=pretrained_blocks, stride=2, EnclaveMode=EnclaveMode
        )
        self.sgx_layers += self.layer3
        self.layer_blocks += self.layer3_blocks

        pretrained_blocks = pretrained_model.layers[sum(layers[:3]):sum(layers[:4])]
        self.layer4_blocks, self.layer4 = self._make_layer(
            block, 512, layers[3], sid=sid, prev_layer=self.layer3[-1], name_prefix="Layer4", 
            pretrained_blocks=pretrained_blocks, stride=2, EnclaveMode=EnclaveMode
        )
        self.sgx_layers += self.layer4
        self.layer_blocks += self.layer4_blocks

        self.avgpool = SecretAvgpool2dLayer(sid, f"avgpool", filter_hw=self.img_hw//8, stride=1, padding=0,EnclaveMode=ExecutionModeOptions.CPU)
        self.sgx_layers.append(self.avgpool)
        self.flatten = SecretFlattenLayer(sid, f"flatten", EnclaveMode=ExecutionModeOptions.CPU)
        self.sgx_layers.append(self.flatten)
        self.fc = SGXLinearBase(sid, f"fc", EnclaveMode, batch_size, num_classes, 512 * block.expansion)
        self.sgx_layers.append(self.fc)

        self.output_layer = SecretOutputLayer(sid, "OutputLayer", ExecutionModeOptions.CPU, inference=True)
        self.sgx_layers.append(self.output_layer)


        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # # self.avgpool = nn.AvgPool2d(4, stride=1)
        # # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block, planes, blocks, prev_layer, sid, name_prefix, pretrained_blocks,
        stride=1, dropout=False, EnclaveMode=ExecutionModeOptions.Enclave):
        downsample_layer = []
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            downsample_conv = SGXConvBase(
                sid=self.sid, LayerName=f"{name_prefix}.0.main.downsample.conv",
                n_output_channel=planes * block.expansion, filter_hw=1, stride=stride, padding=0, 
                EnclaveMode=EnclaveMode, bias=False, manually_register_prev=True
            )
            downsample_conv.register_prev_layer(prev_layer)
            downsample_bn = SecretBatchNorm2dLayer(
                sid=self.sid, LayerName=f"{name_prefix}.0.main.downsample.bn", 
                EnclaveMode=EnclaveMode, manually_register_next=True, link_next=False,
            )
            downsample_layers = [downsample_conv, downsample_bn]
        else:
            downsample_conv = SecretIdentityLayer(
                sid=self.sid, LayerName=f"{name_prefix}.0.main.identity", EnclaveMode=EnclaveMode,
                manually_register_prev=True, manually_register_next=True, link_next=False,
            )
            downsample_conv.register_prev_layer(prev_layer)
            downsample_layers = [downsample_conv]

        if stride != 1:
            self.cur_hw = self.cur_hw // stride

        layers, layer_blocks = [], []
        start_res_block = block(
            self.inplanes, planes, sid=sid, name_prefix=f"{name_prefix}.0.main", 
            stride=stride, downsample_layers=downsample_layers, EnclaveMode=EnclaveMode)
        start_teeslice_block = SecretteesliceBlock(
            start_res_block, self.proxy_prior_layers, self.proxy_prior_layer_feat_shape, self.cur_hw, planes, 
            self.proxy_block, sid, f"{name_prefix}.0", pretrained_blocks[0]
        )
        layer_blocks.append(start_teeslice_block)
        layers += start_teeslice_block.layers
        self.inplanes = planes * block.expansion

        self.proxy_prior_layers.appendleft(start_teeslice_block.last_layer())
        self.proxy_prior_layer_feat_shape.appendleft((self.cur_hw, planes))

        prev_layer = start_teeslice_block.last_layer()
        for i in range(1, blocks):
            identity = SecretIdentityLayer(
                sid=self.sid, LayerName=f"{name_prefix}.{i}.main.identity", EnclaveMode=EnclaveMode,
                manually_register_prev=True, manually_register_next=True, link_next=False,
            )
            identity.register_prev_layer(prev_layer)
            identity_layers = [identity]

            middle_res_block = block(
                self.inplanes, planes, sid=sid, name_prefix=f"{name_prefix}.{i}.main", 
                stride=1, downsample_layers=identity_layers, EnclaveMode=EnclaveMode
            )
            middle_teeslice_block = SecretteesliceBlock(
                middle_res_block, self.proxy_prior_layers, self.proxy_prior_layer_feat_shape, self.cur_hw, planes,
                self.proxy_block, sid, f"{name_prefix}.{i}", pretrained_blocks[i]
            )
            layer_blocks.append(middle_teeslice_block)
            layers += middle_teeslice_block.layers
            prev_layer = middle_teeslice_block.last_layer()

            self.proxy_prior_layers.appendleft(middle_teeslice_block.last_layer())
            self.proxy_prior_layer_feat_shape.appendleft((self.cur_hw, planes*block.expansion))

        return layer_blocks, layers

    def inject_params(self, pretrained):
        assert not pretrained.training
        self.conv1.inject_params(pretrained.conv1)
        self.bn1.inject_params(pretrained.bn1)

        for block, pretrained_block in zip(self.layer_blocks, pretrained.layers):
            # print(block.BlockName)
            block.inject_pretrained_weight(pretrained_block)

        if hasattr(self, "fc"):
            self.fc.inject_params(pretrained.classifier)

def secret_resnet18(pretrained_model, EnclaveMode=ExecutionModeOptions.Enclave, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # resnet_constructor = SecretNNResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # resnet_constructor = SecretNNStudentResNetSmall(SecretBasicBlock, [2, 2, 2, 2], EnclaveMode=EnclaveMode, **kwargs)
    resnet_constructor = SecretNNStudentResNetSmall(SecretBasicBlock, [2, 2, 2, 2], pretrained_model, EnclaveMode=EnclaveMode, **kwargs)
    return resnet_constructor

def secret_resnet34(pretrained=False, EnclaveMode=ExecutionModeOptions.Enclave, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # resnet_constructor = SecretNNResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    resnet_constructor = SecretNNStudentResNetSmall(SecretBasicBlock, [3, 4, 6, 3], EnclaveMode=EnclaveMode, **kwargs)
    return resnet_constructor

def secret_resnet50(pretrained=False, EnclaveMode=ExecutionModeOptions.Enclave, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # resnet_constructor = SecretNNResNet(SecretBottleneck, [2, 2, 2, 2], **kwargs)
    resnet_constructor = SecretNNStudentResNetSmall(SecretBottleneckBlock, [3, 4, 6, 3], EnclaveMode=EnclaveMode, **kwargs)
    return resnet_constructor

def secret_resnet101(pretrained=False, EnclaveMode=ExecutionModeOptions.Enclave, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    resnet_constructor = SecretNNStudentResNetSmall(SecretBottleneckBlock, [3, 4, 23, 3], EnclaveMode=EnclaveMode, **kwargs)
    return resnet_constructor

def eval_sgx_teeslice_time(
    arch="resnet18", root="teeslice/cifar10", pretrained_subdir="cifar100-resnet18",
    teeslice_subdir="teeslice", teeslice_full_subdir="full_model", batch_size=64,
    num_classes=10, num_repeat=10, max_skip=3
):
    pretrained = resnet_cifar.create_model(arch, num_classes=num_classes, max_skip=max_skip)
    # pretrained = resnet_cifar.create_model("resnet50", num_classes=1000, max_skip=max_skip)
    # pretrained = resnet34(pretrained=False)
    # pretrained = resnet50(pretrained=False)
    # pretrained = resnet101(pretrained=False)

    pretrained_path = osp.join(root, pretrained_subdir, "checkpoint.pth.tar")
    if not osp.exists(pretrained_path):
        print(f"pretrained_path does not exist")
        print(f"pretrained_path is {pretrained_path}")
        raise RuntimeError
    teeslice_dir = osp.join(root, teeslice_subdir)
    if not osp.exists(osp.join(teeslice_dir, "checkpoint.pth.tar")):
        print(f"teeslice_dir does not exist")
        print(f"teeslice_dir is {teeslice_dir}/checkpoint.pth.tar")
        raise RuntimeError

    pretrained.load_pretrained(pretrained_path)
    load_checkpoint(pretrained, model_dir=teeslice_dir)
    pretrained.eval()
    print(pretrained.alphas_and_complexities())

    GlobalTensor.init()
    resnet_constructor = eval(f"secret_{arch}")(
        EnclaveMode=ExecutionModeOptions.GPU, max_skip=max_skip, num_classes=num_classes,
        pretrained_model=pretrained, batch_size=batch_size
    )
    # resnet_constructor = secret_resnet50(EnclaveMode=ExecutionModeOptions.GPU, max_skip=max_skip)
    # resnet_constructor = secret_resnet34(EnclaveMode=ExecutionModeOptions.Enclave)
    # resnet_constructor = secret_resnet50(EnclaveMode=ExecutionModeOptions.Enclave)
    # resnet_constructor = secret_resnet101(EnclaveMode=ExecutionModeOptions.Enclave)

    layers = resnet_constructor.sgx_layers
    secret_nn = SecretNeuralNetwork(0, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)
    for layer in layers:
        layer.print_connection_info()
    layers[0].print_tensor_link_relation()
    resnet_constructor.inject_params(pretrained)

    inf_times = []
    for i in range(num_repeat):
        input_shape = resnet_constructor.input_shape
        input = torch.rand(input_shape) 
        layers[0].set_input(input)

        with torch.no_grad():
            elapse_time = secret_nn.forward_with_time()
        print(f"Elapse time: {elapse_time:.4f} ms")
        inf_times.append(elapse_time)
        secret_output = resnet_constructor.output_layer.get_prediction()

        with torch.no_grad():
            plain_output = pretrained(input)

        # secret_middle = resnet_constructor.layer2_blocks[0].conv1.get_cpu("input")
        # err = compare_expected_actual(
        #     expected=middle, actual=secret_middle
        # )
        # print(f"Forward Error: {err}")

        err = compare_expected_actual(
            expected=plain_output, actual=secret_output
        )
        print(f"Forward Error: {err}")
    
    print(f"All inference time {inf_times}")
    return inf_times


if __name__=="__main__":
    eval_sgx_teeslice_time()

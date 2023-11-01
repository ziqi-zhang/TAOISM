#!/usr/bin/env python
from __future__ import print_function
import os
from itertools import product
from collections import defaultdict, namedtuple
from pdb import set_trace as st
from time import time

import torch
import torch.nn.functional as F
import torch.distributed as dist

from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.utils.basic_utils import str_hash
from python.enclave_interfaces import GlobalTensor
from python.utils.timer_utils import NamedTimerInstance, NetworkNamedTimerInstance
from python.common_torch import SecretConfig, mod_move_down, union_dicts, \
    get_random_uniform, calc_shape_conv2d_weight, GlobalCppExtension
from python.utils.torch_utils import compare_expected_actual, torch_sync
from python.tensor_loader import TensorLoader
from python.stateless_logger import StatelessLogger

torch.backends.cudnn.deterministic = True

LearnableParamTuple = namedtuple('LearnableParam', ('dw_name', 'w_name', 'shape'))


def conv2d_op(w, x, is_div=True):
    padding = 1

    batch_size, in_chan, img_hw, _ = x.size()
    out_chan, _, fil_hw, __ = w.size()
    y_shape = [batch_size, out_chan, img_hw, img_hw]
    dtype = x.dtype
    device = x.device
    is_cpu = True if device == torch.device("cpu") else False

    def base_conv2d(sub_x, sub_w):
        return F.conv2d(sub_x, sub_w, padding=padding)

    if is_cpu or (is_div is False):
        return base_conv2d(x, w)

    def sum_of_div(best_shape):
        best_batch_size, best_in_chan, best_out_chan = best_shape
        y = torch.zeros(y_shape, device=device, dtype=dtype)
        for idx_batch_size, idx_in_chan, idx_out_chan in product(range(batch_size // best_batch_size),
                                                                 range(in_chan // best_in_chan),
                                                                 range(out_chan // best_out_chan)):
            start_batch_size, end_batch_size = idx_batch_size * best_batch_size, (idx_batch_size + 1) * best_batch_size
            start_in_chan, end_in_chan = idx_in_chan * best_in_chan, (idx_in_chan + 1) * best_in_chan
            start_out_chan, end_out_chan = idx_out_chan * best_out_chan, (idx_out_chan + 1) * best_out_chan

            y[start_batch_size:end_batch_size, start_out_chan:end_out_chan, :, :] += \
                base_conv2d(x[start_batch_size:end_batch_size, start_in_chan:end_in_chan, :, :],
                            w[start_out_chan:end_out_chan, start_in_chan:end_in_chan, :, :])
        return y

    shapes_v100 = {
        (1024, 512, 512, 2): (1024, 512, 128),
        (1024, 512, 512, 4): (1024, 512, 128),
        (1024, 256, 512, 4): (1024, 128, 128),
        (1024, 256, 256, 8): (1024, 64, 128),
        (1024, 128, 256, 8): (1024, 64, 128),
        (512, 512, 512, 2): (512, 512, 128),
        (512, 512, 512, 4): (256, 256, 128),
        (512, 256, 512, 4): (256, 256, 128),
        (512, 256, 256, 8): (512, 128, 128),
        (512, 128, 256, 8): (512, 128, 128),
    }

    tunnable_shape = (batch_size, in_chan, out_chan, img_hw)
    if is_div and tunnable_shape in shapes_v100:
        return sum_of_div(shapes_v100[tunnable_shape])
    else:
        return base_conv2d(x, w)


def conv2d_input_grad_op(w, dy):
    return F.conv_transpose2d(dy, w, padding=1)


def conv2d_weight_grad_op(dy, x, is_div=True):
    batch_size, in_chan, img_hw, _ = x.size()
    _, out_chan, __, ___ = dy.size()
    w_shape = calc_shape_conv2d_weight(dy, x)
    dtype = x.dtype
    device = x.device
    is_cpu = True if device == torch.device("cpu") else False

    if is_cpu:
        return torch.transpose(F.conv2d(torch.transpose(x, 0, 1), torch.transpose(dy, 0, 1), padding=1), 0,
                               1).contiguous()

    def base_conv2d_weight_grad_op(sub_dy, sub_x):
        sub_w_shape = calc_shape_conv2d_weight(sub_dy, sub_x)
        return GlobalCppExtension.get_conv2d_cudnn().backward(sub_w_shape, sub_dy, sub_x, (1, 1), (1, 1), (1, 1), 1, 0, 0)

    if is_div is False:
        return base_conv2d_weight_grad_op(dy, x)

    def sum_of_div(best_shape):
        # print("running conv2d weight div")
        best_batch_size, best_in_chan, best_out_chan = best_shape
        dw = torch.zeros(w_shape, device=device, dtype=dtype)
        for idx_batch_size, idx_in_chan, idx_out_chan in product(range(batch_size // best_batch_size),
                                                                 range(in_chan // best_in_chan),
                                                                 range(out_chan // best_out_chan)):
            start_batch_size, end_batch_size = idx_batch_size * best_batch_size, (idx_batch_size + 1) * best_batch_size
            start_in_chan, end_in_chan = idx_in_chan * best_in_chan, (idx_in_chan + 1) * best_in_chan
            start_out_chan, end_out_chan = idx_out_chan * best_out_chan, (idx_out_chan + 1) * best_out_chan

            dw[start_out_chan:end_out_chan, start_in_chan:end_in_chan, :, :] += \
                base_conv2d_weight_grad_op(dy[start_batch_size:end_batch_size, start_out_chan:end_out_chan, :, :],
                                           x[start_batch_size:end_batch_size, start_in_chan:end_in_chan, :, :])
        return dw

    shapes_v100 = {
        (1024, 512, 512, 2): (1024, 512, 128),
        (1024, 512, 512, 4): (1024, 512, 128),
        (1024, 256, 512, 4): (1024, 128, 128),
        (1024, 128, 256, 8): (1024, 128, 128),
        (512, 512, 512, 2): (512, 512, 128),
        (512, 512, 512, 4): (512, 512, 128),
        (512, 256, 512, 4): (512, 128, 128),
        (512, 128, 256, 8): (128, 128, 256),
    }

    tunnable_shape = (batch_size, in_chan, out_chan, img_hw)
    if is_div and tunnable_shape in shapes_v100:
        return sum_of_div(shapes_v100[tunnable_shape])
    else:
        return base_conv2d_weight_grad_op(dy, x)


def matmul_op(w, x):
    return torch.mm(x, w.t())


def matmul_input_grad_op(w, dy):
    return torch.mm(dy, w)


def matmul_weight_grad_op(dy, x):
    return torch.mm(dy.t(), x)


def set_tensor_name_maybe_quantized(name, quantized):
    return name + ("Q" if quantized else "")

# target_op = conv2d_op
# idealC = ModOnCpu(target_op(AQ.type(torch.double), BQ.type(torch.double))).type(SecretConfig.dtypeForCpuOp)
# Forward
# A: Weight
# B: Input

# A: Weight
# B: dy
InputGradRemap = {
    "Af": "Af", "AQ": "AQ", "A0": "A0", "A1": "A1",
    "Bf": "DerCf", "BQ": "DerCQ", "B0": "DerC0", "B1": "DerC1",
    "E": "EForDerB", "F": "FForDerB",
    "C0": "C0ForDerB", "C1": "C1ForDerB", "CQ": "CQForDerB", "Cf": "CfForDerB", "Z": "ZForDerB",
}

# A: dy
# B: InputQ
WeightGradRemap = {
    "Af": "DerCf", "AQ": "DerCQ", "A0": "DerC0", "A1": "DerC1",
    "Bf": "Bf", "BQ": "BQ", "B0": "B0", "B1": "B1",
    "E": "EForDerA", "F": "FForDerA",
    "C0": "C0ForDerA", "C1": "C1ForDerA", "CQ": "CQForDerA", "Cf": "CfForDerA", "Z": "ZForDerA",
}



def secret_op_class_factory(sid, target_op_name):
    all_target_op = {"Matmul": matmul_op, "MatmulInputGrad": matmul_input_grad_op,
                     "MatmulWeightGrad": matmul_weight_grad_op,
                     "Conv2d": conv2d_op, "Conv2dInputGrad": conv2d_input_grad_op,
                     "Conv2dWeightGrad": conv2d_weight_grad_op}
    all_sid_class = {0: SecretBaseS0, 1: SecretBaseS1, 2: SecretBaseS2}

    target_op_func = all_target_op[target_op_name]
    sid_class = all_sid_class[sid]
    class_name = "Secret%sS%d" % (target_op_name, sid)

    def __init__(self, name):
        sid_class.__init__(self, name)

    # noinspection PyUnusedLocal
    def target_op(self, a, b):
        return target_op_func(a, b)

    new_class = type(class_name, (sid_class,), {"__init__": __init__, "target_op": target_op})
    return new_class


class SecretNeuralNetwork(TensorLoader):
    nn_name = None
    layers = None

    def __init__(self, sid, nn_name):
        super().__init__()
        self.sid = sid
        self.init(start_enclave=False)
        self.nn_name = nn_name

    def set_layers(self, layers):
        self.layers = layers

        if not isinstance(self.layers[0], SecretInputLayer):
            raise ValueError("The first layer has to be input layer")
        if not isinstance(self.layers[-1], SecretOutputLayer):
            raise ValueError("The last layer has to be output layer")
            
        for i in range(len(self.layers) - 1):
            PrevLayer = self.layers[i]
            NextLayer = self.layers[i + 1]
            if not PrevLayer.manually_register_next:
                PrevLayer.register_next_layer(NextLayer)
            if not NextLayer.manually_register_prev:
                NextLayer.register_prev_layer(PrevLayer)

        
        for layer in self.layers:
            # print(f"Init_shape/link layer {layer.LayerName}")
            layer.set_eid(self.get_eid())
            layer.init_shape()
            # if layer.LayerName in ["Layer1.0.weighted_add", "Layer1.0.proxies.0.bn"]:
            #     st()
            layer.link_tensors()
            # print(layer.LayerName)
            # layer.print_tensor_link_relation()
            # if layer.LayerName in ["Layer1.0.weighted_add", "Layer1.0.proxies.0.bn"]:
            #     st()
        
        for idx, layer in enumerate(self.layers):
            # print(f"Init layer {layer.LayerName}")
            # if layer.LayerName == "Layer1.0.main.relu2":
            #     st()
            layer.init(start_enclave=False)
            # if idx > 3:
            #     print(layer.LayerName, self.layers[4].get_cpu("input").shape, self.layers[4].PrevLayer.LayerName)

    def execute_for_each_layer(self, func, reverse=False):
        layers = self.layers[::-1] if reverse else self.layers
        for layer in layers:
            # print(f"SID: {self.sid} {layer.LayerName}, {func}")
            if self.sid == 2 and layer.IsDummyForS2:
                continue
            # print("Processing ", layer.LayerName)
            func(layer)
            
            # st()

    def classifier_output(self):
        with NamedTimerInstance(f"S{self.sid}: {self.nn_name} classifier_output"):
            self.forward()
            if self.sid == 2:
                return
            # layers: input_layer, ..., fc_layer, output_layer
            last_fc = self.layers[-2]
            last_fc.transfer_enclave_to_cpu("output")
            outputs = last_fc.get_cpu("output")
            _, predicted = torch.max(outputs.data, 1)
            return predicted

    def get_loss(self):
        return self.layers[-1].get_loss()

    def forward_with_time(self):
        def run_forward(layer):
            layer.forward()
        t0 = time()
        with NetworkNamedTimerInstance(f"S{self.sid}: {self.nn_name} Forward"):
            self.execute_for_each_layer(run_forward)
        t1 = time()
        # time in ms
        elapse_time = (t1 - t0) * (10 ** 3) 
        return elapse_time

    def forward(self):
        def run_forward(layer):
            layer.forward()
        with NetworkNamedTimerInstance(f"S{self.sid}: {self.nn_name} Forward"):
            self.execute_for_each_layer(run_forward)

    def backward(self):
        def run_backward(layer):
            layer.backward()
        with NamedTimerInstance(f"S{self.sid}: {self.nn_name} Backward"):
            self.execute_for_each_layer(run_backward, reverse=True)

    def plain_forward(self):
        with NetworkNamedTimerInstance(f"S{self.sid}: {self.nn_name} PlainForward"):
            self.execute_for_each_layer(lambda x: x.plain_forward())

    def plain_backward(self):
        with NetworkNamedTimerInstance(f"S{self.sid}: {self.nn_name} PlainBackward"):
            self.execute_for_each_layer(lambda x: x.plain_backward(), reverse=True)

    def show_plain_error(self):
        self.execute_for_each_layer(lambda x: x.show_plain_error())


# Take the registered learnable parameters list in layers and update them
# It may need take extra storage
# And execution depends on where the tensors are stored
# https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
class SgdOptimizer(TensorLoader):
    def __init__(self, sid):
        super().__init__()
        self.sid = sid
        self.learning_rate = 0.05
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.momentum_init_flags = defaultdict(lambda: False)
        self.ideal_momentum_buf = {}

        self.lr_gamma = 0.5
        self.lr_step = 30
        self.step_counter = 0

        self.layers = None

    def set_layers(self, layers):
        self.layers = layers

    def generate_tensor_name_list(self, force=False):
        # Run if forced or self.tensor_name_list is not generated
        if not force and self.tensor_name_list:
            return
        if self.sid == 2:
            return

        self.tensor_name_list = []
        for layer in self.layers:
            for (DerName, ParamName, shape) in layer.LearnableParamsList:
                self.tensor_name_list.append((ParamName + "Momentum", shape, None))

    def update_params(self, test_with_ideal=False):
        if self.sid == 2:
            return
        for layer in self.layers:
            self.update_params_in_layer(layer, test_with_ideal=test_with_ideal)

    def update_params_in_layer(self, layer, test_with_ideal=False):
        # ref: https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
        if layer.LearnableParamsList is None:
            return

        task_ids = []
        for (der_name, param_name, shape) in layer.LearnableParamsList:
            momentum_name = param_name + "Momentum"
            global_momentum_name = layer.name_modifier(momentum_name)

            if layer.StoreInEnclave:
                if test_with_ideal:
                    ideal_p, ideal_momentum = self.ideal_update_params_with_name(layer, der_name, param_name, shape)
                first_momentum = not self.momentum_init_flags[global_momentum_name]
                if first_momentum:
                    # print("FIRST MOMENTUM")
                    self.momentum_init_flags[global_momentum_name] = True
                    layer.init_enclave_tensor(momentum_name, shape)
                task_id = layer.sgd_update(param_name=param_name, grad_name=der_name, momentum_name=momentum_name,
                                           lr=self.learning_rate, momentum=self.momentum,
                                           weight_decay=self.weight_decay,
                                           first_momentum=first_momentum, is_async=True)
                if test_with_ideal:
                    while not self.get_task_status(task_id):
                        pass
                    layer.generate_cpu_tensor(momentum_name, shape)
                    layer.transfer_enclave_to_cpu(momentum_name)
                    layer.transfer_enclave_to_cpu(param_name)
                    param_err = compare_expected_actual(ideal_p, layer.get_cpu(param_name), get_relative=True)
                    print(f"S{self.sid}: {layer.LayerName} Param Error: {param_err}")
                    momentum_err = compare_expected_actual(ideal_momentum, layer.get_cpu(momentum_name), get_relative=True)
                    print(f"S{self.sid}: {layer.LayerName} Momentum Error: {momentum_err}")
                else:
                    task_ids.append(task_id)
            else:
                DerCpu = layer.get_cpu(der_name)
                ParamsCpu = layer.get_cpu(param_name)

                if test_with_ideal:
                    ideal_p, ideal_momentum = self.ideal_update_params_with_name(layer, der_name, param_name, shape)

                DerCpu.add_(self.weight_decay, ParamsCpu)

                if not self.momentum_init_flags[global_momentum_name]:
                    self.momentum_init_flags[global_momentum_name] = True
                    layer.generate_cpu_tensor(momentum_name, shape)
                    layer.get_cpu(momentum_name).copy_(DerCpu)
                    MomentumCpu = layer.get_cpu(momentum_name)
                else:
                    MomentumCpu = layer.get_cpu(momentum_name)
                    MomentumCpu.mul_(self.momentum).add_(1, DerCpu)

                ParamsCpu.add_(-self.learning_rate, MomentumCpu)

                if test_with_ideal:
                    param_err = compare_expected_actual(ideal_p, layer.get_cpu(param_name), get_relative=True)
                    print(f"S{self.sid}: {layer.LayerName} Param Error: {param_err}")
                    momentum_err = compare_expected_actual(ideal_momentum, layer.get_cpu(momentum_name), get_relative=True)
                    print(f"S{self.sid}: {layer.LayerName} Momentum Error: {momentum_err}")

        # Wait for all tasks to be finished
        for task_id in task_ids:
            while not self.get_task_status(task_id):
                pass

    def ideal_update_params_with_name(self, layer, der_name, param_name, shape):
        weight_decay = self.weight_decay
        momentum = self.momentum
        dampening = 0
        nesterov = False
        lr = self.learning_rate

        global_momentum_name = layer.name_modifier(param_name + 'Momentum')

        if layer.StoreInEnclave:
            layer.transfer_enclave_to_cpu(der_name)
            layer.transfer_enclave_to_cpu(param_name)
        d_p = torch.clone(layer.get_cpu(der_name)).detach()
        p = torch.clone(layer.get_cpu(param_name)).detach()

        if weight_decay != 0:
            d_p.add_(weight_decay, p)
        if global_momentum_name not in self.ideal_momentum_buf:
            buf = self.ideal_momentum_buf[global_momentum_name] = torch.clone(d_p).detach()
        else:
            buf = self.ideal_momentum_buf[global_momentum_name]
            buf.mul_(momentum).add_(1 - dampening, d_p)
        if nesterov:
            d_p = d_p.add(momentum, buf)
        else:
            d_p = buf
        p.add_(-lr, d_p)

        return p, buf


def warming_up_cuda():
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    print("Execution device: ", device)
    print("PyTorch version: ", torch.__version__)
    print("CUDA version: ", torch.version.cuda)
    print("CUDA device:", torch.cuda.get_device_name(0))

    batch_size, n_input_channel, n_output_channel, img_hw, filter_hw = 512, 512, 256, 4, 3
    x_shape = [batch_size, n_input_channel, img_hw, img_hw]
    w_shape = [n_output_channel, n_input_channel, filter_hw, filter_hw]
    with NamedTimerInstance("Warming up Cuda double"):
        dummy_a = get_random_uniform(SecretConfig.PrimeLimit, x_shape).type(SecretConfig.dtypeForSave)
        dummy_b = get_random_uniform(SecretConfig.PrimeLimit, w_shape).type(SecretConfig.dtypeForSave)
        F.conv2d(dummy_a.cuda().type(SecretConfig.dtypeForCudaMm), dummy_b.cuda().type(SecretConfig.dtypeForCudaMm),
                 padding=1)

    with NamedTimerInstance("Warming up Cuda dobule 2nd"):
        F.conv2d(dummy_a.cuda().type(torch.double), dummy_b.cuda().type(torch.double),
                 padding=1)

    with NamedTimerInstance("Warming up Cuda float"):
        F.conv2d(dummy_a.cuda().type(torch.float), dummy_b.cuda().type(torch.float), padding=1)

    with NamedTimerInstance("Warming up Cuda float 2nd"):
        F.conv2d(dummy_a.cuda().type(torch.float), dummy_b.cuda().type(torch.float), padding=1)

    batch_size, n_input_channel, n_output_channel, img_hw, filter_hw = 64, 64, 64, 8, 3
    x_shape = [batch_size, n_input_channel, img_hw, img_hw]
    w_shape = [n_output_channel, n_input_channel, filter_hw, filter_hw]
    with NamedTimerInstance("Warming up Cpu"):
        dummy_a = get_random_uniform(SecretConfig.PrimeLimit, x_shape).type(SecretConfig.dtypeForSave)
        dummy_b = get_random_uniform(SecretConfig.PrimeLimit, w_shape).type(SecretConfig.dtypeForSave)
        F.conv2d(dummy_a.type(SecretConfig.dtypeForCpuOp), dummy_b.type(SecretConfig.dtypeForCpuOp),
                 padding=1)

    with NamedTimerInstance("Warming up CppExtension"):
        GlobalCppExtension.get_conv2d_cudnn()


def init_communicate(rank, master_address, master_port, backend='gloo'):
    os.environ['MASTER_ADDR'] = master_address
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=SecretConfig.worldSize)

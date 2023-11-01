import numpy as np
import torch
from pdb import set_trace as st

from python.layers.activation import SecretActivationLayer
from python.sgx_net import LearnableParamTuple
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions
from python.global_config import SecretConfig


class SecretBatchNorm2dLayer(SecretActivationLayer):
    # https://pytorch.org/docs/stable/nn.html#batchnorm2d

    BatchSize = None
    NumChannel = None
    ImgH = None
    ImgW = None
    WeightShape = None
    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False, merge_own_tensors=False
    ):
        
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next, merge_own_tensors
        )
        
        self.ForwardFuncName = "BatchNorm2d"
        self.BackwardFuncName = "DerBatchNorm2d"
        self.PlainFunc = torch.nn.BatchNorm2d
        self.IsAffine = True
        self.momentum = 0.1
        self.IsCumulative = (self.momentum is None)
        self.epsilon = 1e-5

        if EnclaveMode is ExecutionModeOptions.CPU or EnclaveMode is ExecutionModeOptions.GPU:
            self.ForwardFunc = torch.nn.BatchNorm2d
        # if self.is_enclave_mode:
        #     self.StoreInEnclave = True
        # else:
        #     self.ForwardFunc = torch.nn.BatchNorm2d
        #     self.StoreInEnclave = False
        

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = self.InputShape
        self.BatchSize, self.NumChannel, self.ImgH, self.ImgW = self.InputShape
        self.WeightShape = [self.NumChannel]
        self.LearnableParamsList = [
            LearnableParamTuple(dw_name="DerWeight", w_name="weight", shape=self.WeightShape),
            LearnableParamTuple(dw_name="DerBias", w_name="bias", shape=self.WeightShape),
        ]
        

    # def init(self, start_enclave=True):
        
    #     if self.sid == 2:
    #         return
    #     TensorLoader.init(self, start_enclave)

    #     if self.is_enclave_mode:
    #         self.PlainFunc = self.PlainFunc(self.InputShape[1])
    #         self.PlainFunc.eval()
    #         self.get_cpu("weight").data.copy_(self.PlainFunc.weight.data)
    #         self.get_cpu("bias").data.copy_(self.PlainFunc.bias.data)
    #         self.get_cpu("RunMean").data.copy_(self.PlainFunc.running_mean.data)
    #         # inject sqrt(running_var) instead of running_var for precision
    #         self.get_cpu("RunVar").data.copy_(self.PlainFunc.running_var.data)
    #         self.transfer_cpu_to_enclave("weight")
    #         self.transfer_cpu_to_enclave("bias")
    #         self.transfer_cpu_to_enclave("RunMean")
    #         self.transfer_cpu_to_enclave("RunVar")
    #         self.batchnorm_init(
    #             self.LayerName,
    #             "input", "output", "weight", "bias",
    #             "DerInput", "DerOutput", "DerWeight", "DerBias",
    #             "RunMean", "RunVar", "CurMean", "CurVar",
    #             "mu",
    #             self.BatchSize, self.NumChannel, self.ImgH, self.ImgW,
    #             int(self.IsAffine), int(self.IsCumulative), self.momentum, self.epsilon)
    #     else:
    #         self.ForwardFunc = self.ForwardFunc(self.InputShape[1])
    #         self.PlainFunc = self.PlainFunc(self.InputShape[1])
    #         self.PlainFunc.eval()
    #         self.ForwardFunc.weight.data.copy_(self.PlainFunc.weight.data)
    #         self.ForwardFunc.bias.data.copy_(self.PlainFunc.bias.data)
    #         self.ForwardFunc.running_mean.data.copy_(self.PlainFunc.running_mean.data)
    #         self.ForwardFunc.running_var.data.copy_(self.PlainFunc.running_var.data)
    #         self.set_cpu("weight", list(self.ForwardFunc.parameters())[0].data)
    #         self.set_cpu("bias", list(self.ForwardFunc.parameters())[1].data)
    #         self.set_cpu("RunMean", self.ForwardFunc.running_mean.data)
    #         self.set_cpu("RunVar", self.ForwardFunc.running_var.data)
    #         self.ForwardFunc.eval()

    def init(self, start_enclave=True):
        # if self.LayerName == "Layer3.10.proxies.0.bn2":
        #     st()
        TensorLoader.init(self, start_enclave)

        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            self.PlainFunc = self.PlainFunc(self.InputShape[1])
            self.PlainFunc.eval()
            self.get_cpu("weight").data.copy_(self.PlainFunc.weight.data)
            self.get_cpu("bias").data.copy_(self.PlainFunc.bias.data)
            self.get_cpu("RunMean").data.copy_(self.PlainFunc.running_mean.data)
            # inject sqrt(running_var) instead of running_var for precision
            self.get_cpu("RunVar").data.copy_(self.PlainFunc.running_var.data)
            self.transfer_cpu_to_enclave("weight")
            self.transfer_cpu_to_enclave("bias")
            self.transfer_cpu_to_enclave("RunMean")
            self.transfer_cpu_to_enclave("RunVar")
            self.batchnorm_init(
                self.LayerName,
                "input", "output", "weight", "bias",
                # "DerInput", "DerOutput", "DerWeight", "DerBias",
                "RunMean", "RunVar", "CurMean", "CurVar",
                "mu",
                self.BatchSize, self.NumChannel, self.ImgH, self.ImgW,
                int(self.IsAffine), int(self.IsCumulative), self.momentum, self.epsilon)
        elif self.EnclaveMode is ExecutionModeOptions.CPU:
            self.ForwardFunc = self.ForwardFunc(self.InputShape[1])
            self.PlainFunc = self.PlainFunc(self.InputShape[1])
            self.PlainFunc.eval()
            self.ForwardFunc.weight.data.copy_(self.PlainFunc.weight.data)
            self.ForwardFunc.bias.data.copy_(self.PlainFunc.bias.data)
            self.ForwardFunc.running_mean.data.copy_(self.PlainFunc.running_mean.data)
            self.ForwardFunc.running_var.data.copy_(self.PlainFunc.running_var.data)
            self.set_cpu("weight", list(self.ForwardFunc.parameters())[0].data)
            self.set_cpu("bias", list(self.ForwardFunc.parameters())[1].data)
            self.set_cpu("RunMean", self.ForwardFunc.running_mean.data)
            self.set_cpu("RunVar", self.ForwardFunc.running_var.data)
            self.ForwardFunc.eval()
        elif self.EnclaveMode is ExecutionModeOptions.GPU:
            self.ForwardFunc = self.ForwardFunc(self.InputShape[1])
            self.PlainFunc = self.PlainFunc(self.InputShape[1])
            self.ForwardFunc.weight.data.copy_(self.PlainFunc.weight.data)
            self.ForwardFunc.bias.data.copy_(self.PlainFunc.bias.data)
            self.ForwardFunc.running_mean.data.copy_(self.PlainFunc.running_mean.data)
            self.ForwardFunc.running_var.data.copy_(self.PlainFunc.running_var.data)
            self.set_gpu("weight", list(self.ForwardFunc.parameters())[0].data)
            self.set_gpu("bias", list(self.ForwardFunc.parameters())[1].data)
            self.set_gpu("RunMean", self.ForwardFunc.running_mean.data)
            self.set_gpu("RunVar", self.ForwardFunc.running_var.data)
            self.PlainFunc.eval()
            self.ForwardFunc.cuda().eval()

    # def inject_params(self, params):
    #     if self.sid == -2:
    #         raise ValueError("S2 has no learnable parameters for injection")
    #     self.get_cpu("weight").copy_(params.weight.data)
    #     self.get_cpu("bias").copy_(params.bias.data)
    #     self.get_cpu("RunMean").copy_(params.running_mean.data)
    #     # inject sqrt(running_var) instead of running_var for precision
    #     self.get_cpu("RunVar").copy_(params.running_var.data)
    #     if self.is_enclave_mode:
    #         self.transfer_cpu_to_enclave("weight")
    #         self.transfer_cpu_to_enclave("bias")
    #         self.transfer_cpu_to_enclave("RunMean")
    #         self.transfer_cpu_to_enclave("RunVar")

    def inject_params(self, params):
        if self.sid == -2:
            raise ValueError("S2 has no learnable parameters for injection")
        if self.EnclaveMode in [ExecutionModeOptions.CPU, ExecutionModeOptions.Enclave]: 
            self.get_cpu("weight").copy_(params.weight.data)
            self.get_cpu("bias").copy_(params.bias.data)
            self.get_cpu("RunMean").copy_(params.running_mean.data)
            self.get_cpu("RunVar").copy_(params.running_var.data)
            if self.EnclaveMode is ExecutionModeOptions.Enclave:
                self.transfer_cpu_to_enclave("weight")
                self.transfer_cpu_to_enclave("bias")
                self.transfer_cpu_to_enclave("RunMean")
                self.transfer_cpu_to_enclave("RunVar")
        elif self.EnclaveMode is ExecutionModeOptions.GPU:
            self.get_gpu("weight").copy_(params.weight.data)
            self.get_gpu("bias").copy_(params.bias.data)
            self.get_gpu("RunMean").copy_(params.running_mean.data)
            self.get_gpu("RunVar").copy_(params.running_var.data)

    def reset_plain_bn(self):
        # module = torch.BatchNorm2d()
        self.get_cpu("weight").copy_(torch.ones(self.InputShape[1]))
        self.get_cpu("bias").copy_(torch.zeros(self.InputShape[1]))
        self.get_cpu("RunMean").copy_(torch.zeros(self.InputShape[1]))
        self.get_cpu("RunVar").copy_(torch.ones(self.InputShape[1]))
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            self.transfer_cpu_to_enclave("weight")
            self.transfer_cpu_to_enclave("bias")
            self.transfer_cpu_to_enclave("RunMean")
            self.transfer_cpu_to_enclave("RunVar")


    def inject_to_plain(self, plain_layer: torch.nn.Module) -> None:
        raise NotImplementedError
        if self.sid == -2:
            raise ValueError("S2 has no learnable parameters for injection")
        self.make_sure_cpu_is_latest("weight")
        self.make_sure_cpu_is_latest("bias")
        plain_layer.weight.data.copy_(self.get_cpu("weight"))
        plain_layer.bias.data.copy_(self.get_cpu("bias"))
        plain_layer.running_mean.data.copy_(self.get_cpu("RunMean"))
        plain_layer.running_var.data.copy_(self.get_cpu("RunVar"))

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        if self.sid == 2:
            self.tensor_name_list = {}
            return

        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            NeededTensorNames = [
                ("input", self.InputShape, None),
                # ("DerInput", self.InputShape, None),
                ("output", self.OutputShape, None),
                # ("DerOutput", self.OutputShape, None),
                ("weight", self.WeightShape, None),
                # ("DerWeight", self.WeightShape, None),
                ("bias", self.WeightShape, None),
                # ("DerBias", self.WeightShape, None),
                ("RunMean", self.WeightShape, None),
                ("CurMean", self.WeightShape, None),
                ("RunVar", self.WeightShape, None),
                ("CurVar", self.WeightShape, None),
                ("mu", self.InputShape, None),
            ]
        else:
            NeededTensorNames = [
                ("output", self.OutputShape, None),
                # ("DerInput", self.InputShape, None),
                ("input", self.InputShape, None),
                ("weight", self.WeightShape, None),
                # ("DerWeight", self.WeightShape, None),
                ("bias", self.WeightShape, None),
                # ("DerBias", self.WeightShape, None),
                # ("DerOutput", self.OutputShape, None)
            ]

        self.tensor_name_list = NeededTensorNames

    # def forward(self):
    #     if self.sid == 2:
    #         return
    #     with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
    #         if self.is_enclave_mode:
    #             self.forward_tensor_transfer()
    #             self.batchnorm_forward(self.LayerName, int(False))
    #         else:
    #             self.forward_tensor_transfer()
    #             self.requires_grad_on_cpu("input")
    #             self.ForwardFunc.bias.data.copy_(self.get_cpu("bias"))
    #             self.ForwardFunc.weight.data.copy_(self.get_cpu("weight"))
    #             self.ForwardFunc.running_mean.data.copy_(self.get_cpu("RunMean"))
    #             # running_var of PlainFunc is ^2 of that in the enclave
    #             enclave_running_var = self.get_cpu("RunVar")
    #             self.ForwardFunc.running_var.data.copy_(enclave_running_var)
    #             self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))

    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            if self.EnclaveMode == ExecutionModeOptions.Enclave:
                # if self.LayerName == "Layer2.0.downsample.bn":
                #     st()
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Input Preprocess", verbose_level=VerboseLevel.LAYER):
                    self.forward_tensor_transfer()
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} batchnorm_forward", verbose_level=VerboseLevel.LAYER):
                    self.batchnorm_forward(self.LayerName, int(False))
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                self.forward_tensor_transfer()
                self.ForwardFunc.bias.data.copy_(self.get_cpu("bias"))
                self.ForwardFunc.weight.data.copy_(self.get_cpu("weight"))
                self.ForwardFunc.running_mean.data.copy_(self.get_cpu("RunMean"))
                # running_var of PlainFunc is ^2 of that in the enclave
                enclave_running_var = self.get_cpu("RunVar")
                self.ForwardFunc.running_var.data.copy_(enclave_running_var)
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.forward_tensor_transfer()
                self.ForwardFunc.bias.data.copy_(self.get_gpu("bias"))
                self.ForwardFunc.weight.data.copy_(self.get_gpu("weight"))
                self.ForwardFunc.running_mean.data.copy_(self.get_gpu("RunMean"))
                # running_var of PlainFunc is ^2 of that in the enclave
                enclave_running_var = self.get_gpu("RunVar")
                self.ForwardFunc.running_var.data.copy_(enclave_running_var)
                # st()
                # print(self.get_gpu("input")[0,0,0])
                self.set_gpu("output", self.ForwardFunc(self.get_gpu("input").type(SecretConfig.dtypeForCpuOp)))

    def backward(self):
        raise NotImplementedError
        if self.sid == 2:
            return
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Backward", verbose_level=VerboseLevel.LAYER):
            if self.is_enclave_mode:
                self.backward_tensor_transfer()
                self.batchnorm_backward(self.LayerName)
            else:
                self.backward_tensor_transfer()
                BackwardInput, BackwardWeight, BackwardBias = self.get_cpu("output").grad_fn(self.get_cpu("DerOutput"))
                self.set_cpu("DerInput", BackwardInput.data)
                self.set_cpu("DerWeight", BackwardWeight.data)
                self.set_cpu("DerBias", BackwardBias.data)
                if list(self.get_cpu("DerWeight").shape) != self.WeightShape:
                    real_shape = self.get_cpu("DerWeight").shape
                    ideal_shape = self.WeightShape
                    raise ValueError(
                        f"DerWeight is not of shape self.AffineShape: real: {real_shape}, ideal: {ideal_shape}")
                if list(self.get_cpu("DerBias").shape) != self.WeightShape:
                    raise ValueError("DerBias is not of shape self.AffineShape")

    def plain_forward(self, NeedBackward=False):
        if self.sid == 2:
            return
        if self.EnclaveMode in [ExecutionModeOptions.Enclave, ExecutionModeOptions.GPU]:
            self.make_sure_cpu_is_latest("input")
            self.make_sure_cpu_is_latest("bias")
            self.make_sure_cpu_is_latest("weight")
            self.requires_grad_on_cpu("input")
            self.PlainFunc.bias.data.copy_(self.get_cpu("bias"))
            self.PlainFunc.weight.data.copy_(self.get_cpu("weight"))
            self.PlainFunc.running_mean.data.copy_(self.get_cpu("RunMean"))
            # self.PlainFunc.running_var.data.copy_(self.get_cpu("RunVar"))
            # running_var of PlainFunc is ^2 of that in the enclave
            enclave_running_var = self.get_cpu("RunVar")
            self.PlainFunc.running_var.data.copy_(enclave_running_var)
        else:
            self.make_sure_cpu_is_latest("input")
            self.requires_grad_on_cpu("input")

        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
            torch.set_num_threads(1)
            self.PlainForwardResult = self.PlainFunc(self.get_cpu("input"))
            torch.set_num_threads(4)

    def plain_backward(self):
        if self.sid == 2:
            return
        self.make_sure_cpu_is_latest("DerOutput")
        GradFunction = self.PlainForwardResult.grad_fn
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainBackward"):
            torch.set_num_threads(1)
            self.PlainBackwardResult = GradFunction(self.get_cpu("DerOutput"))
            torch.set_num_threads(4)

    def show_plain_error(self):
        if self.sid == 2:
            return
        self.make_sure_cpu_is_latest("output")
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"), get_relative=True)
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")

        if self.PlainBackwardResult is None:
            return
        if self.is_enclave_mode:
            self.make_sure_cpu_is_latest("DerInput")
            self.make_sure_cpu_is_latest("DerWeight")
            self.make_sure_cpu_is_latest("DerBias")
        else:
            self.make_sure_cpu_is_latest("DerInput")
        BackwardInput, BackwardWeight, BackwardBias = self.PlainBackwardResult
        err_input = compare_expected_actual(BackwardInput, self.get_cpu("DerInput"), show_where_err=False, get_relative=True)
        err_weight = compare_expected_actual(BackwardWeight, self.get_cpu("DerWeight"), show_where_err=False,
                                             get_relative=True)
        err_bias = compare_expected_actual(BackwardBias, self.get_cpu("DerBias"))
        print(f"S{self.sid}: {self.LayerName} Backward Error input: {err_input}, weight {err_weight}, bias: {err_bias}")

    def show_plain_error_forward(self):
        if self.sid == 2:
            return
        self.make_sure_cpu_is_latest("output")
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"), get_relative=False, show_values=False)
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")


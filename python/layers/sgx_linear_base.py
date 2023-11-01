from python.layers.base import SecretLayerBase
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.sgx_net import LearnableParamTuple
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions
from python.global_config import SecretConfig

import torch
from pdb import set_trace as st

class SGXLinearBase(SecretLayerBase):
    batch_size = None
    InputShape = None
    WeightShape = None
    OutputShape = None

    def __init__(
        self, sid, LayerName, EnclaveMode, batch_size, n_output_features, 
        n_input_features=None, is_enclave_mode=False, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)

        self.ForwardFuncName = "SGXLinear"
        self.BackwardFuncName = "DerSGXLinear"
        self.PlainFunc = torch.nn.Linear
        self.is_enclave_mode = is_enclave_mode
        self.n_output_features = n_output_features
        self.n_input_features = n_input_features
        self.batch_size = batch_size

        if EnclaveMode is ExecutionModeOptions.CPU or EnclaveMode is ExecutionModeOptions.GPU:
            self.ForwardFunc = torch.nn.Linear
        # if self.is_enclave_mode:
        #     self.StoreInEnclave = True
        # else:
        #     self.ForwardFunc = torch.nn.Linear
        #     self.StoreInEnclave = False

    def init_shape(self):
        self.WeightShape = self.DerWeightShape = [self.n_output_features, self.n_input_features]
        self.BiasShape = self.DerBiasShape = [self.n_output_features]
        if self.n_input_features is None:
            self.InputShape = self.PrevLayer.get_output_shape()
        else:
            self.InputShape = self.DerInputShape = [self.batch_size, self.n_input_features]
        self.OutputShape = self.DerOutputShape = [self.batch_size, self.n_output_features]
        self.LearnableParamsList = [
            LearnableParamTuple(dw_name="DerWeight", w_name="weight", shape=self.WeightShape),
            LearnableParamTuple(dw_name="DerBias", w_name="bias", shape=self.WeightShape),
        ]

    def init(self, start_enclave=True):
        TensorLoader.init(self, start_enclave)
        
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            self.PlainFunc = self.PlainFunc(self.n_input_features, self.n_output_features)
            self.get_cpu("weight").data.copy_(self.PlainFunc.weight.data)
            self.get_cpu("bias").data.copy_(self.PlainFunc.bias.data)
            self.transfer_cpu_to_enclave("weight")
            self.transfer_cpu_to_enclave("bias")
            self.sgx_linear_init(
                self.LayerName,
                "input", "output", "weight", "bias",
                # "DerInput", "DerOutput", "DerWeight", "DerBias",
                self.batch_size, self.n_input_features, self.n_output_features)
        else:
            self.ForwardFunc = self.ForwardFunc(self.n_input_features, self.n_output_features)
            self.PlainFunc = self.PlainFunc(self.n_input_features, self.n_output_features)
            self.ForwardFunc.weight.data.copy_(self.PlainFunc.weight.data)
            self.ForwardFunc.bias.data.copy_(self.PlainFunc.bias.data)
            if self.EnclaveMode is ExecutionModeOptions.CPU:
                self.set_cpu("weight", list(self.ForwardFunc.parameters())[0].data)
                self.set_cpu("bias", list(self.ForwardFunc.parameters())[1].data)
            elif self.EnclaveMode is ExecutionModeOptions.GPU:
                self.set_gpu("weight", list(self.ForwardFunc.parameters())[0].data)
                self.set_gpu("bias", list(self.ForwardFunc.parameters())[1].data)
                self.ForwardFunc.cuda()
        # print("======== SGX linear init finish")

    def link_tensors(self):
        super().link_tensors()

    def init_params(self):
        cpu_w = torch.zeros(self.w_shape)
        torch.nn.init.xavier_normal_(cpu_w, 1)
        self.set_tensor_cpu_enclave("weight", cpu_w)
        cpu_b = torch.zeros(self.b_shape)
        torch.nn.init.constant_(cpu_b, 0)
        self.set_tensor_cpu_enclave("bias", cpu_b)

    def get_output_shape(self):
        return self.OutputShape

    def inject_params(self, params):
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            cpu_w = self.get_cpu("weight")
            cpu_w.copy_(params.weight.data)
            self.transfer_cpu_to_enclave("weight")
            cpu_b = self.get_cpu("bias")
            cpu_b.copy_(params.bias.data)
            self.transfer_cpu_to_enclave("bias")
        elif self.EnclaveMode is ExecutionModeOptions.CPU:
            cpu_w = self.get_cpu("weight")
            cpu_w.copy_(params.weight.data)
            cpu_b = self.get_cpu("bias")
            cpu_b.copy_(params.bias.data)
        elif self.EnclaveMode is ExecutionModeOptions.GPU:
            cpu_w = self.get_gpu("weight")
            cpu_w.copy_(params.weight.data)
            cpu_b = self.get_gpu("bias")
            cpu_b.copy_(params.bias.data)

    def inject_to_plain(self, plain_layer: torch.nn.Module) -> None:
        self.make_sure_cpu_is_latest("weight")
        plain_layer.weight.data.copy_(self.get_cpu("weight"))
        self.make_sure_cpu_is_latest("bias")
        plain_layer.bias.data.copy_(self.get_cpu("bias"))

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        NeededTensorNames = [("output", self.OutputShape, None),
                            # ("DerInput", self.InputShape, None),
                            ("input", self.InputShape, None),
                            # ("DerOutput", self.OutputShape, None),
                            ("weight", self.WeightShape, None),
                            ("bias", self.BiasShape, None),
                            ]

        self.tensor_name_list = NeededTensorNames

    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            if self.EnclaveMode is ExecutionModeOptions.Enclave:
                self.forward_tensor_transfer()
                self.sgx_linear_forward(self.LayerName)
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                self.forward_tensor_transfer()
                self.requires_grad_on_cpu("input")
                self.ForwardFunc.weight.data.copy_(self.get_cpu("weight"))
                self.ForwardFunc.bias.data.copy_(self.get_cpu("bias"))
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.forward_tensor_transfer()
                self.ForwardFunc.weight.data.copy_(self.get_gpu("weight"))
                self.ForwardFunc.bias.data.copy_(self.get_gpu("bias"))
                self.set_gpu("output", self.ForwardFunc(self.get_gpu("input").type(SecretConfig.dtypeForCpuOp)))

    def plain_forward(self, NeedBackward=False):
        if self.is_enclave_mode:
            self.make_sure_cpu_is_latest("input")
            self.make_sure_cpu_is_latest("weight")
            self.make_sure_cpu_is_latest("bias")
            # self.requires_grad_on_cpu("input")
            self.PlainFunc.weight.data.copy_(self.get_cpu("weight"))
            self.PlainFunc.bias.data.copy_(self.get_cpu("bias"))
        else:
            self.make_sure_cpu_is_latest("input")
            self.requires_grad_on_cpu("input")
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
            # torch.set_num_threads(1)
            self.PlainForwardResult = self.PlainFunc(self.get_cpu("input"))
            # torch.set_num_threads(4)

    def show_plain_error_forward(self):
        self.make_sure_cpu_is_latest("output")
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"), get_relative=True)
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")


import torch

from python.layers.activation import SecretActivationLayer
from python.sgx_net import TensorLoader
from python.utils.basic_utils import ExecutionModeOptions
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel

from pdb import set_trace as st

class SecretAvgpool2dLayer(SecretActivationLayer):
    def __init__(
        self, sid, LayerName, EnclaveMode, filter_hw, stride, padding, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.ForwardFuncName = "Avgpool2d"
        self.BackwardFuncName = "DerAvgpool2d"
        self.filter_hw = filter_hw
        self.startmaxpool = False
        self.PlainFunc = torch.nn.AvgPool2d
        self.avgpoolpadding = padding
        self.stride = stride

        self.ForwardFunc = torch.nn.AvgPool2d

        # if is_enclave_mode:
        #     raise NotImplementedError
        #     self.ForwardFunc = self.maxpoolfunc
        #     self.BackwardFunc = self.maxpoolbackfunc
        #     self.StoreInEnclave = True
        # else:
        #     self.ForwardFunc = torch.nn.AvgPool2d
        #     self.StoreInEnclave = False

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        if len(self.InputShape) != 4:
            raise ValueError("Maxpooling2d apply only to 4D Tensor")
        if self.InputShape[2] != self.InputShape[3]:
            raise ValueError("The input tensor has to be square images")
        if self.InputShape[2] % self.stride != 0:
            raise ValueError("The input tensor needs padding for this filter size")
        InputHw = self.InputShape[2]
        output_hw = int( (InputHw + 2*self.avgpoolpadding - self.filter_hw) / self.stride ) + 1
        self.OutputShape = [self.InputShape[0], self.InputShape[1], output_hw, output_hw]
        self.HandleShape = self.InputShape

    def init(self, start_enclave=True):
        # if self.is_enclave_mode:
        #     raise NotImplementedError
        #     self.PlainFunc = self.PlainFunc(self.filter_hw)

        #     TensorLoader.init(self, start_enclave)
        #     if self.startmaxpool is False:
        #         self.startmaxpool = True
        #         return self.maxpoolinit(self.LayerName, "inputtrans", "outputtrans")
        # else:
        #     self.ForwardFunc = self.ForwardFunc(self.filter_hw, stride=self.stride, padding=self.avgpoolpadding)
        #     self.PlainFunc = self.PlainFunc(self.filter_hw, stride=self.stride, padding=self.avgpoolpadding)

        #     TensorLoader.init(self, start_enclave)
        self.ForwardFunc = self.ForwardFunc(self.filter_hw, stride=self.stride, padding=self.avgpoolpadding)
        self.PlainFunc = self.PlainFunc(self.filter_hw, stride=self.stride, padding=self.avgpoolpadding)

        TensorLoader.init(self, start_enclave)

    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            self.forward_tensor_transfer()
            # self.requires_grad_on_cpu("input")
            if self.EnclaveMode == ExecutionModeOptions.Enclave:
                if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.Enclave:
                    self.transfer_enclave_to_cpu("input")
                    if torch.sum(self.get_cpu("input").abs()) == 0:
                        raise RuntimeError(f"{self.LayerName}: SGX input not load")
                    self.transfer_cpu_to_enclave("input")
                self.transfer_enclave_to_cpu("input")
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
                self.transfer_cpu_to_enclave("output")
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.CPU and torch.sum(self.get_cpu("input").abs()) == 0:
                    raise RuntimeError(f"{self.LayerName}: SGX input not load")
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.GPU and torch.sum(self.get_gpu("input").abs()) == 0:
                    raise RuntimeError(f"{self.LayerName}: SGX input not load")
                self.set_gpu("output", self.ForwardFunc(self.get_gpu("input")))
            else:
                raise RuntimeError


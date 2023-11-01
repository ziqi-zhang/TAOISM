import torch

from python.layers.activation import SecretActivationLayer
from python.sgx_net import TensorLoader
from python.utils.basic_utils import ExecutionModeOptions
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel

from pdb import set_trace as st

class SecretMaxpool2dLayer(SecretActivationLayer):
    def __init__(
        self, sid, LayerName, EnclaveMode, filter_hw, stride, padding, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.ForwardFuncName = "Maxpool2d"
        self.BackwardFuncName = "DerMaxpool2d"
        self.filter_hw = filter_hw
        self.startmaxpool = False
        self.PlainFunc = torch.nn.MaxPool2d
        self.maxpoolpadding = padding
        self.stride = stride
        self.STORE_CHUNK_ELEM = 401408

        self.ForwardFunc = torch.nn.MaxPool2d

        if EnclaveMode == ExecutionModeOptions.Enclave :
            self.ForwardFunc = self.maxpoolfunc
            self.BackwardFunc = self.maxpoolbackfunc
        else:
            self.ForwardFunc = torch.nn.MaxPool2d

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        if len(self.InputShape) != 4:
            raise ValueError("Maxpooling2d apply only to 4D Tensor")
        if self.InputShape[2] != self.InputShape[3]:
            raise ValueError("The input tensor has to be square images")
        if self.InputShape[2] % self.stride != 0:
            raise ValueError("The input tensor needs padding for this filter size")
        InputHw = self.InputShape[2]
        output_hw = InputHw // self.stride
        self.OutputShape = [self.InputShape[0], self.InputShape[1], output_hw, output_hw]
        self.HandleShape = self.InputShape
        # self.Shapefortranspose = [int(round(((self.InputShape[0] * self.InputShape[1] * self.InputShape[2] * self.InputShape[3])/262144)+1/2)), 262144, 1, 1]
        self.Shapefortranspose = [
            int(round(((self.InputShape[0] * self.InputShape[1] * self.InputShape[2] * self.InputShape[3])/self.STORE_CHUNK_ELEM)+1/2)), self.STORE_CHUNK_ELEM, 1, 1]


    def init(self, start_enclave=True):
        if self.EnclaveMode == ExecutionModeOptions.Enclave:
            self.PlainFunc = self.PlainFunc(self.filter_hw, self.stride, self.maxpoolpadding)
            TensorLoader.init(self, start_enclave)

            if self.startmaxpool is False:
                self.startmaxpool = True
                return self.maxpoolinit(self.LayerName, "inputtrans", "outputtrans")
        else:
            self.ForwardFunc = self.ForwardFunc(self.filter_hw, stride=self.stride, padding=self.maxpoolpadding)
            self.PlainFunc = self.PlainFunc(self.filter_hw, stride=self.stride, padding=self.maxpoolpadding)

        #     TensorLoader.init(self, start_enclave)
        # self.ForwardFunc = self.ForwardFunc(self.filter_hw, stride=self.stride, padding=self.maxpoolpadding)
        # self.PlainFunc = self.PlainFunc(self.filter_hw, stride=self.stride, padding=self.maxpoolpadding)

        # TensorLoader.init(self, start_enclave)

    # def forward(self):
    #     with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
    #         self.forward_tensor_transfer()
    #         # self.requires_grad_on_cpu("input")
    #         if self.EnclaveMode == ExecutionModeOptions.Enclave:
    #             self.set_gpu("output", self.ForwardFunc(self.get_gpu("input")))
    #             st()

    #             # if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.Enclave:
    #             #     self.transfer_enclave_to_cpu("input")
    #             #     if torch.sum(self.get_cpu("input").abs()) == 0:
    #             #         raise RuntimeError(f"{self.LayerName}: SGX input not load")
    #             #     self.transfer_cpu_to_enclave("input")
    #             # self.transfer_enclave_to_cpu("input")
    #             # self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
    #             # self.transfer_cpu_to_enclave("output")
    #         elif self.EnclaveMode == ExecutionModeOptions.CPU:
    #             if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.CPU and torch.sum(self.get_cpu("input").abs()) == 0:
    #                 raise RuntimeError(f"{self.LayerName}: SGX input not load")
    #             self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
    #         elif self.EnclaveMode == ExecutionModeOptions.GPU:
    #             if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.GPU and torch.sum(self.get_gpu("input").abs()) == 0:
    #                 raise RuntimeError(f"{self.LayerName}: SGX input not load")
    #             self.set_gpu("output", self.ForwardFunc(self.get_gpu("input")))
    #         else:
    #             raise RuntimeError

    def maxpoolfunc(self, namein, nameout):
        # assume row_stride and col_stride are both None or both not None
        # assume row_pad and col_pad are both None or both not None
        # if self.LayerName == "Layer3.0.proxies.2.maxpool":
        #     print(self.LayerName, "Input: ", self.get_cpu("input")[0,0,0,:10])
        output = self.maxpoolnew(self.LayerName, namein, nameout, self.InputShape, self.OutputShape[2], self.OutputShape[3],
                               self.filter_hw, self.filter_hw, self.stride, self.stride, self.maxpoolpadding,
                               self.maxpoolpadding)
        # if self.LayerName == "Layer3.0.proxies.2.maxpool":
        #     self.transfer_enclave_to_cpu("output")
        #     print(self.LayerName, "Output: ", self.get_cpu("output")[0,0,0,:])
        #     self.transfer_cpu_to_enclave("output")
        return output

    def maxpoolbackfunc(self, nameout, namedout, namedin):
        return self.maxpoolback(self.LayerName, namedout, namedin, self.InputShape, self.OutputShape[2], self.OutputShape[3],
                                self.filter_hw, self.filter_hw, self.row_stride, self.col_stride, self.maxpoolpadding,
                                self.maxpoolpadding)
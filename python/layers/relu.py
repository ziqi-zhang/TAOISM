import torch
from pdb import set_trace as st

from python.layers.activation import SecretActivationLayer
from python.utils.basic_utils import ExecutionModeOptions
from python.utils.torch_utils import compare_expected_actual


class SecretReLULayer(SecretActivationLayer):
    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False, merge_own_tensors=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next, merge_own_tensors
        )
        self.ForwardFuncName = "ReLU"
        self.BackwardFuncName = "DerReLU"
        self.PlainFunc = torch.nn.ReLU
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            self.ForwardFunc = self.relufunc
            self.BackwardFunc = self.relubackfunc
        elif self.EnclaveMode is ExecutionModeOptions.CPU:
            self.ForwardFunc = torch.nn.ReLU
        elif self.EnclaveMode is ExecutionModeOptions.GPU:
            self.ForwardFunc = torch.nn.ReLU

        # if self.is_enclave_mode:
        #     self.ForwardFunc = self.relufunc
        #     self.BackwardFunc = self.relubackfunc
        #     self.StoreInEnclave = True
        # else:
        #     self.ForwardFunc = torch.nn.ReLU
        #     self.StoreInEnclave = False

    def init(self, start_enclave=True):
        super().init(start_enclave)
        self.PlainFunc = self.PlainFunc()
        # if not self.is_enclave_mode:
        if self.EnclaveMode is not ExecutionModeOptions.Enclave:
            self.ForwardFunc = self.ForwardFunc()

    def relufunc(self, namein, nameout):
        return self.relunew(namein, nameout, self.InputShape)

    def relubackfunc(self, nameout, namedout, namedin):
        return self.relubackward(nameout, namedout, namedin, self.InputShape)

    def show_plain_error_forward(self):
        if self.sid == 2:
            return
        self.make_sure_cpu_is_latest("output")
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"), get_relative=False, show_values=False)
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")


# Assume to be CrossEntropyLoss
import torch
from pdb import set_trace as st

from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions


class SecretOutputLayer(SecretNonlinearLayer):
    TargetShape = None
    loss = 0

    def __init__(
        self, sid, LayerName, EnclaveMode, inference=False, link_prev=True, link_next=True, 
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.ForwardFunc = torch.nn.CrossEntropyLoss()
        self.PlainFunc = torch.nn.CrossEntropyLoss()
        self.EnclaveMode = ExecutionModeOptions.CPU
        self.inference = inference


    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = [1]
        self.TargetShape = [self.InputShape[0]]  # number of Minibatch

    def init(self, start_enclave=True):
        TensorLoader.init(self, start_enclave)

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        if self.sid == 2:
            self.tensor_name_list = {}
            return

        NeededTensorNames = [
            ("output", self.OutputShape, None),
            ("DerInput", self.InputShape, None),
            ("input", self.InputShape, None),
            ("target", self.TargetShape, None),
        ]

        self.tensor_name_list = NeededTensorNames

    def load_target(self, tensor):
        self.set_tensor_with_name("target", tensor)

    def get_loss(self):
        return self.loss
    
    def get_prediction(self):
        self.forward_tensor_transfer("input")
        if torch.sum(self.get_cpu("input").abs()) == 0:
            raise RuntimeError("SGX input not load")
        return self.get_cpu("input")

    def forward(self):
        if not self.inference:
            with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
                self.forward_tensor_transfer()
                self.set_cpu("input", self.get_cpu("input").detach())
                self.requires_grad_on_cpu("input")
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input"), self.get_cpu("target")))
            loss = self.get_cpu("output").item()
            self.loss = loss

    def backward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Backward", verbose_level=VerboseLevel.LAYER):
            self.backward_tensor_transfer(transfer_tensor="output")
            self.get_cpu("output").backward()
            self.set_cpu("DerInput", self.get_cpu("input").grad)

    def plain_forward(self):
        if not self.inference:
            self.make_sure_cpu_is_latest("input")
            self.set_cpu("input", self.get_cpu("input").detach())
            self.requires_grad_on_cpu("input")
            with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
                self.PlainForwardResult = self.PlainFunc(self.get_cpu("input"), self.get_cpu("target"))

    def plain_backward(self):
        self.make_sure_cpu_is_latest("output")
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainBackward"):
            self.PlainForwardResult.backward()
            self.set_cpu("DerInput", self.get_cpu("input").grad)

    def show_plain_error(self):
        self.make_sure_cpu_is_latest("output")
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"))
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")

        if self.PlainBackwardResult is None:
            return
        self.make_sure_cpu_is_latest("DerInput")

        err = compare_expected_actual(self.PlainBackwardResult, self.get_cpu("DerInput"))
        print(f"S{self.sid}: {self.LayerName} Backward Error {err}")

    def print_connection_info(self):
        print(f"{self.LayerName:30} shape{self.InputShape}{' ':30} input {self.PrevLayer.LayerName:30}")


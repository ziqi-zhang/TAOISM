from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.basic_utils import ExecutionModeOptions

from pdb import set_trace as st
import torch

class SecretActivationLayer(SecretNonlinearLayer):
    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False, merge_own_tensors=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.Shapefortranspose = None
        self.link_prev = link_prev
        self.link_next = link_next
        self.manual_register_prev = manually_register_prev
        self.manual_register_next = manually_register_next
        self.merge_own_tensors = merge_own_tensors

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape

    def init(self, start_enclave=True):
        TensorLoader.init(self, start_enclave)

    def link_tensors(self):
        if self.merge_own_tensors:
            self.manually_link_owned_two_tensors("input", "output")
        super().link_tensors()


    def get_output_shape(self):
        return self.OutputShape

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        if self.sid == 2:
            self.tensor_name_list = {}
            return
        if len(self.InputShape) == 4:
            # self.Shapefortranspose = [int(round(((self.InputShape[0] * self.InputShape[1] * self.InputShape[2] * self.InputShape[3])/262144+1/2))), 262144, 1, 1]
            self.Shapefortranspose = [int(round(((self.InputShape[0] * self.InputShape[1] * self.InputShape[2] * self.InputShape[3])/602112+1/2))), 602112, 1, 1]
            
        else:
            self.Shapefortranspose = self.InputShape
        NeededTensorNames = [("output", self.OutputShape, None),
                            ("handle", self.HandleShape, None),
                            # ("DerInput", self.InputShape, None),
                            ("input", self.InputShape, None),
                            # ("DerOutput", self.OutputShape, None),
                            ("inputtrans", self.Shapefortranspose, None),
                            ("outputtrans", self.Shapefortranspose, None),
                            ]

        self.tensor_name_list = NeededTensorNames

    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Input Preprocess", verbose_level=VerboseLevel.LAYER):
                self.forward_tensor_transfer()
            # self.requires_grad_on_cpu("input")
            if self.EnclaveMode == ExecutionModeOptions.Enclave:
                # if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.Enclave:
                #     with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Input Preprocess", verbose_level=VerboseLevel.LAYER):
                #         self.transfer_enclave_to_cpu("input")
                #         if torch.sum(self.get_cpu("input").abs()) == 0:
                #             raise RuntimeError(f"{self.LayerName}: SGX input not load")
                #         self.transfer_cpu_to_enclave("input")
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} ForwardFunc", verbose_level=VerboseLevel.LAYER):
                    self.ForwardFunc("input", "output")
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

    def backward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Backward", verbose_level=VerboseLevel.LAYER):
            self.backward_tensor_transfer()
            if self.is_enclave_mode:
                self.BackwardFunc("output", "DerOutput", "DerInput")
            else:
                self.set_cpu("DerInput", self.get_cpu("output").grad_fn(self.get_cpu("DerOutput")))



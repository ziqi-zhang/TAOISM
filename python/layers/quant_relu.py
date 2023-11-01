import torch
from pdb import set_trace as st

from python.layers.activation import SecretActivationLayer
from python.utils.basic_utils import ExecutionModeOptions
from python.utils.torch_utils import compare_expected_actual
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel

import ctypes as C
from ctypes.util import find_library
import numpy as np

class SecretEnclaveQuantReLULayer(SecretActivationLayer):
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
        # if self.EnclaveMode is ExecutionModeOptions.Enclave:
        #     self.ForwardFunc = self.relufunc
        #     self.BackwardFunc = self.relubackfunc
        # elif self.EnclaveMode is ExecutionModeOptions.CPU:
        #     self.ForwardFunc = torch.nn.ReLU
        # elif self.EnclaveMode is ExecutionModeOptions.GPU:
        #     self.ForwardFunc = torch.nn.ReLU
        # self.ForwardFunc = self.quant_relufunc
        self.BackwardFunc = self.relubackfunc
        self.EnclaveMode = ExecutionModeOptions.GPU

    def init(self, start_enclave=True):
        super().init(start_enclave)
        self.PlainFunc = self.PlainFunc()

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape
        assert self.InputShape[1]%4 == 0
        self.QuantizedInputShape = [self.InputShape[0], self.InputShape[1]//4, self.InputShape[2], self.InputShape[3]]

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
                            ("quant_input", self.QuantizedInputShape, None),
                            ("quant_output", self.QuantizedInputShape, None),
                            ("inputtrans", self.Shapefortranspose, None),
                            ("outputtrans", self.Shapefortranspose, None),
                            ]

        self.tensor_name_list = NeededTensorNames

    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Input Preprocess", verbose_level=VerboseLevel.LAYER):
                self.forward_tensor_transfer()
                input = self.get_gpu("input").cpu()
                # ones = torch.ones(self.InputShape).to(torch.uint8)

                # symmetric quantization
                v_max, v_min = input.max(), input.min()
                abs_max = max(abs(v_max), abs(v_min))
                # input = input - v_min
                scale = 127 / abs_max
                input = torch.round(input * scale)

                input[input<-128] = -128
                input[input>127] = 127
                input += 128
                zero = (0-0)*scale
                zero = zero.to(torch.uint8)
                # print(f"zero {zero}")
                input = input.to(torch.uint8).contiguous()
                input_np = input.numpy()
                pointer, read_only_flag = input_np.__array_interface__['data']
                data_pointer = C.cast(pointer,C.POINTER(C.c_float))
                new_array = np.ctypeslib.as_array(data_pointer,shape=self.QuantizedInputShape)
                converted_input = torch.Tensor(new_array)
                # print(f"Scale {scale:.4f}, abs_max {abs_max:.4f}")
                # print("Quanted input:")
                # print(input)
                # print("Compressed input :")
                # print(converted_input)
                self.set_cpu("quant_input", converted_input)
                self.transfer_cpu_to_enclave("quant_input")
                # self.quant_relufunc("quant_input", "quant_output")
                self.quant_relunew("quant_input", "quant_output", self.QuantizedInputShape, scale, abs_max, zero)
                
                self.transfer_enclave_to_cpu("quant_output")
                quant_output = self.get_cpu("quant_output")
                quant_output_np = quant_output.numpy()
                pointer, read_only_flag = quant_output_np.__array_interface__['data']
                data_pointer = C.cast(pointer,C.POINTER(C.c_uint8))
                quant_array = np.ctypeslib.as_array(data_pointer,shape=self.InputShape)
                quant_array = (quant_array - 128) / scale

                quant_array = quant_array.float().cuda()
                self.set_gpu("output", quant_array)

                # print(quant_array)

                # new_pointer, read_only_flag = new_array.__array_interface__['data']
                # new_pointer = C.cast(new_pointer,C.POINTER(C.c_uint8))
                # recheck_array = np.ctypeslib.as_array(new_pointer,shape=self.InputShape)
                # st()


            # # self.requires_grad_on_cpu("input")
            # if self.EnclaveMode == ExecutionModeOptions.Enclave:
            #     with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} ForwardFunc", verbose_level=VerboseLevel.LAYER):
            #         self.ForwardFunc("input", "output")
            # elif self.EnclaveMode == ExecutionModeOptions.CPU:
            #     if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.CPU and torch.sum(self.get_cpu("input").abs()) == 0:
            #         raise RuntimeError(f"{self.LayerName}: SGX input not load")
            #     self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
            # elif self.EnclaveMode == ExecutionModeOptions.GPU:
            #     if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.GPU and torch.sum(self.get_gpu("input").abs()) == 0:
            #         raise RuntimeError(f"{self.LayerName}: SGX input not load")
            #     self.set_gpu("output", self.ForwardFunc(self.get_gpu("input")))
            # else:
            #     raise RuntimeError
        
                # print("123")

    # def quant_relufunc(self, namein, nameout):
    #     return self.quant_relunew(namein, nameout, self.QuantizedInputShape)

    def relubackfunc(self, nameout, namedout, namedin):
        return self.relubackward(nameout, namedout, namedin, self.InputShape)

    def show_plain_error_forward(self):
        if self.sid == 2:
            return
        self.make_sure_cpu_is_latest("output")
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"), get_relative=False, show_values=False)
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")


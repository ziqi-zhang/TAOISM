from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.enclave_interfaces import GlobalTensor as gt
from python.utils.basic_utils import ExecutionModeOptions

from pdb import set_trace as st

class SecretAddLayer(SecretNonlinearLayer):
    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.Shapefortranspose = None
        self.PrevLayer = []
        assert link_prev

    def register_prev_layer(self, layer):
        if layer not in self.PrevLayer:
            self.PrevLayer.append(layer)

    def init_shape(self):
        assert len(self.PrevLayer) == 2
        output_shape1 = self.PrevLayer[0].get_output_shape()
        output_shape2 = self.PrevLayer[1].get_output_shape()
        if not output_shape1 == output_shape2:
            print(self.PrevLayer[0].LayerName, output_shape1)
            print(self.PrevLayer[1].LayerName, output_shape2)
        assert output_shape1 == output_shape2
        self.InputShape = output_shape1
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape

    def init(self, start_enclave=True):
        TensorLoader.init(self, start_enclave)

    def get_output_shape(self):
        return self.OutputShape

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        NeededTensorNames = [("output", self.OutputShape, None),
                            ("input1", self.InputShape, None),
                            ("input", self.InputShape, None),
                            ]

        self.tensor_name_list = NeededTensorNames

    def link_tensors(self):
        if self.link_prev and self.PrevLayer is not None:
            gt.link_tags(self.get_tag("input1", remap=False), self.PrevLayer[0].get_tag("output", remap=False))
            # change "input2" to "input"
            gt.link_tags(self.get_tag("input2", remap=False), self.PrevLayer[1].get_tag("output", remap=False))
        if self.link_next and self.NextLayer is not None:
            gt.link_tags(self.get_tag("output", remap=False), self.NextLayer.get_tag("input", remap=False))


    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            # self.forward_tensor_transfer()
            # self.transfer_to_cpu("input1")
            # self.transfer_to_cpu("input2")
            assert self.PrevLayer[0] is not None and self.PrevLayer[1] is not None

            if self.PrevLayer[0].EnclaveMode is ExecutionModeOptions.Enclave and self.PrevLayer[1].EnclaveMode is ExecutionModeOptions.Enclave:
                self.transfer_enclave_to_cpu("input1")
                self.transfer_enclave_to_cpu("input2")
                input1 = self.get_cpu("input1")
                input2 = self.get_cpu("input2")
                self.set_cpu("output", input1+input2)
                self.transfer_from_cpu("output")
            elif self.PrevLayer[0].EnclaveMode is ExecutionModeOptions.CPU and self.PrevLayer[1].EnclaveMode is ExecutionModeOptions.CPU:
                input1 = self.get_cpu("input1")
                input2 = self.get_cpu("input2")
                self.set_cpu("output", input1+input2)
                assert self.EnclaveMode == ExecutionModeOptions.CPU
                # self.transfer_from_cpu("output")
            elif self.PrevLayer[0].EnclaveMode is ExecutionModeOptions.GPU and self.PrevLayer[1].EnclaveMode is ExecutionModeOptions.GPU:
                input1 = self.get_gpu("input1")
                input2 = self.get_gpu("input2")
                self.set_gpu("output", input1+input2)
                assert self.EnclaveMode == ExecutionModeOptions.GPU
                # self.transfer_from_gpu("output")
            elif self.PrevLayer[0].EnclaveMode is ExecutionModeOptions.Enclave and self.PrevLayer[1].EnclaveMode is ExecutionModeOptions.GPU:
                self.transfer_enclave_to_cpu("input1")
                input1 = self.get_cpu("input1").cuda()
                input2 = self.get_gpu("input2")
                self.set_gpu("output", input1+input2)
                assert self.EnclaveMode == ExecutionModeOptions.GPU

            else:
                print(f"PrevLayer0 {self.PrevLayer[0].LayerName} and PrevLayer1 {self.PrevLayer[1].LayerName} not consistent")
                print(f"PrevLayer0 {self.PrevLayer[0].LayerName} mode {self.PrevLayer[0].EnclaveMode}")
                print(f"PrevLayer1 {self.PrevLayer[1].LayerName} mode {self.PrevLayer[1].EnclaveMode}")
                st()
            
            # if self.PrevLayer[0] is not None and self.PrevLayer[0].EnclaveMode is ExecutionModeOptions.Enclave:
            #     self.transfer_enclave_to_cpu("input1")
            # if self.PrevLayer[0] is not None and self.PrevLayer[0].EnclaveMode is ExecutionModeOptions.GPU:
            #     self.transfer_gpu_to_cpu("input1")
            # if self.PrevLayer[1] is not None and self.PrevLayer[1].EnclaveMode is ExecutionModeOptions.Enclave:
            #     self.transfer_enclave_to_cpu("input2")
            # if self.PrevLayer[1] is not None and self.PrevLayer[1].EnclaveMode is ExecutionModeOptions.GPU:
            #     self.transfer_gpu_to_cpu("input2")
            # input1 = self.get_cpu("input1")
            # input2 = self.get_cpu("input2")
            # # print(self.PrevLayer[0].LayerName, "Input1: ", input1[0,0,0,:10])
            # # print(self.PrevLayer[1].LayerName, "Input2: ", input2[0,0,0,:10])
            # self.set_cpu("output", input1+input2)
            # self.transfer_from_cpu("output")
            # # if self.is_enclave_mode:
            # #     self.transfer_cpu_to_enclave("output")

    def backward(self):
        return super().backward()

    def forward_tensor_transfer(self):
        if self.PrevLayer[0] is not None and self.PrevLayer[0].StoreInEnclave is True and self.StoreInEnclave is False:
            self.transfer_enclave_to_cpu("input1")
        if self.PrevLayer[1] is not None and self.PrevLayer[1].StoreInEnclave is True and self.StoreInEnclave is False:
            self.transfer_enclave_to_cpu("input2")
        if self.PrevLayer[0] is not None and self.PrevLayer[0].StoreInEnclave is False and self.StoreInEnclave is True:
            self.transfer_cpu_to_enclave("input1")
        if self.PrevLayer[1] is not None and self.PrevLayer[1].StoreInEnclave is False and self.StoreInEnclave is True:
            self.transfer_cpu_to_enclave("input2")


    def print_connection_info(self):
        print(f"{self.LayerName:20} shape{self.InputShape}{' ':30} mode{self.EnclaveMode}{' ':20} input {self.PrevLayer[0].LayerName},{self.PrevLayer[1].LayerName} output {self.NextLayer.LayerName:20}")


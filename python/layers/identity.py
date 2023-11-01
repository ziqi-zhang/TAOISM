from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.enclave_interfaces import GlobalTensor as gt
from pdb import set_trace as st

class SecretIdentityLayer(SecretNonlinearLayer):
    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.Shapefortranspose = None

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
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
                            ("input", self.InputShape, None),
                            ]

        self.tensor_name_list = NeededTensorNames

    def link_tensors(self):
        if self.link_prev and self.PrevLayer is not None:
            gt.link_tags(self.get_tag("input", remap=False), self.PrevLayer.get_tag("output", remap=False))
        if self.link_next and self.NextLayer is not None:
            gt.link_tags(self.get_tag("output", remap=False), self.NextLayer.get_tag("input", remap=False))


    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            self.transfer_to_cpu("input")
            input = self.get_cpu("input")
            self.set_cpu("output", input.clone())
            # print("Identity: ", input[0,0,0,:10])
            self.transfer_from_cpu("output")


    def backward(self):
        return super().backward()

    def print_connection_info(self):
        print(f"{self.LayerName:20} shape{self.InputShape}{' ':30} input {self.PrevLayer.LayerName:20} output {self.NextLayer.LayerName:20}")



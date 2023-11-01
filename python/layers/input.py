from python.layers.nonlinear import SecretNonlinearLayer
from python.enclave_interfaces import GlobalTensor as gt
from pdb import set_trace as st

class SecretInputLayer(SecretNonlinearLayer):
    shape = None

    def __init__(
        self, sid, LayerName, input_shape, EnclaveMode, link_prev=True, link_next=True, 
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.shape = input_shape

    def link_tensors(self):
        gt.link_tags(self.get_tag("input", remap=False), self.get_tag("output", remap=False))
        super().link_tensors()

    def init_shape(self):
        return

    def set_input(self, tensor):
        self.set_tensor_cpu_gpu_enclave("input", tensor)

    def get_output_shape(self):
        return self.shape

    def forward(self):
        return

    def backward(self):
        return

    def plain_forward(self):
        return

    def plain_backward(self):
        return

    def show_plain_error(self):
        return

    def print_connection_info(self):
        print(f"{self.LayerName:30} shape{self.shape} output {self.NextLayer.LayerName:30}")



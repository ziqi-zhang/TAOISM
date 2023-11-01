from python.enclave_interfaces import GlobalTensor as gt
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.torch_utils import compare_expected_actual
from python.tensor_loader import TensorLoader
import torch
from python.utils.basic_utils import ExecutionModeOptions

from pdb import set_trace as st

class SecretLayerBase(TensorLoader):
    PrevLayer = None
    NextLayer = None
    PlainForwardResult = None
    PlainBackwardResult = None
    PlainBackwardResult = None
    LearnableParamsList = None
    StoreInEnclave = True
    IsDummyForS2 = True
    ForwardFunc = None
    BackwardFunc = None
    PlainFunc = None
    SecretOpList = None
    GradFunction = None
    grad_func_for_speed = None
    LearnableParamsList = None

    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True, 
        manually_register_prev=False, manually_register_next=False,
    ):
        super().__init__()
        self.sid = sid
        self.LayerName = LayerName
        self.link_prev = link_prev
        self.link_next = link_next
        self.manually_register_prev = manually_register_prev
        self.manually_register_next = manually_register_next
        self.EnclaveMode = EnclaveMode

    def set_eid(self, eid):
        super().set_eid(eid)
        # for f in self.SecretOpList:
        #     f.set_eid(eid)

    def init_shape(self):
        raise NotImplementedError

    def init_params(self):
        return

    def name_modifier(self, name):
        # if self.LayerName + "--" + str(name) == "Layer1.0.add--input":
        #     st()
        return self.LayerName + "--" + str(name)

    def link_tensors(self):
        if self.link_prev and self.PrevLayer is not None:
            gt.link_tags(self.get_tag("input", remap=False), self.PrevLayer.get_tag("output", remap=False))
            # gt.link_tags(self.get_tag("DerInput", remap=False), self.PrevLayer.get_tag("DerOutput", remap=False))
        if self.link_next and self.NextLayer is not None:
            gt.link_tags(self.get_tag("output", remap=False), self.NextLayer.get_tag("input", remap=False))
            # gt.link_tags(self.get_tag("DerOutput", remap=False), self.NextLayer.get_tag("DerInput", remap=False))

    def manually_link_owned_two_tensors(self, name1, name2):
        gt.link_tags(self.get_tag(name1, remap=False), self.get_tag(name2, remap=False))

    def register_next_layer(self, layer):
        if self.NextLayer is None:
            self.NextLayer = layer
        else:
            if isinstance(self.PrevLayer, list):
                prev_layer_names = [layer.LayerName for layer in self.PrevLayer]
                print(f"{self.LayerName} has already had PrevLayer {prev_layer_names}, can not register new NextLayer {layer.LayerName}")
            else:
                print(f"{self.LayerName} has already had PrevLayer {self.PrevLayer.LayerName}, can not register new NextLayer {layer.LayerName}")

    def register_prev_layer(self, layer):
        if self.PrevLayer is None:
            self.PrevLayer = layer
        else:
            print(f"{self.LayerName} has already had PrevLayer {self.PrevLayer.LayerName}, will not register {layer.LayerName}")

    # def forward_tensor_transfer(self, transfer_tensor="input"):
    #     if self.PrevLayer is not None and self.PrevLayer.StoreInEnclave is True and self.StoreInEnclave is False:
    #         self.transfer_enclave_to_cpu(transfer_tensor)
    #     if self.PrevLayer is not None and self.PrevLayer.StoreInEnclave is False and self.StoreInEnclave is True:
    #         self.transfer_cpu_to_enclave(transfer_tensor)

    def forward_tensor_transfer(self, transfer_tensor="input"):
        if self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.Enclave and self.EnclaveMode is ExecutionModeOptions.CPU:
            self.transfer_enclave_to_cpu(transfer_tensor)
        if self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.Enclave and self.EnclaveMode is ExecutionModeOptions.GPU:
            # self.transfer_enclave_to_cpu(transfer_tensor)
            # self.transfer_cpu_to_gpu(transfer_tensor)
            with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} Enclave to CPU", verbose_level=VerboseLevel.LAYER):
                self.transfer_enclave_to_cpu(transfer_tensor)
            with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} CPU to GPU", verbose_level=VerboseLevel.LAYER):
                self.transfer_cpu_to_gpu(transfer_tensor)
        if self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.CPU and self.EnclaveMode is ExecutionModeOptions.Enclave:
            self.transfer_cpu_to_enclave(transfer_tensor)
        if self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.CPU and self.EnclaveMode is ExecutionModeOptions.GPU:
            self.transfer_cpu_to_gpu(transfer_tensor)
        if self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.GPU and self.EnclaveMode is ExecutionModeOptions.Enclave:
            # self.transfer_gpu_to_cpu(transfer_tensor)
            # self.transfer_cpu_to_enclave(transfer_tensor)
            with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} GPU to CPU", verbose_level=VerboseLevel.LAYER):
                self.transfer_gpu_to_cpu(transfer_tensor)
            with NamedTimerInstance(f"          S{self.sid}: {self.LayerName} CPU to Enclave", verbose_level=VerboseLevel.LAYER):
                self.transfer_cpu_to_enclave(transfer_tensor)
        if self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.GPU and self.EnclaveMode is ExecutionModeOptions.CPU:
            self.transfer_gpu_to_cpu(transfer_tensor)

    # transfer to CPU regardless of the EnclaveMode of PrevLayer
    def transfer_to_cpu(self, transfer_tensor="input"):
        if self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.Enclave:
            self.transfer_enclave_to_cpu(transfer_tensor)
        if self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.GPU:
            self.transfer_gpu_to_cpu(transfer_tensor)

    # transfer cpu tensor to EnclaveMode device
    def transfer_from_cpu(self, transfer_tensor="input"):
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            self.transfer_cpu_to_enclave(transfer_tensor)
        if self.EnclaveMode is ExecutionModeOptions.GPU:
            self.transfer_cpu_to_gpu(transfer_tensor)

    def backward_tensor_transfer(self, transfer_tensor="DerOutput"):
        if self.NextLayer is not None and self.NextLayer.StoreInEnclave is True and self.StoreInEnclave is False:
            self.transfer_enclave_to_cpu(transfer_tensor)
        if self.NextLayer is not None and self.NextLayer.StoreInEnclave is False and self.StoreInEnclave is True:
            self.transfer_cpu_to_enclave(transfer_tensor)

    def set_tensor_with_name(self, name, t):
        raise NotImplementedError
        if t is not None:
            self.set_cpu(name, t)
            if self.StoreInEnclave:
                self.set_tensor(name, t)

    def forward_transfer_to_plain(self, name):
        raise NotImplementedError
        if self.PrevLayer is not None and self.PrevLayer.StoreInEnclave:
            self.transfer_enclave_to_cpu(name)

    def backward_transfer_to_plain(self, name):
        if self.NextLayer is not None and self.NextLayer.StoreInEnclave:
            self.transfer_enclave_to_cpu(name)

    # If this layer is store in enclave, then load the tensor from enclave to plaintext cpu
    def make_sure_cpu_is_latest(self, name):
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            self.transfer_enclave_to_cpu(name)
        if self.EnclaveMode is ExecutionModeOptions.GPU:
            self.transfer_gpu_to_cpu(name)
        # if self.StoreInEnclave:
        #     self.transfer_enclave_to_cpu(name)

    def load_tensors(self, input_tensor, der_output_tensor):
        self.set_tensor_with_name("input", input_tensor)
        # self.set_tensor_with_name("DerOutput", der_output_tensor)

    def requires_grad_on_cpu(self, name):
        tensor = self.get_cpu(name)
        if tensor.is_leaf is False:
            return
        tensor.requires_grad = True

    def plain_forward(self):
        self.make_sure_cpu_is_latest("input")
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
            torch.set_num_threads(1)
            self.PlainForwardResult = self.PlainFunc(self.get_cpu("input"))
            torch.set_num_threads(4)

    def plain_backward(self):
        self.make_sure_cpu_is_latest("DerOutput")
        GradFunction = self.PlainForwardResult.grad_fn
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainBackward"):
            torch.set_num_threads(1)
            self.PlainBackwardResult = GradFunction(self.get_cpu("DerOutput"))
            torch.set_num_threads(4)

    def show_plain_error(self):
        # if self.StoreInEnclave:
        #     self.transfer_enclave_to_cpu("output")
        self.make_sure_cpu_is_latest("output")
        # print("PlainForward: ", self.PlainForwardResult)
        # print("EnclaveForward: ", self.get_cpu("output"))
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"), get_relative=True)
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")

        # if self.PlainBackwardResult is None:
        #     return
        # if self.StoreInEnclave:
        #     self.transfer_enclave_to_cpu("DerInput")
        # err = compare_expected_actual(self.PlainBackwardResult, self.get_cpu("DerInput"), show_where_err=False, get_relative=True)
        # print(f"S{self.sid}: {self.LayerName} Backward Error {err}")

    def inject_params(self, param):
        return

    def print_connection_info(self):
        print(f"{self.LayerName:20} shape{self.InputShape}{' ':20} mode{self.EnclaveMode}{' ':20} input {self.PrevLayer.LayerName:20} output {self.NextLayer.LayerName:20}")

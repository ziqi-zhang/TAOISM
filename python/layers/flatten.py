from python.layers.nonlinear import SecretNonlinearLayer
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions


# Assume the prev. layer is of 4d. It outputs a 2d mat
# This layer doesnt pull the input in enclave if it is not so reduce duplicated action
class SecretFlattenLayer(SecretNonlinearLayer):
    batch_size = None
    n_features = None
    input_shape = None
    output_shape = None

    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.StoreInEnclave = False
        self.ForwardFuncName = "Flatten"
        self.BackwardFuncName = "DerFlatten"


    def init(self, start_enclave=True):
        super().init(start_enclave)
        self.ForwardFunc = lambda x: x.view(-1, self.n_features)
        self.PlainFunc = lambda x: x.view(-1, self.n_features)

    def init_shape(self):
        self.input_shape = self.PrevLayer.get_output_shape()
        if len(self.input_shape) != 4:
            return ValueError("The dimension of the tensor form prev. layer has to be 4D.")

        self.batch_size = self.input_shape[0]
        self.n_features = self.input_shape[1] * self.input_shape[2] * self.input_shape[3]
        self.output_shape = [self.batch_size, self.n_features]

    def get_output_shape(self):
        return self.output_shape

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        if self.sid == 2:
            self.tensor_name_list = {}
            return

        NeededTensorNames = [("output", self.output_shape, None),
                             ("input", self.input_shape, None),
                             ("DerInput", self.input_shape, None),
                             ("DerOutput", self.output_shape, None)
                             ]

        self.tensor_name_list = NeededTensorNames

    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            if self.EnclaveMode == ExecutionModeOptions.Enclave:
                self.transfer_enclave_to_cpu("input")
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
                self.transfer_cpu_to_enclave("output")
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.set_gpu("output", self.ForwardFunc(self.get_gpu("input")))

            # self.forward_tensor_transfer()
            # self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))

    def backward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Backward", verbose_level=VerboseLevel.LAYER):
            self.backward_tensor_transfer()
            self.set_cpu("DerInput", self.get_cpu("DerOutput").view(self.input_shape))

    def plain_forward(self, NeedBackward=False):
        self.requires_grad_on_cpu("input")
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
            self.PlainForwardResult = self.PlainFunc(self.get_cpu("input"))

    def plain_backward(self):
        self.make_sure_cpu_is_latest("DerOutput")
        GradFunction = self.PlainForwardResult.grad_fn
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainBackward"):
            self.PlainBackwardResult = GradFunction(self.get_cpu("DerOutput"))

    def show_plain_error(self):
        if self.StoreInEnclave:
            self.transfer_enclave_to_cpu("output")
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"))
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")

        if self.PlainBackwardResult is None:
            return
        err = compare_expected_actual(self.PlainBackwardResult, self.get_cpu("DerInput"), get_relative=True)
        print(f"S{self.sid}: {self.LayerName} Backward Error {err}")



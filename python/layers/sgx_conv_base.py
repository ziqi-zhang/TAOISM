from python.layers.base import SecretLayerBase
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.sgx_net import LearnableParamTuple
from python.utils.torch_utils import compare_expected_actual
from python.common_torch import calc_conv2d_output_shape_stride
from python.utils.basic_utils import ExecutionModeOptions
from python.global_config import SecretConfig

import torch
from pdb import set_trace as st

class SGXConvBase(SecretLayerBase):
    batch_size = None
    pytorch_x_shape, sgx_x_shape = None, None
    pytorch_w_shape, sgx_w_shape = None, None
    bias_shape = None
    pytorch_y_shape, sgx_y_shape = None, None

    def __init__(
        self, sid, LayerName, EnclaveMode,
        n_output_channel, filter_hw, stride, padding, batch_size=None, n_input_channel=None,
        img_hw=None, bias=True,
        is_enclave_mode=False, link_prev=True, link_next=True, manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)

        self.ForwardFuncName = "SGXConv"
        self.BackwardFuncName = "DerSGXConv"
        self.PlainFunc = torch.nn.Conv2d
        self.is_enclave_mode = is_enclave_mode
        self.batch_size = batch_size
        self.n_input_channel = n_input_channel
        self.n_output_channel = n_output_channel
        self.img_hw = img_hw
        self.filter_hw = filter_hw
        self.padding = padding
        self.stride = stride
        self.bias = bias

        if EnclaveMode is ExecutionModeOptions.CPU or EnclaveMode is ExecutionModeOptions.GPU:
            self.ForwardFunc = torch.nn.Conv2d

    # --------------
    # Add BIAS!!!!!
    # --------------

    def init_shape(self):
        if self.batch_size is None and self.PrevLayer is not None:
            self.pytorch_x_shape = self.PrevLayer.get_output_shape()
            self.batch_size, self.n_input_channel, self.img_hw, _ = self.pytorch_x_shape
        else:
            self.pytorch_x_shape = [self.batch_size, self.n_input_channel, self.img_hw, self.img_hw]
        # print(self.LayerName)
        # st()
        # BHWC
        self.sgx_x_shape = [self.pytorch_x_shape[0], self.pytorch_x_shape[2], self.pytorch_x_shape[3], self.pytorch_x_shape[1]]
        # pytorch weight is out * in * h * w
        self.pytorch_w_shape = [self.n_output_channel, self.n_input_channel, self.filter_hw, self.filter_hw]
        # w shape is in * w * h * out, the transpose of out * h * w * in
        self.sgx_w_shape = [self.n_output_channel, self.filter_hw, self.filter_hw, self.n_input_channel]
        # BCHW
        self.pytorch_y_shape = calc_conv2d_output_shape_stride(self.pytorch_x_shape, self.pytorch_w_shape, self.padding, self.stride)
        # BHWC
        self.sgx_y_shape = [self.pytorch_y_shape[0], self.pytorch_y_shape[2], self.pytorch_y_shape[3], self.pytorch_y_shape[1]]
        self.bias_shape = [self.n_output_channel]

        # print(
        #     f"Init_shape pytorch_input {self.pytorch_x_shape}, sgx_input {self.sgx_x_shape}, "
        #     f"pytorch_output {self.pytorch_y_shape}, sgx_output {self.sgx_y_shape}, "
        #     f"pytorch_weight {self.pytorch_w_shape}, sgx_weight {self.sgx_w_shape}, "
        #     f"bias {self.bias_shape}"
        # )

        self.LearnableParamsList = [
            LearnableParamTuple(dw_name="DerWeight", w_name="weight", shape=self.sgx_w_shape),
            LearnableParamTuple(dw_name="DerBias", w_name="bias", shape=self.bias_shape),
        ]

    def init(self, start_enclave=True):
        # print(f"Weight shape {self.sgx_w_shape}")
        TensorLoader.init(self, start_enclave)
        
        
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            self.PlainFunc = self.PlainFunc(
                self.n_input_channel, self.n_output_channel, self.filter_hw,
                self.stride, self.padding, bias=self.bias)
            weight_pytorch_form = self.PlainFunc.weight.data
            weight_tf_form = self.weight_pytorch2tf(weight_pytorch_form)
            self.get_cpu("weight").data.copy_(weight_tf_form)
            self.transfer_cpu_to_enclave("weight")
            # Bias
            if self.bias:
                bias_data = self.PlainFunc.bias.data
            else:
                bias_data = torch.zeros(self.bias_shape)
            self.get_cpu("bias").data.copy_(bias_data)
            self.transfer_cpu_to_enclave("bias")
            self.sgx_conv_init(
                self.LayerName,
                "sgx_input", "sgx_output", "weight", "bias",
                # "sgx_DerInput", "sgx_DerOutput", "DerWeight", "DerBias",
                # "input", "output", "weight", 
                # "DerInput", "DerOutput", "DerWeight", 
                self.batch_size, self.img_hw, self.img_hw, self.n_input_channel, 
                self.pytorch_y_shape[2], self.pytorch_y_shape[3], self.n_output_channel, 
                self.filter_hw, self.padding, self.stride)
        elif self.EnclaveMode in[ ExecutionModeOptions.CPU, ExecutionModeOptions.GPU]:
            self.ForwardFunc = self.ForwardFunc(
                self.n_input_channel, self.n_output_channel, self.filter_hw,
                self.stride, self.padding, bias=self.bias)
            self.PlainFunc = self.PlainFunc(
                self.n_input_channel, self.n_output_channel, self.filter_hw,
                self.stride, self.padding, bias=self.bias)
            self.ForwardFunc.weight.data.copy_(self.PlainFunc.weight.data)
            weight_pytorch_form = list(self.ForwardFunc.parameters())[0].data
            weight_tf_form = self.weight_pytorch2tf(weight_pytorch_form)
            if self.EnclaveMode is ExecutionModeOptions.CPU:
                self.set_cpu("weight", weight_tf_form)
                if self.bias:
                    self.ForwardFunc.bias.data.copy_(self.PlainFunc.bias.data)
                    bias_data = self.PlainFunc.bias.data
                    self.set_cpu("bias", bias_data)
            elif self.EnclaveMode is ExecutionModeOptions.GPU:
                self.set_gpu("weight", weight_tf_form)
                if self.bias:
                    self.ForwardFunc.bias.data.copy_(self.PlainFunc.bias.data)
                    bias_data = self.PlainFunc.bias.data
                    self.set_gpu("bias", bias_data)
                self.ForwardFunc.cuda()


    def link_tensors(self):
        super().link_tensors()

    def init_params(self):
        cpu_w = torch.zeros(self.sgx_w_shape)
        torch.nn.init.xavier_normal_(cpu_w, 1)
        self.set_tensor_cpu_gpu_enclave("weight", cpu_w)

    def get_output_shape(self):
        return self.pytorch_y_shape
    
    def weight_pytorch2tf(self, weight_pytorch_form):
        # weight_pytorch_form is out * in * h * w
        # out * (h * w) * in, 
        # h and w dont transpose
        # weight_tf_form = weight_pytorch_form.permute(1,3,2,0).contiguous()
        weight_tf_form = weight_pytorch_form.permute(0,2,3,1).contiguous()
        return weight_tf_form

    def weight_tf2pytorch(self, weight_tf_form):
        # weight_tf_form is out * (h * w) * in, the transpose of out * (h * w) * in
        # out * in * h * w
        # h and w dont transpose
        # weight_pytorch_form = weight_tf_form.permute(3, 0, 2, 1).contiguous()
        weight_pytorch_form = weight_tf_form.permute(0,3,1,2).contiguous()
        return weight_pytorch_form

    def feature_pytorch2tf(self, tensor_pytorch_form):
        # tensor_pytorch_form is b * in * h * w
        # b * h * w * in
        tensor_tf_form = tensor_pytorch_form.permute(0, 2, 3, 1).contiguous()
        return tensor_tf_form
    
    def feature_tf2pytorch(self, tensor_tf_form):
        # tensor_tf_form is b * h * w * in
        # b * in * h * w
        tensor_pytorch_form = tensor_tf_form.permute(0, 3, 1, 2).contiguous()
        return tensor_pytorch_form

    def inject_params(self, params):
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            cpu_w = self.get_cpu("weight")
            weight_pytorch_form = params.weight.data
            weight_tf_form = self.weight_pytorch2tf(weight_pytorch_form)
            cpu_w.copy_(weight_tf_form)
            self.transfer_cpu_to_enclave("weight")

            # bias
            assert (
                (self.bias and params.bias is not None) or
                (not self.bias and params.bias is None)
            )
            if self.bias:
                bias_data = params.bias.data
            else:
                bias_data = torch.zeros(self.n_output_channel)
            cpu_b = self.get_cpu("bias")
            cpu_b.copy_(bias_data)
            self.transfer_cpu_to_enclave("bias")
        elif self.EnclaveMode is ExecutionModeOptions.CPU:
            weight_pytorch_form = params.weight.data
            weight_tf_form = self.weight_pytorch2tf(weight_pytorch_form)
            self.get_cpu("weight").copy_(weight_tf_form)
            # bias
            assert (
                (self.bias and params.bias is not None) or
                (not self.bias and params.bias is None)
            )
            if self.bias:
                self.get_cpu("bias").copy_(params.bias.data)

            # Move weight to ForwardFunc
            weight_tf_form = self.get_cpu("weight")
            weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
            self.ForwardFunc.weight.data.copy_(weight_pytorch_form)

        elif self.EnclaveMode is ExecutionModeOptions.GPU:
            weight_pytorch_form = params.weight.data
            weight_tf_form = self.weight_pytorch2tf(weight_pytorch_form)
            self.get_gpu("weight").copy_(weight_tf_form)
            # bias
            assert (
                (self.bias and params.bias is not None) or
                (not self.bias and params.bias is None)
            )
            if self.bias:
                self.get_gpu("bias").copy_(params.bias.data)

            # Move weight to ForwardFunc
            weight_tf_form = self.get_gpu("weight")
            weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
            self.ForwardFunc.weight.data.copy_(weight_pytorch_form)


    def inject_to_plain(self, plain_layer: torch.nn.Module) -> None:
        self.make_sure_cpu_is_latest("weight")
        weight_tf_form = self.get_cpu("weight")
        weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
        plain_layer.weight.data.copy_(weight_pytorch_form)

        assert (
            (self.bias and plain_layer.bias is not None) or
            (not self.bias and plain_layer.bias is None)
        )
        if self.bias:
            self.make_sure_cpu_is_latest("bias")
            bias_data = self.get_cpu("bias")
            plain_layer.weight.data.copy_(bias_data)

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        NeededTensorNames = [("output", self.pytorch_y_shape, None), ("sgx_output", self.sgx_y_shape, None),
                            ("DerInput", self.pytorch_x_shape, None), ("sgx_DerInput", self.sgx_x_shape, None),
                            ("input", self.pytorch_x_shape, None), ("sgx_input", self.sgx_x_shape, None),
                            ("DerOutput", self.pytorch_y_shape, None), ("sgx_DerOutput", self.sgx_y_shape, None),
                            ("weight", self.sgx_w_shape, None),
                            ("bias", self.bias_shape, None),
                            ]
        self.tensor_name_list = NeededTensorNames


    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            self.forward_tensor_transfer("input")
            if self.EnclaveMode == ExecutionModeOptions.Enclave:
                
                # "input" is pytorch form
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Input Preprocess", verbose_level=VerboseLevel.LAYER):
                    if self.PrevLayer.EnclaveMode is ExecutionModeOptions.Enclave:
                        self.transfer_enclave_to_cpu("input")
                    input_pytorch_form = self.get_cpu("input")
                    
                    if torch.sum(self.get_cpu("input").abs()) == 0:
                        print(self.LayerName)
                        raise RuntimeError("SGX input not load")
                    input_tf_form = self.feature_pytorch2tf(input_pytorch_form)
                    self.set_cpu("sgx_input", input_tf_form)
                    self.transfer_cpu_to_enclave("sgx_input")
                    # self.forward_tensor_transfer("sgx_input")
                    # print(self.get_cpu("sgx_input").squeeze())
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} sgx_conv_forward", verbose_level=VerboseLevel.LAYER):
                    # if self.LayerName == "Layer2.0.downsample.conv":
                    #     st()
                    self.sgx_conv_forward(self.LayerName)
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Output Postprocess", verbose_level=VerboseLevel.LAYER):
                    self.make_sure_cpu_is_latest("sgx_output")
                    output_tf_form = self.get_cpu("sgx_output")
                    output_pytorch_form = self.feature_tf2pytorch(output_tf_form)
                    self.set_cpu("output", output_pytorch_form)
                    self.transfer_cpu_to_enclave("output")
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Input Preprocess", verbose_level=VerboseLevel.LAYER):
                    self.forward_tensor_transfer()
                # self.requires_grad_on_cpu("input")
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Weight Transfer", verbose_level=VerboseLevel.LAYER):
                    with NamedTimerInstance(f"      S{self.sid}: {self.LayerName} get weight_tf_form", verbose_level=VerboseLevel.LAYER):
                        weight_tf_form = self.get_cpu("weight")
                    with NamedTimerInstance(f"      S{self.sid}: {self.LayerName} weight_tf2pytorch", verbose_level=VerboseLevel.LAYER):
                        weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
                    with NamedTimerInstance(f"      S{self.sid}: {self.LayerName} copy data", verbose_level=VerboseLevel.LAYER):
                        self.ForwardFunc.weight.data.copy_(weight_pytorch_form)
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} GPU conv forward", verbose_level=VerboseLevel.LAYER):
                    self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Input Preprocess", verbose_level=VerboseLevel.LAYER):
                    self.forward_tensor_transfer()
                # self.requires_grad_on_cpu("input")
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Weight Transfer", verbose_level=VerboseLevel.LAYER):
                    with NamedTimerInstance(f"      S{self.sid}: {self.LayerName} get weight_tf_form", verbose_level=VerboseLevel.LAYER):
                        weight_tf_form = self.get_gpu("weight")
                    with NamedTimerInstance(f"      S{self.sid}: {self.LayerName} weight_tf2pytorch", verbose_level=VerboseLevel.LAYER):
                        weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
                    with NamedTimerInstance(f"      S{self.sid}: {self.LayerName} copy data", verbose_level=VerboseLevel.LAYER):
                        self.ForwardFunc.weight.data.copy_(weight_pytorch_form)
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} GPU conv forward", verbose_level=VerboseLevel.LAYER):
                    self.set_gpu("output", self.ForwardFunc(self.get_gpu("input").type(SecretConfig.dtypeForCpuOp)))


    def plain_forward(self, NeedBackward=False):
        if self.EnclaveMode == ExecutionModeOptions.Enclave:
            self.make_sure_cpu_is_latest("input")
            self.make_sure_cpu_is_latest("weight")
            if self.bias:
                self.make_sure_cpu_is_latest("bias")
            # self.requires_grad_on_cpu("input")
            weight_tf_form = self.get_cpu("weight")
            weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
            self.PlainFunc.weight.data.copy_(weight_pytorch_form)
            if self.bias:
                bias_data = self.get_cpu("bias")
                self.PlainFunc.bias.data.copy_(bias_data)
        elif self.EnclaveMode in [ExecutionModeOptions.CPU, ExecutionModeOptions.GPU]:
            self.make_sure_cpu_is_latest("input")
            self.requires_grad_on_cpu("input")
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
            # torch.set_num_threads(1)
            self.PlainForwardResult = self.PlainFunc(self.get_cpu("input"))
            # torch.set_num_threads(4)

    def show_plain_error_forward(self):
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"), get_relative=True)
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")

    def print_connection_info(self):
        print(f"{self.LayerName:20} shape{self.pytorch_x_shape}{' ':20} mode{self.EnclaveMode}{' ':20} input {self.PrevLayer.LayerName:20} output {self.NextLayer.LayerName:20}")

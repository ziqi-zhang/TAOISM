from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.enclave_interfaces import GlobalTensor as gt
from python.utils.basic_utils import ExecutionModeOptions

import torch
from pdb import set_trace as st

class SecretWeightedAddLayer(SecretNonlinearLayer):
    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False, num_layers=2,
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.Shapefortranspose = None
        self.PrevLayer = []
        self.num_layers = num_layers
        self.weights = None
        assert link_prev

    def register_main_prev_layer(self, layer):
        assert len(self.PrevLayer) == 0
        self.PrevLayer.append(layer)

    def register_prev_layer(self, layer):
        if layer not in self.PrevLayer:
            self.PrevLayer.append(layer)

    def init_shape(self):
        print(self.LayerName, self.num_layers, self.PrevLayer)
        assert len(self.PrevLayer) == self.num_layers
        if len(self.PrevLayer) > 1:
            for layer1, layer2 in zip(self.PrevLayer, self.PrevLayer[1:]):
                output_shape1 = layer1.get_output_shape()
                output_shape2 = layer2.get_output_shape()
                if not output_shape1 == output_shape2:
                    print(layer1.LayerName, output_shape1)
                    print(layer2.LayerName, output_shape2)
                    st()
                assert output_shape1 == output_shape2
        else:
            output_shape1 = self.PrevLayer[0].get_output_shape()
        self.InputShape = output_shape1
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape

    def set_weights(self, weights):
        # print(len(weights), self.num_layers)
        assert len(weights) == self.num_layers
        self.weights = weights

    def init(self, start_enclave=True):
        TensorLoader.init(self, start_enclave)

    def get_output_shape(self):
        return self.OutputShape

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        NeededTensorNames = [("output", self.OutputShape, None)]
        for idx, layer in enumerate(self.PrevLayer):
            if idx == 0:
                input_name = "input"
            else:
                input_name = f"input{idx}"
            NeededTensorNames.append(
                (input_name, self.InputShape, None)
            )
        self.tensor_name_list = NeededTensorNames

    def link_tensors(self):
        if self.link_prev and self.PrevLayer is not None:
            for idx, layer in enumerate(self.PrevLayer):
                if idx == 0:
                    input_name = "input"
                else:
                    input_name = f"input{idx}"
                gt.link_tags(self.get_tag(input_name, remap=False), layer.get_tag("output", remap=False))
        if self.link_next and self.NextLayer is not None:
            gt.link_tags(self.get_tag("output", remap=False), self.NextLayer.get_tag("input", remap=False))


    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            inputs = []
            for idx, layer in enumerate(self.PrevLayer):
                if idx == 0:
                    input_name = "input"
                else:
                    input_name = f"input{idx}"
                if layer.EnclaveMode is ExecutionModeOptions.Enclave:
                    self.transfer_enclave_to_cpu(input_name)
                if layer.EnclaveMode is ExecutionModeOptions.GPU:
                    self.transfer_gpu_to_cpu(input_name)
                inputs.append(self.get_cpu(input_name))
            # if len(self.PrevLayer) > 1:
            #     print(self.PrevLayer[0].LayerName, "Input1: ", inputs[0][0,0,0,:10])
            #     print(self.PrevLayer[1].LayerName, "Input2: ", inputs[1][0,0,0,:10])
            # if self.LayerName == "Layer3.0.weighted_add":
            #     for idx, layer in enumerate(self.PrevLayer):
            #         print(f"PrevLayer {idx}: ")
            #         print(inputs[idx][0,0,0,:10])

            input_sum = 0
            # print(self.LayerName, self.weights)
            for weight, input in zip(self.weights, inputs):
                input_sum += weight * input
            self.set_cpu("output", input_sum)
            self.transfer_from_cpu("output")
            # if self.is_enclave_mode:
            #     self.transfer_cpu_to_enclave("output")

    def backward(self):
        return super().backward()

    def print_connection_info(self):
        input_names = f"{self.PrevLayer[0].LayerName}"
        for layer in self.PrevLayer[1:]:
            input_names += f", {layer.LayerName}"
        print(f"{self.LayerName:20} shape{self.InputShape}{' ':30} input {input_names} output {self.NextLayer.LayerName:20}")


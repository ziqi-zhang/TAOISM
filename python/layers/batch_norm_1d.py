import torch

from python.sgx_net import LearnableParamTuple
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer


class SecretBatchNorm1dLayer(SecretBatchNorm2dLayer):
    def __init__(self, sid, layer_name, is_enclave_mode=False):
        super().__init__(sid, layer_name, is_enclave_mode)
        self.ForwardFuncName = "BatchNorm1d"
        self.BackwardFuncName = "DerBatchNorm1d"
        self.ForwardFunc = torch.nn.BatchNorm1d
        self.PlainFunc = torch.nn.BatchNorm1d

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = self.InputShape
        self.BatchSize, self.NumChannel = self.InputShape
        self.WeightShape = [self.NumChannel]
        self.LearnableParamsList = [
            LearnableParamTuple(dw_name="DerWeight", w_name="weight", shape=self.WeightShape),
            LearnableParamTuple(dw_name="DerBias", w_name="bias", shape=self.WeightShape),
        ]

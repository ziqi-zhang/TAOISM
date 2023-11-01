from python.layers.base import SecretLayerBase


class SecretNonlinearLayer(SecretLayerBase):
    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)

        self.InputShape = None
        self.OutputShape = None
        self.HandleShape = None
        self.NameTagRemap = {}

    def init_shape(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

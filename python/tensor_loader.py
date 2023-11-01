from types import MethodType
from pdb import set_trace as st
import torch

from python.utils.basic_utils import str_hash
from python.enclave_interfaces import EnclaveInterface, GlobalTensor
from python.global_config import SecretConfig


class TensorLoader(EnclaveInterface):
    def __init__(self):
        super().__init__()
        self.sid = -1
        self.tensor_name_list = []
        self.encryption_tensor_name_list = {}
        self.RandomVarName = None
        self.ShareVarName = None
        self.ShareTuple = None

    def init(self, start_enclave=True):
        if start_enclave:
            print("Initializing sid: %d" % self.sid)
            self.init_enclave()
        
        self.generate_tensor_name_list()
        # if hasattr(self, "LayerName") and self.LayerName == "Layer1.0.main.relu2":
        #     st()

        self.init_enclave_tensors()
        self.init_cpu_tensor()
        self.init_encryption_tensor()
        

    def generate_tensor_name_list(self, force=False):
        return

    def link_tensors(self):
        pass

    def init_enclave_tensors(self):
        self.generate_tensor_name_list()
        for TensorName, shape, SeedList in self.tensor_name_list:
            if shape is None:
                raise ValueError("The shape is None. Please setup the shape before init_enclave_tensor")
            # print(f"TensorLoader init {TensorName}, {shape}")
            self.init_enclave_tensor(TensorName, shape)
            if SeedList is None:
                continue
            for seed in SeedList:
                self.set_seed(TensorName, seed)

    def set_cpu(self, name, t):
        # print("---", name, self.get_tag(name))
        GlobalTensor.set_cpu(self.get_tag(name), t)

    def set_gpu(self, name, t):
        GlobalTensor.set_gpu(self.get_tag(name), t)

    def set_encryption(self, name, t):
        GlobalTensor.set_encryption(self.get_tag(name), t)

    def get_cpu(self, name):
        return GlobalTensor.get_cpu(self.get_tag(name))

    def get_gpu(self, name):
        return GlobalTensor.get_gpu(self.get_tag(name))

    def get_encryption(self, name):
        return GlobalTensor.get_encryption(self.get_tag(name))

    def generate_cpu_tensor(self, name, shape):
        self.set_cpu(name, torch.zeros(shape).type(SecretConfig.dtypeForCpuOp))
        # self.CpuTensors[name] = torch.zeros(shape).type(SecretConfig.dtypeForCpuOp)

    def transfer_cpu_to_gpu(self, name):
        self.set_gpu(name, self.get_cpu(name).cuda(non_blocking=True).type(SecretConfig.dtypeForCudaMm))
        # self.GpuTensors[name] = self.CpuTensors[name].cuda(non_blocking=True).type(SecretConfig.dtypeForCudaMm)

    def transfer_gpu_to_cpu(self, name):
        cpu_tensor = self.get_cpu(name)
        gpu_tensor = self.get_gpu(name)
        cpu_tensor.copy_(gpu_tensor.type(SecretConfig.dtypeForCpuOp))

    def transfer_enclave_to_cpu(self, name):
        self.from_enclave(name, self.get_cpu(name))

    def transfer_cpu_to_enclave(self, name):
        self.set_tensor(name, self.get_cpu(name))

    def init_cpu_tensor(self):
        self.generate_tensor_name_list()

        for TensorName, shape, _ in self.tensor_name_list:
            self.generate_cpu_tensor(TensorName, shape)

    def init_encryption_tensor(self):
        self.generate_tensor_name_list()

        for name, shape in self.encryption_tensor_name_list:
            GlobalTensor.init_encrypted_tensor(self.get_tag(name), shape)
            # self.EncrtyptedTensors[name] = self.CreateEncryptTorch(shape)

    def set_tensor_cpu_enclave(self, name, tensor):
        # GlobalTensor.SetNamedTensor(self.GetTag(tag), tensor)
        self.set_cpu(name, tensor)
        self.set_tensor(name, tensor)
        # print("Set cpu enclave: ", tensor[0,:10])

    def set_tensor_cpu_gpu_enclave(self, name, tensor):
        # GlobalTensor.SetNamedTensor(self.GetTag(tag), tensor)
        self.set_cpu(name, tensor)
        self.set_tensor(name, tensor)
        self.set_gpu(name, tensor)
        # print("Set cpu enclave: ", tensor[0,:10])

    def from_enclave(self, name, tensor):
        self.get_tensor(name, tensor)

    # def generate_enclave_tensor(self, name):
    #     if name in self.RandomVarName:
    #         return self.async_get_random(name, self.get_cpu(name))
    #     elif name in self.ShareVarName:
    #         original, seed = self.ShareTuple[name]
    #         return self.async_get_share(original, self.get_cpu(name), seed)
    #     else:
    #         raise Exception("Doesnt how to generate this tensor")


def tensor_loader_factory(sid, tensor_loader_name):
    GlobalTensor.init()
    tensor_loader = TensorLoader()
    tensor_loader.Name = tensor_loader_name
    tensor_loader.LayerId = str_hash(tensor_loader.Name)
    tensor_loader.Sid = sid
    tensor_loader.set_eid(GlobalTensor.get_eid())

    def name_modifier(self, name):
        return self.Name + "--" + str(name)

    tensor_loader.name_modifier = MethodType(name_modifier, tensor_loader)
    return tensor_loader



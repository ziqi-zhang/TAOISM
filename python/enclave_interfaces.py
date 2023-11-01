from collections import defaultdict
from ctypes import c_uint8, c_ulong, c_ulonglong, c_uint64, c_float, c_ubyte, cdll, POINTER, c_int, c_uint, c_uint32, c_bool

import numpy as np
import torch
from pdb import set_trace as st

from python.utils.basic_utils import str_hash
from python.utils.torch_utils import get_prod


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class GlobalState(object):
    __instance = None

    @staticmethod
    def get_instance():
        if GlobalState.__instance is None:
            GlobalState()
        return GlobalState.__instance

    def __init__(self):
        GlobalState.__instance = self
        self.IterEpoch = 0

    @staticmethod
    def set_iter_epoch(iter_epoch):
        GlobalState.get_instance().IterEpoch = iter_epoch

    @staticmethod
    def get_iter_epoch():
        return GlobalState.get_instance().IterEpoch

    @staticmethod
    def get_train_state():
        if GlobalState.get_instance().get_iter_epoch() < 5:
            return 0
        else:
            return 1


class GlobalParam(object):
    @staticmethod
    def get_for_error():
        if GlobalState.get_train_state() == 0:
            return 1 << 13, 1 << 5
            # return 1 << 13, 1 << 7
        else:
            return 1 << 17, 1 << 7

    @staticmethod
    def get_for_active():
        if GlobalState.get_train_state() == 0:
            return 2 ** (-2), 1 << 5
            # return 2 ** (-2), 1 << 7
        else:
            return 2 ** (-3), 1 << 7

    @staticmethod
    def get_for_grad():
        if GlobalState.get_train_state() == 0:
            return 1 << 10, 1 << 9
        else:
            return 1 << 10, 1 << 9

    @staticmethod
    def get_for_weight():
        if GlobalState.get_train_state() == 0:
            return 1 << 4, 1 << 5
        else:
            return 1 << 4, 1 << 5

    @staticmethod
    def get_for_enum(x):
        if x == SecretEnum.Error:
            return GlobalParam.get_for_error()
        if x == SecretEnum.Activate:
            return GlobalParam.get_for_active()
        if x == SecretEnum.Grad:
            return GlobalParam.get_for_grad()
        if x == SecretEnum.Weight:
            return GlobalParam.get_for_weight()
        if x == SecretEnum.Identical:
            return (1, 1)
        raise ValueError("Unknown Enum For GlobalParams")


class SecretEnum(object):
    Error = 0
    Weight = 1
    Activate = 2
    Grad = 4
    Identical = 4

    ReLU = 0
    DerReLU = 1
    BatchNorm2d = 2
    DerBatchNorm2d = 3


class GlobalTensor(object):
    cpu_tensor = {}
    gpu_tensors = {}
    encrypted_tensors = {}
    LinkedTags = {}
    InverseLinkedTags = {}
    IsInitEnclaveTensor = {}
    EnclaveInterface = None
    eid = None
    is_init_global_tensor = False

    @staticmethod
    def init():
        if GlobalTensor.is_init_global_tensor:
            return
        GlobalTensor.EnclaveInterface = EnclaveInterface()
        GlobalTensor.EnclaveInterface.init_enclave()
        GlobalTensor.is_init_global_tensor = True

    @staticmethod
    def destroy():
        GlobalTensor.EnclaveInterface.destroy_enclave()

        GlobalTensor.cpu_tensor = {}
        GlobalTensor.gpu_tensors = {}
        GlobalTensor.encrypted_tensors = {}
        GlobalTensor.LinkedTags = {}
        GlobalTensor.InverseLinkedTags = {}
        GlobalTensor.IsInitEnclaveTensor = {}
        GlobalTensor.EnclaveInterface = None
        GlobalTensor.eid = None
        GlobalTensor.is_init_global_tensor = False


    @staticmethod
    def get_eid():
        return GlobalTensor.EnclaveInterface.get_eid()

    @staticmethod
    def link_tags(tag1, tag2):
        if tag1 == tag2:
            return

        friends = []

        def add_friends(tag):
            nonlocal friends
            if tag in GlobalTensor.LinkedTags:
                its_leader_tag = GlobalTensor.LinkedTags[tag]
                if its_leader_tag in GlobalTensor.InverseLinkedTags:
                    friends += GlobalTensor.InverseLinkedTags.pop(its_leader_tag)
            else:
                friends += [tag]

        add_friends(tag1)
        add_friends(tag2)
        leader_tag = min(friends)

        GlobalTensor.InverseLinkedTags[leader_tag] = friends
        for t in friends:
            if t in GlobalTensor.IsInitEnclaveTensor:
                raise ValueError("Tags must linked before tensor initialization")
            GlobalTensor.LinkedTags[t] = leader_tag

    @staticmethod
    def get_remapped_tags(tag):
        return GlobalTensor.LinkedTags[tag] if tag in GlobalTensor.LinkedTags else tag

    @staticmethod
    def set_cpu(tag, tensor):
        GlobalTensor.cpu_tensor[tag] = tensor.to(torch.device("cpu"))

    @staticmethod
    def set_gpu(tag, tensor):
        GlobalTensor.gpu_tensors[tag] = tensor

    @staticmethod
    def set_encrypted(tag, tensor):
        GlobalTensor.encrypted_tensors[tag] = tensor

    @staticmethod
    def get_cpu(tag):
        return GlobalTensor.cpu_tensor[tag]

    @staticmethod
    def get_gpu(tag):
        return GlobalTensor.gpu_tensors[tag]

    @staticmethod
    def get_encryption(tag):
        return GlobalTensor.encrypted_tensors[tag]

    @staticmethod
    def init_enclave_tensor(tag, size):
        size = list(size)
        if len(size) < 4:
            size = [1] * (4 - len(size)) + size
        remapped_tag = GlobalTensor.get_remapped_tags(tag)
        if remapped_tag in GlobalTensor.IsInitEnclaveTensor:
            return
        else:
            GlobalTensor.IsInitEnclaveTensor[remapped_tag] = True
        eid = GlobalTensor.get_eid()
        GlobalTensor.EnclaveInterface.lib.InitTensor(eid, remapped_tag, size[0], size[1], size[2], size[3])

    @staticmethod
    def init_encrypted_tensor(tag, shape):
        GlobalTensor.encrypted_tensors[GlobalTensor.get_remapped_tags(tag)] = \
            GlobalTensor.EnclaveInterface.create_encrypt_torch(shape)


def get_float_ptr(x):
    return x.detach().numpy().ctypes.data_as(EnclaveInterface.ArrToEnclaveT)


def get_encryption_ptr(x):
    return x.numpy().ctypes.data_as(EnclaveInterface.ArrEncryptedT)


class EnclaveInterface(object):
    EidT = c_uint64
    IdT = c_ulonglong
    TaskIdT = c_uint64
    ArrToEnclaveT = POINTER(c_float)
    ArrEncryptedT = POINTER(c_ubyte)
    SGXLIB = "App/bin/enclave_bridge.so"
    DNNLIB = "lib/sgxdnn.so"

    tag_to_names = {}

    def __init__(self):
        import os
        cwd = os.getcwd()

        self.eid = None
        self.batchnormID = None
        self.lib = cdll.LoadLibrary(self.SGXLIB)

        self.lib.initialize_enclave.restype = self.EidT
        self.lib.destroy_enclave.argtypes = [self.EidT]
        self.lib.CalcEncNeededInByte.argtypes = [c_uint]
        self.lib.CalcEncNeededInByte.restype = c_int
        self.lib.AesEncryptTensor.argtypes = [self.ArrToEnclaveT, c_uint, self.ArrEncryptedT]
        self.lib.AesDecryptTensor.argtypes = [self.ArrEncryptedT, c_uint, self.ArrToEnclaveT]
        self.lib.InitTensor.argtypes = [self.EidT, self.IdT] + [c_int] * 4
        self.lib.SetTen.argtypes = [self.EidT, self.IdT, self.ArrToEnclaveT]
        self.lib.GetTen.argtypes = [self.EidT, self.IdT, self.ArrToEnclaveT]
        self.lib.SetSeed.argtypes = [self.EidT, self.IdT, c_ulonglong]
        self.lib.GetRandom.argtypes = [self.EidT, self.IdT, self.ArrToEnclaveT, c_ulonglong]
        self.lib.AsyncGetRandom.argtypes = self.lib.GetRandom.argtypes
        self.lib.AsyncGetRandom.restype = self.TaskIdT
        self.lib.SgdUpdate.argtypes = [self.EidT] + [self.IdT] * 3 + [c_float] * 4 + [c_bool] * 2
        self.lib.AsyncSgdUpdate.argtypes = [self.EidT] + [self.IdT] * 3 + [c_float] * 4 + [c_bool] * 2
        self.lib.AsyncSgdUpdate.restype = self.TaskIdT
        self.lib.GetTaskStatus.argtypes = [self.TaskIdT]
        self.lib.GetTaskStatus.restype = c_int
        self.lib.AddFromCpu.argtypes = [self.EidT, self.ArrToEnclaveT, self.IdT]
        self.lib.AsyncTask.argtypes = [self.EidT,
                                       self.IdT, self.ArrToEnclaveT, c_uint64, c_uint64,
                                       self.IdT, self.ArrToEnclaveT, c_uint64, c_uint64,
                                       self.IdT, self.ArrToEnclaveT, c_uint64, c_uint64,
                                       self.IdT, self.ArrToEnclaveT, c_uint64, c_uint64]
        self.lib.ReLUfunction.argtypes = [self.EidT, self.IdT, self.IdT, c_uint64]
        self.lib.QuantReLUfunction.argtypes = [self.EidT, self.IdT, self.IdT, c_uint64, c_float, c_float, c_uint8]
        self.lib.ReLUbackward.argtypes = [self.EidT, self.IdT, self.IdT, self.IdT, c_uint64]
        self.lib.InitMaxpool.argtypes = [self.EidT, self.IdT, self.IdT, self.IdT]
        self.lib.Maxpoolfunction.argtypes = [self.EidT, self.IdT, self.IdT, self.IdT] + [c_uint32] * 12
        self.lib.Maxpoolbackwardfunction.argtypes = [self.EidT, self.IdT, self.IdT, self.IdT] + [c_uint32] * 10
        self.lib.InitBatchnorm.argtypes = [self.EidT] + [self.IdT] * 10 + [c_uint32] * 4 + [c_int] * 2 + [c_float] * 2
        self.lib.BatchnormForward.argtypes = [self.EidT, self.IdT, c_int]
        self.lib.BatchnormBackward.argtypes = [self.EidT, self.IdT]
        self.lib.StochasticQuantize.argtypes = [self.EidT, c_uint64, c_uint64, c_uint64]
        self.lib.AsyncStochasticQuantize.argtypes = self.lib.StochasticQuantize.argtypes
        self.lib.AsyncStochasticQuantize.restype = c_int

        self.lib.InitSGXLinear.argtypes = [self.EidT] + [self.IdT] * 5 + [c_uint32] * 3
        self.lib.SGXLinearForward.argtypes = [self.EidT, self.IdT]
        self.lib.InitSGXConv.argtypes = [self.EidT] + [self.IdT] * 5 + [c_uint32] * 10
        self.lib.SGXConvForward.argtypes = [self.EidT, self.IdT]

        self.deployed_name_seed = defaultdict(list)

    def init_enclave(self):
        self.eid = self.lib.initialize_enclave()
        # print(f"Eid {self.eid}")

    def destroy_enclave(self):
        self.lib.destroy_enclave(self.get_eid())

    def get_eid(self):
        if self.eid is None:
            raise ValueError("Eid is None")
        return self.eid

    def set_eid(self, eid):
        self.eid = eid

    def name_modifier(self, name):
        return name

    def get_tag(self, name, remap=True):
        tag = str_hash(self.name_modifier(name))
        if tag not in self.tag_to_names:
            self.tag_to_names[tag] = self.name_modifier(name)
        if remap:
            return GlobalTensor.get_remapped_tags(tag)
        else:
            return tag

    def calc_enc_needed_bytes(self, NumBtye):
        return self.lib.CalcEncNeededInByte(NumBtye)

    def create_encrypt_torch(self, shape):
        NeededNumBtype = self.calc_enc_needed_bytes(get_prod(shape))
        return torch.zeros(NeededNumBtype).type(torch.uint8)

    def aes_encrypt(self, plain_tensor, enc_tensor):
        self.lib.AesEncryptTensor(get_float_ptr(plain_tensor), get_prod(plain_tensor.size()),
                                  get_encryption_ptr(enc_tensor))

    def aes_decrypt(self, enc_tensor, plain_tensor):
        self.lib.AesDecryptTensor(get_encryption_ptr(enc_tensor), get_prod(plain_tensor.size()),
                                  get_float_ptr(plain_tensor))

    def init_tensor_unsafe(self, tag, size):
        GlobalTensor.init_enclave_tensor(tag, size)

    def init_enclave_tensor(self, name, size):
        self.init_tensor_unsafe(self.get_tag(name), size)
        # GlobalTensor.InitEnclaveTensor(self.GetTag(name), size)
        # self.lib.InitTensor(self.GetEid(), self.GetTag(name), size[0], size[1], size[2], size[3])

    def set_tensor_unsafe(self, tag, tensor):
        # if len(tensor.shape)>1:
        #     print(f"Set tag {tag}, ", tensor[0,:10])
        self.lib.SetTen(self.get_eid(), tag, get_float_ptr(tensor))

    def set_tensor(self, name, tensor):
        # if len(tensor.shape)>1:
        #     print(f"Set {name}, ", tensor[0,:10])
        self.set_tensor_unsafe(self.get_tag(name), tensor)
        # self.lib.SetTen(self.GetEid(), self.GetTag(name), GetFloatPtr(tensor))

    def set_enclave_tensor(self, name, tensor):
        self.set_tensor(name, tensor)

    def get_tensor(self, name, tensor):
        self.lib.GetTen(self.get_eid(), self.get_tag(name), get_float_ptr(tensor))

    def get_enclave_tensor(self, name, tensor):
        self.get_tensor(name, tensor)

    def set_seed(self, name, seed):
        name_tag = self.get_tag(name)
        seed_tag = self.get_tag(seed, remap=False)
        self.deployed_name_seed[name_tag].append(seed_tag)
        self.lib.SetSeed(self.get_eid(), name_tag, seed_tag)

    def get_validated_name_seed_tag(self, name, seed):
        name_tag = self.get_tag(name)
        seed_tag = self.get_tag(seed, remap=False)
        if seed_tag not in self.deployed_name_seed[name_tag]:
            raise ValueError(f"Not existing name seed tag pair: name_tag: {name_tag}, seed_tag: {seed_tag}")
        return name_tag, seed_tag

    def get_random(self, name, tensor):
        name_tag, seed_tag = self.get_validated_name_seed_tag(name, name)
        self.lib.GetRandom(self.get_eid(), name_tag, get_float_ptr(tensor), seed_tag)

    def enclave_add_from_cpu(self, src, dst_name):
        if isinstance(src, str):
            src_ptr = get_float_ptr(self.get_cpu(src))
        elif isinstance(src, torch.tensor):
            src_ptr = get_float_ptr(src)
        else:
            raise ValueError("src has to be str or troch.tensor")
        self.lib.AddFromCpu(self.get_eid(), src_ptr, self.get_tag(dst_name))

    def get_task_status(self, task_id):
        # return True if the task is finished
        # return False otherwise
        res = self.lib.GetTaskStatus(task_id)
        return res == 1

    def wait_tasks(self, task_ids):
        while len(task_ids) > 0:
            to_be_removed = []
            for task_id in task_ids:
                status = self.get_task_status(task_id)
                if status:
                    to_be_removed.append(task_id)
            for task_id in to_be_removed:
                task_ids.remove(task_id)

    def async_get_random(self, name, tensor, seed=""):
        if seed == "":
            seed = name
        return self.lib.AsyncGetRandom(self.get_eid(), self.get_tag(name),
                                       get_float_ptr(tensor), self.get_tag(seed, remap=False))

    def async_task(self, name1, arr1, seed1, name2, arr2, seed2, name3, arr3, seed3, name4, arr4, seed4):
        def get_size(t):
            return np.prod(t.size())

        print(get_size(arr1))
        self.lib.AsyncTask(self.get_eid(),
                           self.get_tag(name1), get_float_ptr(arr1), get_size(arr1), self.get_tag(seed1, remap=False),
                           self.get_tag(name2), get_float_ptr(arr2), get_size(arr2), self.get_tag(seed2, remap=False),
                           self.get_tag(name3), get_float_ptr(arr3), get_size(arr3), self.get_tag(seed3, remap=False),
                           self.get_tag(name4), get_float_ptr(arr4), get_size(arr4), self.get_tag(seed4, remap=False),
                           )

    def relunew(self, namein, nameout, sizelist):
        self.lib.ReLUfunction(self.get_eid(), self.get_tag(namein), self.get_tag(nameout),
                              np.prod(sizelist))
    
    def quant_relunew(self, namein, nameout, sizelist, scale, v_min, zero):
        self.lib.QuantReLUfunction(self.get_eid(), self.get_tag(namein), self.get_tag(nameout),
                              np.prod(sizelist), scale, v_min, zero)

    def relubackward(self, nameout, namedout, namedin, sizelist):
        self.lib.ReLUbackward(self.get_eid(), self.get_tag(nameout), self.get_tag(namedout), self.get_tag(namedin),
                              np.prod(sizelist))

    def roundup8(self, number):
        return ((number + 7) & -(8))

    def maxpoolinit(self, layer_name, name_in_trans, name_out_trans):
        return self.lib.InitMaxpool(self.get_eid(), self.get_tag(layer_name), self.get_tag(name_in_trans), self.get_tag(name_out_trans))

    def maxpoolnew(self, layer_name, namein, nameout, sizelist, outputheight, outputwidth, filterh, filterw, rowstride, colstride,
                   rowpad, colpad):
        if rowstride is None and colstride is None and rowpad is None and colpad is None:
            self.lib.Maxpoolfunction(self.get_eid(), self.get_tag(layer_name), self.get_tag(namein), self.get_tag(nameout), sizelist[0],
                                     sizelist[1], sizelist[2], sizelist[3], outputheight, outputwidth, filterh, filterw,
                                     filterh, filterw, 0, 0)
        elif rowstride is None and colstride is None and rowpad is not None and colpad is not None:
            self.lib.Maxpoolfunction(self.get_eid(), self.get_tag(layer_name), self.get_tag(namein), self.get_tag(nameout), sizelist[0],
                                     sizelist[1], sizelist[2], sizelist[3], outputheight, outputwidth, filterh, filterw,
                                     filterh, filterw, rowpad, colpad)
        elif rowstride is not None and colstride is not None and rowpad is None and colpad is None:
            self.lib.Maxpoolfunction(self.get_eid(), self.get_tag(layer_name), self.get_tag(namein), self.get_tag(nameout), sizelist[0],
                                     sizelist[1], sizelist[2], sizelist[3], outputheight, outputwidth, filterh, filterw,
                                     rowstride, colstride, 0, 0)
        else:
            self.lib.Maxpoolfunction(self.get_eid(), self.get_tag(layer_name), self.get_tag(namein), self.get_tag(nameout), sizelist[0],
                                     sizelist[1], sizelist[2], sizelist[3], outputheight, outputwidth, filterh, filterw,
                                     rowstride, colstride, rowpad, colpad)

    def maxpoolback(self, layer_name, namedout, namedin, sizelist, outputheight, outputwidth, filterh, filterw, rowstride,
                    colstride, rowpad, colpad):
        if rowstride is None and colstride is None and rowpad is None and colpad is None:
            self.lib.Maxpoolbackwardfunction(self.get_eid(), self.get_tag(layer_name), self.get_tag(namedout), self.get_tag(namedin), sizelist[0],
                                             sizelist[1], sizelist[2], sizelist[3], outputheight, outputwidth, filterh,
                                             filterw, filterh, filterw)
        elif rowstride is None and colstride is None and rowpad is not None and colpad is not None:
            self.lib.Maxpoolbackwardfunction(self.get_eid(), self.get_tag(layer_name), self.get_tag(namedout), self.get_tag(namedin), sizelist[0],
                                             sizelist[1], sizelist[2] + rowpad, sizelist[3] + colpad, outputheight,
                                             outputwidth, filterh, filterw, filterh, filterw)
        elif rowstride is not None and colstride is not None and rowpad is None and colpad is None:
            self.lib.Maxpoolbackwardfunction(self.get_eid(), self.get_tag(layer_name), self.get_tag(namedout), self.get_tag(namedin), sizelist[0],
                                             sizelist[1], sizelist[2], sizelist[3], outputheight, outputwidth, filterh,
                                             filterw, rowstride, colstride)
        else:
            self.lib.Maxpoolbackwardfunction(self.get_eid(), self.get_tag(layer_name), self.get_tag(namedout), self.get_tag(namedin), sizelist[0],
                                             sizelist[1], sizelist[2] + rowpad, sizelist[3] + colpad, outputheight,
                                             outputwidth, filterh, filterw, rowstride, colstride)

    def batchnorm_init(self, layer_name,
                       input_name, output_name, gamma_name, beta_name,
                    #    der_input_name, der_output_name, der_gamma_name, der_beta_name,
                       run_mean_name, run_var_name, cur_mean_name, cur_var_name,
                       mu_name,
                       batch_size, num_channel, img_h, img_w,
                       is_affine, is_cumulative, momentum, epsilon):
        
        self.lib.InitBatchnorm(
            self.get_eid(), 
            self.get_tag(layer_name), self.get_tag(input_name), self.get_tag(output_name), self.get_tag(gamma_name), self.get_tag(beta_name),
            self.get_tag(run_mean_name), self.get_tag(run_var_name), self.get_tag(cur_mean_name), self.get_tag(cur_var_name),
            self.get_tag(mu_name),
            batch_size, num_channel, img_h, img_w,
            is_affine, is_cumulative, momentum, epsilon)

    def batchnorm_forward(self, layer_name, training):
        self.lib.BatchnormForward(self.get_eid(), self.get_tag(layer_name), int(training))

    def batchnorm_backward(self, layer_name):
        self.lib.BatchnormBackward(self.get_eid(), self.get_tag(layer_name))

    def async_masking_c01(self, store_name, main_seed, seed0, seed1, dst_tensor):
        return self.lib.AsyncMaskingC01(self.get_eid(), self.get_tag(store_name),
                                        self.get_tag(main_seed, remap=False),
                                        self.get_tag(seed0, remap=False),
                                        self.get_tag(seed1, remap=False),
                                        get_float_ptr(dst_tensor))

    def sgd_update(self, param_name=required, grad_name=required, momentum_name=None,
                   lr=None, momentum=0, weight_decay=0, dampening=0, nesterov=False,
                   first_momentum=False, is_async=True):
        if param_name is required:
            raise ValueError("param_name is required")
        if grad_name is required:
            raise ValueError("grad_name is required")
        if not (0 <= momentum <= 1):
            raise ValueError("momentum has to in [0, 1]")
        if 0 < momentum < 1 and momentum_name is None:
            raise ValueError("momentum name cannot be None")
        if not (0 <= weight_decay <= 1):
            raise ValueError("momentum has to in [0, 1]")
        if lr is None or lr < 0:
            raise ValueError("learning rate has to be positive")
        if momentum_name is None:
            raise NotImplementedError
        if dampening != 0:
            raise NotImplementedError
        if nesterov:
            raise NotImplementedError

        param_tag = self.get_tag(param_name)
        grad_tag = self.get_tag(grad_name)
        momentum_tag = self.get_tag(momentum_name) if momentum_name is not None else 0

        func = self.lib.AsyncSgdUpdate if is_async else self.lib.SgdUpdate

        return func(self.get_eid(), param_tag, grad_tag, momentum_tag,
                    lr, momentum, weight_decay, dampening, nesterov, first_momentum)

    def quantize(self, src_name, dst_name, q_tag, is_async=True):
        if is_async:
            func = self.lib.AsyncStochasticQuantize
        else:
            func = self.lib.StochasticQuantize
        return func(self.get_eid(), self.get_tag(src_name), self.get_tag(dst_name), self.get_tag(q_tag, remap=False))

    def sgx_linear_init(
        self, layer_name, 
        input_name, output_name, weight_name, bias_name,
        # der_input_name, der_output_name, der_weight_name, der_bias_name,
        batch_size, input_size, output_size,
    ):
        self.lib.InitSGXLinear(
            self.get_eid(), self.get_tag(layer_name),
            self.get_tag(input_name), self.get_tag(output_name), self.get_tag(weight_name), self.get_tag(bias_name), 
            # self.get_tag(der_input_name), self.get_tag(der_output_name), self.get_tag(der_weight_name), self.get_tag(der_bias_name), 
            batch_size, input_size, output_size,
        )

    def sgx_linear_forward(self, layer_name):
        self.lib.SGXLinearForward(
            self.get_eid(), self.get_tag(layer_name),
        )

    def sgx_conv_init(
        self, layer_name, 
        input_name, output_name, weight_name, bias_name,
        # der_input_name, der_output_name, der_weight_name, der_bias_name,
        batch_size, input_h, input_w, input_c, output_h, output_w, output_c, 
        kernel, padding, stride,
    ):
        self.lib.InitSGXConv(
            self.get_eid(), self.get_tag(layer_name),
            self.get_tag(input_name), self.get_tag(output_name), self.get_tag(weight_name), self.get_tag(bias_name),
            # self.get_tag(der_input_name), self.get_tag(der_output_name), self.get_tag(der_weight_name), self.get_tag(der_bias_name),
            batch_size, input_h, input_w, input_c, output_h, output_w, output_c,
            kernel, padding, stride,
        )

    def sgx_conv_forward(self, layer_name):
        self.lib.SGXConvForward(
            self.get_eid(), self.get_tag(layer_name),
        )

    @staticmethod
    def print_tensor_link_relation():
        print("="*30, "Print Tensor Link Relation", "="*30)
        for tag, friends in GlobalTensor.InverseLinkedTags.items():
            if "Der" in EnclaveInterface.tag_to_names[tag]:
                continue
            friend_info = "["
            for f in friends:
                friend_info += EnclaveInterface.tag_to_names[f]
                friend_info += ", "
            friend_info += "]"
            print(EnclaveInterface.tag_to_names[tag], ": ", friend_info)
        print("="*60)

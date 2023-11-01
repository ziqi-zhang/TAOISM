#!usr/bin/env python
from __future__ import print_function
from multiprocessing.pool import ThreadPool

import numpy as np
import torch

from python.utils.basic_utils import str_hash
from python.enclave_interfaces import GlobalTensor, SecretEnum
from python.quantize_net import swalp_quantize, NamedParam, dequantize_op
from python.utils.timer_utils import NamedTimer, NamedTimerInstance
from python.common_torch import SecretConfig, mod_on_cpu,  get_random_uniform, GlobalParam, \
    quantize, generate_unquantized_tensor, dequantize, mod_move_down
from python.sgx_net import TensorLoader, conv2d_weight_grad_op
from python.utils.torch_utils import compare_expected_actual

np.random.seed(123)
minibatch, inChan, outChan, imgHw, filHw = 64, 128, 128, 16, 3
minibatch, inChan, outChan, imgHw, filHw = 64, 64, 3, 32, 3
xShape = [minibatch, inChan, imgHw, imgHw]
wShape = [outChan, inChan, filHw, filHw]

# s consume the dummy self
ConvOp = lambda w, x: torch.conv2d(x, w, padding=1)
MatmOp = lambda w, x: torch.mm(w, x)
TargetOp = ConvOp

AQ = get_random_uniform(SecretConfig.PrimeLimit, wShape).type(SecretConfig.dtypeForCpuOp)
A0 = torch.zeros(AQ.size()).type(SecretConfig.dtypeForCpuOp)
A1 = torch.zeros(AQ.size()).type(SecretConfig.dtypeForCpuOp)
BQ = get_random_uniform(SecretConfig.PrimeLimit, xShape).type(SecretConfig.dtypeForCpuOp)
B0 = get_random_uniform(SecretConfig.PrimeLimit, xShape).type(SecretConfig.dtypeForCpuOp)
B1 = mod_on_cpu(BQ - B0)

idealC = mod_on_cpu(TargetOp(AQ.type(torch.double), BQ.type(torch.double))).type(SecretConfig.dtypeForCpuOp)
yShape = list(idealC.size()) 
C0 = get_random_uniform(SecretConfig.PrimeLimit, yShape).type(SecretConfig.dtypeForCpuOp)
C1 = get_random_uniform(SecretConfig.PrimeLimit, yShape).type(SecretConfig.dtypeForCpuOp)
Z  = mod_on_cpu(idealC - C0 - C1)


class EnclaveInterfaceTester(TensorLoader):
    def __init__(self):
        super().__init__()
        self.Name = "SingleLayer"
        self.LayerId = str_hash(self.Name)
        self.Sid = 0

    def name_modifier(self, name):
        return self.Name + "--" + str(name)

    def init_test(self):
        print()
        GlobalTensor.init()
        self.set_eid(GlobalTensor.get_eid())

    def test_tensor(self):
        print()
        print("minibatch, inChan, outChan, imgHw, filHw = %d, %d, %d, %d, %d"
              % (minibatch, inChan, outChan, imgHw, filHw))
        print("wShape", wShape)
        print("xShape", xShape)
        print("yShape", yShape)
        print()

        NamedTimer.start("InitTensor")
        self.init_enclave_tensor("BQ", BQ.size())
        NamedTimer.end("InitTensor")
        print()

        NamedTimer.start("Preprare Decrypt")
        C0Enc = self.create_encrypt_torch(C0.size())
        NamedTimer.end("Preprare Decrypt")
        C0Rec = torch.zeros(C0.size()).type(SecretConfig.dtypeForCpuOp)

        # AES Encryption and Decryption
        NamedTimer.start("AesEncrypt")
        self.aes_encrypt(C0, C0Enc)
        NamedTimer.end("AesEncrypt")

        NamedTimer.start("AesDecrypt")
        self.aes_decrypt(C0Enc, C0Rec)
        NamedTimer.end("AesDecrypt")

        print("Error of Enc and Dec:", compare_expected_actual(C0, C0Rec))
        print()

        self.init_enclave_tensor("AQ", AQ.size())
        self.init_enclave_tensor("A0", A0.size())
        self.init_enclave_tensor("A1", A1.size())
        self.init_enclave_tensor("BQ", BQ.size())
        self.init_enclave_tensor("B0", B0.size())
        self.init_enclave_tensor("B1", B1.size())

        NamedTimer.start("SetTen")
        self.set_tensor("AQ", AQ)
        NamedTimer.end("SetTen")

        # Test the Random Generation
        NamedTimer.start("GenRandomUniform: x (A)");
        get_random_uniform(SecretConfig.PrimeLimit, xShape).type(SecretConfig.dtypeForCpuOp)
        NamedTimer.end("GenRandomUniform: x (A)");

        npAQ = AQ.numpy()
        print("PrimeLimit:", SecretConfig.PrimeLimit)
        print("Python Rand max, min, avg:", np.max(npAQ), np.min(npAQ), np.average(npAQ))


    def test_plain_compute(self):
        print()
        with NamedTimerInstance("Time of Plain Computation"):
            TargetOp(AQ, BQ)

    def test_async_test(self):
        print()
        x_shape = [512, 64, 32, 32]
        w_shape = x_shape

        def init_set(n):
            self.init_enclave_tensor(n, w_shape)
            self.generate_cpu_tensor(n, w_shape)
            self.set_seed(n, n)

        init_set("AQ")
        init_set("BQ")
        init_set("CQ")
        init_set("DQ")
        name1, tensor1 = "AQ", self.get_cpu("AQ")
        name2, tensor2 = "BQ", self.get_cpu("BQ")
        name3, tensor3 = "CQ", self.get_cpu("CQ")
        name4, tensor4 = "DQ", self.get_cpu("DQ")
        with NamedTimerInstance("GetRandom * 4"):
            self.get_random("AQ", self.get_cpu("AQ"))
            self.get_random("BQ", self.get_cpu("BQ"))
            self.get_random("CQ", self.get_cpu("CQ"))
            self.get_random("DQ", self.get_cpu("DQ"))
        with NamedTimerInstance("AsyncTask"):
            self.async_task(name1, tensor1, name1,
                            name2, tensor2, name2,
                            name3, tensor3, name3,
                            name4, tensor4, name4)
        print(torch.sum(self.get_cpu("AQ")))
        print(torch.sum(self.get_cpu("BQ")))



Tester = EnclaveInterfaceTester()
Tester.init_test()

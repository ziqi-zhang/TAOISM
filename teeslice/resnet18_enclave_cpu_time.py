from .sgx_resnet import *
from .resnet import *

from python.enclave_interfaces import GlobalTensor
from python.utils.timer_utils import Timer

def end_to_end_time():
    GlobalTensor.init()

    batch_size = 16
    resnet_constructor = secret_resnet18(batch_size=batch_size, EnclaveMode=ExecutionModeOptions.Enclave)
    layers = resnet_constructor.sgx_layers

    pretrained = resnet18(pretrained=True)
    pretrained.eval()

    secret_nn = SecretNeuralNetwork(0, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    timer = Timer(name="EnclaveTimer")
    timer.start()
    for i in range(10):
        input_shape = resnet_constructor.input_shape
        input = torch.rand(input_shape)

        layers[0].set_input(input)

        secret_nn.forward()
    timer.end()
    enclave_time = timer.get_elapse_time()

    GlobalTensor.destroy()


    GlobalTensor.init()

    batch_size = 16
    resnet_constructor = secret_resnet18(batch_size=batch_size, EnclaveMode=ExecutionModeOptions.CPU)
    layers = resnet_constructor.sgx_layers

    pretrained = resnet18(pretrained=True)
    pretrained.eval()

    secret_nn = SecretNeuralNetwork(0, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    timer = Timer(name="CPUTimer")
    timer.start()
    for i in range(10):
        input_shape = resnet_constructor.input_shape
        input = torch.rand(input_shape) 

        layers[0].set_input(input)

        secret_nn.forward()
    timer.end()
    cpu_time = timer.get_elapse_time()

    GlobalTensor.destroy()

    print("EnclaveTime: ", enclave_time, "CPUTime: ", cpu_time)


    # GlobalTensor.init()
    # batch_size = 16
    # resnet_constructor = secret_resnet18(batch_size=batch_size, EnclaveMode=ExecutionModeOptions.GPU)
    # layers = resnet_constructor.sgx_layers

    # pretrained = resnet18(pretrained=True).cuda()
    # pretrained.eval()

    # timer = Timer(name="PytorchCPUTimer")
    # timer.start()
    # for i in range(10):
    #     input_shape = resnet_constructor.input_shape
    #     input = torch.rand(input_shape).cuda()
    #     pretrained(input)
    # timer.end()
    # pytorch_cpu_time = timer.get_elapse_time()
    # print(pytorch_cpu_time)

if __name__=="__main__":
    end_to_end_time()
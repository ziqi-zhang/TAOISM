import torch

layer_output = {}
layer_input = {}
layer_output_grad = {}
layer_input_grad = {}
layer_weight = {}

linear_layer_names = []
layer_names = []


def hooking_layer(name):
    def hook(model, input, output):
        layer_input[name] = input[0].detach()
        layer_output[name] = output.detach()

    return hook


def hooking_layer_backward(name):
    def hook(model, input, output):
        if input[0] is not None:
            layer_input_grad[name] = input[0].detach()
        if output[0] is not None:
            layer_output_grad[name] = output[0].detach()

    return hook


def register_layer(layer, name):
    layer.register_forward_hook(hooking_layer(name))
    layer.register_backward_hook(hooking_layer_backward(name))
    layer_names.append(name)


def register_weight_layer(layer, name):
    register_layer(layer, name)
    layer_weight[name] = layer.weight
    linear_layer_names.append(name)


def get_layer_input(name):
    return layer_input[name]


def get_layer_input_grad(name):
    return layer_input_grad[name]


def get_layer_output(name):
    return layer_output[name]


def get_layer_output_grad(name):
    return layer_output_grad[name]


def get_layer_weight(name):
    return layer_weight[name]


def get_layer_weight_grad(name):
    return layer_weight[name].grad.data


def store_layer(store_layer_name, store_name, prepath="./data/"):
    # store_layer_name = 'conv2'
    # store_name = f"{store_layer_name}_{epoch + 1}_{i + 1}"
    prefix = f"{prepath}/{store_name}"
    torch.save(get_layer_input(store_layer_name), prefix + "_input.pt")
    torch.save(get_layer_input_grad(store_layer_name), prefix + "_input_grad.pt")
    torch.save(get_layer_weight(store_layer_name), prefix + "_weight.pt")
    torch.save(get_layer_weight_grad(store_layer_name), prefix + "_weight_grad.pt")
    torch.save(get_layer_output(store_layer_name), prefix + "_output.pt")
    torch.save(get_layer_output_grad(store_layer_name), prefix + "_output_grad.pt")

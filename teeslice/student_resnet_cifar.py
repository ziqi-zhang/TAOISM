import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import collections
import torch.utils.model_zoo as model_zoo
from pdb import set_trace as st
import os

__all__ = [ 
    'resnet18',
    'resnet34',
    'resnet50',
]

backbone_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv(in_planes,  out_planes, kernel_size, stride=1, padding=0, use_bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)


# def conv1x1(in_planes,  out_planes, stride=1, use_bias=True):
#     return conv(in_planes, out_planes, 1, stride=stride, padding=0, use_bias=use_bias)

def conv1x1(in_planes,  out_planes, stride=1, use_bias=True):
    module = conv(in_planes, out_planes, 1, stride=stride, padding=0, use_bias=use_bias)
    torch.nn.init.kaiming_normal_(module.weight)
    return module


def conv3x3(in_planes,  out_planes, stride=1, use_bias=True):
    return conv(in_planes, out_planes, 3, stride=stride, padding=1, use_bias=use_bias)

# def conv3x3(in_planes,  out_planes, stride=1, use_bias=True):
#     module = conv(in_planes, out_planes, 3, stride=stride, padding=1, use_bias=use_bias)
#     torch.nn.init.kaiming_normal_(module.weight)
#     return module


class BasicResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Conv1
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride, use_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv2
        self.conv2 = conv3x3(out_channels, out_channels, use_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ConvRes
        self.needs_conv_res = stride>1 or in_channels!=out_channels
        if self.needs_conv_res:
            self.conv_res = conv1x1(in_channels, out_channels, stride=stride, use_bias=False)
            self.bn_res = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        res = self.bn_res(self.conv_res(x)) if self.needs_conv_res else x
        return y + res

    def num_params(self):
        n_params = 0.
        for p in self.parameters():
            n_params += np.prod(p.size())
        return n_params

    def FLOPS(self, in_shape):
        spatial_dim = float(in_shape[1] * in_shape[2]) / (float(self.stride)**2.)
        flops =  spatial_dim * self.in_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        flops += spatial_dim * self.out_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        if self.needs_conv_res:
            flops += spatial_dim * self.in_channels * self.out_channels * 1
            flops += spatial_dim * self.out_channels * 2
        return flops
    
    def data_overhead(self, in_shape):
        if self.stride == 1:
            out_w = in_shape[1] 
        else:
            out_w = in_shape[1] // self.stride
        conv1_out_shape = (self.conv1.weight.shape[0], out_w, out_w)
        # input_overhead + weight_overhead + output_overhead
        conv1_overhead = np.prod(in_shape) + np.prod(self.conv1.weight.shape) + np.prod(conv1_out_shape)

        bn1_overhead = np.prod(conv1_out_shape) *2 + np.prod(self.bn1.weight.shape)

        conv2_overhead = np.prod(conv1_out_shape) + np.prod(self.conv2.weight.shape) + np.prod(conv1_out_shape)
        bn2_overhead = np.prod(conv1_out_shape) *2 + np.prod(self.bn2.weight.shape)

        all_overhead = conv1_overhead + bn1_overhead + conv2_overhead + bn2_overhead

        if self.needs_conv_res:
            conv_overhead = np.prod(in_shape) + np.prod(self.conv_res.weight.shape) + np.prod(conv1_out_shape)
            bn_overhead = np.prod(conv1_out_shape) * 2 + np.prod(self.bn_res.weight.shape)
            res_overhead = conv_overhead + bn_overhead
            all_overhead += res_overhead
        return all_overhead
        


class BottleneckResBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckResBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Conv1
        self.conv1 = conv1x1(in_channels, out_channels, use_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv2
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride, use_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Conv3
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion, use_bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # ConvRes
        self.needs_conv_res = stride>1 or in_channels!=out_channels*self.expansion
        if self.needs_conv_res:
            self.conv_res = conv1x1(in_channels, out_channels * self.expansion, stride=stride, use_bias=False)
            self.bn_res = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        res = self.bn_res(self.conv_res(x)) if self.needs_conv_res else x
        return y + res

    def num_params(self):
        n_params = 0.
        for p in self.parameters():
            n_params += np.prod(p.size())
        return n_params

    def FLOPS(self, in_shape):
        spatial_dim = float(in_shape[1] * in_shape[2]) / (float(self.stride)**2.)
        flops =  spatial_dim * float(self.stride)**2. * self.in_channels * self.out_channels * 1
        flops += spatial_dim * float(self.stride)**2. * self.out_channels * 2
        flops += spatial_dim * self.out_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        flops += spatial_dim * self.out_channels * self.out_channels * self.expansion * 1
        flops += spatial_dim * self.out_channels * self.expansion * 2
        if self.needs_conv_res:
            flops += spatial_dim * self.in_channels * self.out_channels * self.expansion * 1
            flops += spatial_dim * self.out_channels * self.expansion * 2
        return flops
    
    def data_overhead(self, in_shape):
        
        conv1_out_shape = (self.out_channels, in_shape[1], in_shape[2])
        # input_overhead + weight_overhead + output_overhead
        conv1_overhead = np.prod(in_shape) + np.prod(self.conv1.weight.shape) + np.prod(conv1_out_shape)
        bn1_overhead = np.prod(conv1_out_shape) *2 + np.prod(self.bn1.weight.shape)

        if self.stride == 1:
            out_w = in_shape[1] 
        else:
            out_w = in_shape[1] // self.stride
        conv2_out_shape = (self.out_channels, out_w, out_w)
        conv2_overhead = np.prod(conv1_out_shape) + np.prod(self.conv2.weight.shape) + np.prod(conv2_out_shape)
        bn2_overhead = np.prod(conv2_out_shape) *2 + np.prod(self.bn2.weight.shape)

        conv3_out_shape = (self.out_channels * self.expansion, out_w, out_w)
        conv3_overhead = np.prod(conv2_out_shape) + np.prod(self.conv2.weight.shape) + np.prod(conv3_out_shape)
        bn3_overhead = np.prod(conv3_out_shape) *2 + np.prod(self.bn3.weight.shape)

        all_overhead = conv1_overhead + bn1_overhead + conv2_overhead + bn2_overhead + conv3_overhead + bn3_overhead

        if self.needs_conv_res:
            conv_overhead = np.prod(in_shape) + np.prod(self.conv_res.weight.shape) + np.prod(conv3_out_shape)
            bn_overhead = np.prod(conv3_out_shape) * 2 + np.prod(self.bn_res.weight.shape)
            res_overhead = conv_overhead + bn_overhead
            all_overhead += res_overhead
        return all_overhead
        


class BasicProxy(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicProxy, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Layer definition
        if stride > 1:
            self.maxpool = nn.MaxPool2d(kernel_size=stride+1, stride=stride, padding=stride//2)
        self.conv = conv1x1(in_channels, out_channels, use_bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.stride > 1:
            x = self.maxpool(x)
        return self.bn(self.conv(x))

    def num_params(self):
        n_params = 0.
        for p in self.parameters():
            n_params += np.prod(p.size())
        return n_params

    def FLOPS(self, in_shape):
        spatial_dim = float(in_shape[1] * in_shape[2]) / (float(self.stride)**2.)
        flops =  spatial_dim * self.in_channels * self.out_channels * 1
        flops += spatial_dim * self.out_channels * 2
        return flops
    
    def data_overhead(self, in_shape):
        if self.stride > 1:
            operate_shape = (in_shape[0], in_shape[1]//self.stride, in_shape[2]//self.stride)
            max_pool_overhead = np.prod(in_shape) + np.prod(operate_shape)
        else:
            operate_shape = in_shape
            max_pool_overhead = 0
        conv_overhead = np.prod(operate_shape) * 2 + np.prod(self.conv.weight.shape)
        bn_overhead = np.prod(operate_shape) * 2 + np.prod(self.bn.weight.shape)
        return  max_pool_overhead + conv_overhead + bn_overhead
        


class BottleneckProxy(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckProxy, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Layer definition
        if stride > 1:
            self.maxpool = nn.MaxPool2d(kernel_size=stride+1, stride=stride, padding=stride//2)
        self.conv1 = conv1x1(in_channels, max(out_channels//32, 16), use_bias=False)
        self.bn1 = nn.BatchNorm2d(max(out_channels//32, 16))
        self.conv2 = conv1x1(max(out_channels//32, 16), out_channels, use_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.stride > 1:
            x = self.maxpool(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return self.bn2(self.conv2(x))

    def num_params(self):
        n_params = 0.
        for p in self.parameters():
            n_params += np.prod(p.size())
        return n_params

    def FLOPS(self, in_shape):
        spatial_dim = float(in_shape[1] * in_shape[2]) / (float(self.stride)**2.)
        flops =  spatial_dim * self.in_channels * max(self.out_channels//32, 16) * 1
        flops += spatial_dim * max(self.out_channels//32, 16) * 2
        flops += spatial_dim * max(self.out_channels//32, 16) * self.out_channels * 1
        flops += spatial_dim * self.out_channels * 2
        return flops
    
    def data_overhead(self, in_shape):
        if self.stride > 1:
            operate_w = in_shape[1]//self.stride
            maxpool_out_shape = (self.in_channels, operate_w, operate_w)
            max_pool_overhead = np.prod(in_shape) + np.prod(maxpool_out_shape)
        else:
            operate_w = in_shape[1]
            max_pool_overhead = 0
        
        conv1_in_shape = (self.in_channels, operate_w, operate_w)
        conv1_out_shape = (self.conv1.weight.shape[0], operate_w, operate_w)
        conv1_overhead = np.prod(conv1_in_shape) + np.prod(self.conv1.weight.shape) + np.prod(conv1_out_shape)
        bn1_overhead = np.prod(conv1_out_shape) * 2 + np.prod(self.bn1.weight.shape)

        conv2_out_shape = (self.out_channels, operate_w, operate_w)
        conv2_overhead = np.prod(conv1_out_shape) + np.prod(self.conv2.weight.shape) + np.prod(conv2_out_shape)
        bn2_overhead = np.prod(conv2_out_shape) * 2 + np.prod(self.bn2.weight.shape)

        return  max_pool_overhead + conv1_overhead + bn1_overhead + conv2_overhead + bn2_overhead
        



class NetTailorBlock(nn.Module):
    def __init__(self, universal_block, proxy_block, out_channels, stride, skip_shapes):
        super(NetTailorBlock, self).__init__()
        self.skip_shapes = list(skip_shapes)[:]

        # Main block
        self.main = universal_block(skip_shapes[0][0], out_channels, stride)
        self.flops = [self.main.FLOPS(skip_shapes[0])]
        self.data_overhead_list = [self.main.data_overhead(skip_shapes[0])]
        target_shape = (out_channels * universal_block.expansion, skip_shapes[0][1]//stride, skip_shapes[0][2]//stride)

        # Proxies
        proxies = []
        for i, shape in enumerate(list(skip_shapes)):
            stride = shape[1]//target_shape[1]
            proxies.append(proxy_block(shape[0], target_shape[0], stride))
            self.flops.append(proxies[-1].FLOPS(shape))
            self.data_overhead_list.append(proxies[-1].data_overhead(shape))
        self.proxies = nn.ModuleList(proxies)

        # Block attention params
        self.alphas_params = Parameter(torch.Tensor(len(skip_shapes)+1), requires_grad=True)
        for i in range(len(skip_shapes)+1):
            self.alphas_params[i].data.fill_(2. if  i == 0 else -2.)
            # self.alphas_params[i].data.fill_(3. if  i == 0 else -3.)

        # Pruning flags
        self.keep_flag = [True]*(len(self.proxies) + 1)

    def forward(self, ends):
        if not any(self.keep_flag) or all([e is None for e in ends]):
            return None
        out = 0.
        for i, (alpha, k) in enumerate(zip(self.alphas(), self.keep_flag)):
            if not k:
                continue    # Skip prunned blocks
            inp = ends[0] if i==0 else ends[i-1]
            if inp is None:
                continue
            out += alpha * (self.main(inp) if i==0 else self.proxies[i-1](inp))
            # if sum(self.keep_flag) > 1:
            # print(f"Nettailor block forward {self.alphas()[0]}")
            # if self.alphas()[0].item() == 0.4221276640892029:
                # if i==0:
                #     print(f"Main input")
                #     print(inp[0,0,0,:10])
                #     print("Main output")
                #     print(self.main(inp)[0,0,0,:10])
                # else:
                # if i==3:
                #     print(f"Proxy input")
                #     print(inp[0,0,0,:10])
                #     print("Proxy output")
                #     print(self.proxies[i-1](inp)[0,0,0,:10])
                #     print(self.proxies[i-1].maxpool(inp)[0,0,0,:])
                #     st()

        return F.relu(out, inplace=True) if not isinstance(out, float) else None

    def alphas(self):
        # Block attention coefficients. Softmax over unprunned blocks.
        if sum(self.keep_flag) == 0:
            return torch.zeros_like(self.alphas_params)
        alphas_params = torch.stack([a for k, a in zip(self.keep_flag, self.alphas_params) if k])
        alphas = F.softmax(alphas_params, 0)
        if any([not k for k in self.keep_flag]):
            alphas = torch.stack([alphas[int(sum(self.keep_flag[:i]))] if k else torch.zeros_like(alphas[0]) for i, k in enumerate(self.keep_flag)])
        return alphas
        
    def num_params(self):
        return [self.main.num_params()] + [a.num_params() for a in self.proxies]

    def FLOPS(self):
        return self.flops
    
    def data_overhead(self):
        return self.data_overhead_list


class NetTailor(nn.Module):
    def __init__(self, backbone, num_classes, max_skip=1, pretrain_path=None):
        super(NetTailor, self).__init__()

        if backbone == 'resnet18':
            universal_block = BasicResBlock
            proxy_block = BasicProxy
            num_layers = [2, 2, 2, 2]

        elif backbone == 'resnet34':
            universal_block = BasicResBlock
            proxy_block = BasicProxy
            num_layers = [3, 4, 6, 3]

        elif backbone == 'resnet50':
            universal_block = BottleneckResBlock
            proxy_block = BottleneckProxy
            num_layers = [3, 4, 6, 3]

        elif backbone == 'resnet101':
            universal_block = BottleneckResBlock
            proxy_block = BottleneckProxy
            num_layers = [3, 4, 23, 3]
        else:
            raise NotImplementedError

        self.universal_block = universal_block
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_skip = max_skip

        # Initial convolution
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Track tensor shapes of last max_skip layers
        cur_shape = (64, 32, 32)
        # cur_shape = (64, 28, 28)
        skip_shapes = collections.deque(maxlen=self.max_skip)
        skip_shapes.appendleft(cur_shape)

        # Track FLOPS per block
        self.flops =[[float(cur_shape[1] * cur_shape[2]) * (3 * 64 * 7**2 + 64 * 2)]]
        # Track data overhead
        input_w = cur_shape[1] * 4
        conv1_output_w = cur_shape[1] * 2
        conv_overhead = input_w * input_w * 3 + conv1_output_w * conv1_output_w * 64 + 3 * 64 * 7 * 7
        bn_overhead = conv1_output_w * conv1_output_w * 64 * 2 + np.prod(self.bn1.weight.shape)
        maxpool_overhead = conv1_output_w * conv1_output_w * 64 + cur_shape[1] * cur_shape[2] * 64
        self.data_overhead_list = [
            [conv_overhead+bn_overhead+maxpool_overhead]
        ]
        layers = []
        for b in range(len(num_layers)):
            # Filter progression: 64 -> 128 -> 256 -> 512
            num_channels = 64*(2**b)

            for bb in range(num_layers[b]):
                # Add NetTailor layer consisting of 1 universal block and max_skip task-specific blocks
                stride = 2 if b > 0 and bb == 0 else 1
                # stride = 2 if bb == 0 else 1
                layers.append(NetTailorBlock(universal_block, proxy_block, num_channels, stride, skip_shapes))

                # Track tensor shapes of last max_skip layers
                cur_shape = (num_channels*universal_block.expansion, cur_shape[1]//stride, cur_shape[2]//stride)
                skip_shapes.appendleft(cur_shape)

                # Track FLOPS per block
                self.flops.append(layers[-1].FLOPS())
                self.data_overhead_list.append(layers[-1].data_overhead())
        self.layers = nn.ModuleList(layers)

        # Add final classifier
        self.classifier = nn.Linear(cur_shape[0], num_classes)
        self.flops.append([cur_shape[0] * num_classes + num_classes])
        self.data_overhead_list.append(
            [cur_shape[0] + num_classes + cur_shape[0] * num_classes]
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load universal blocks
        if pretrain_path is not None:
            self.load_pretrained(pretrain_path)

        # Freeze universal layers
        self.freeze_backbone()
        self.eval()

    # def forward(self, x, return_internal=False):
    #     ends = collections.deque(maxlen=self.max_skip)
    #     # x = self.maxpool(F.relu(self.bn1(self.conv1(x)), inplace=True))
    #     x = F.relu(self.bn1(self.conv1(x)), inplace=True)
    #     ends.appendleft(x)
    #     proxies = []
    #     for l in range(sum(self.num_layers)):
    #         p = self.layers[l]([e.detach() if e is not None else None for e in ends])
    #         proxies.append(p)
    #         x = self.layers[l](ends)
    #         ends.appendleft(x)
    #     x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
    #     x = self.classifier(x)
    #     if return_internal:
    #         return (x, proxies)
    #     else:
    #         return x

    def forward(self, x, return_internal=False):
        ends = collections.deque(maxlen=self.max_skip)
        # x = self.maxpool(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # x = self.bn1(self.conv1(x))

        ends.appendleft(x)
        proxies = []
        for l in range(sum(self.num_layers)):
        # for l in range(4):
            # p = self.layers[l]([e.detach() if e is not None else None for e in ends])
            # proxies.append(p)
            x = self.layers[l](ends)
            ends.appendleft(x)
        #     print(x.shape)
        # st()
        # return x

        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.classifier(x)
        if return_internal:
            return (x, proxies)
        else:
            return x

    def load_pretrained(self, pretrain_path):
        state = self.state_dict()
        # checkpoint = model_zoo.load_url(backbone_urls[base_arch])
        assert os.path.exists(pretrain_path)
        checkpoint = torch.load(pretrain_path)['state_dict']

        in_module = False
        if list(checkpoint.keys())[0].startswith('module.'):
            in_module = True

        # Match resnet layer names with nettailor's
        for k_st in state:

            if 'proxies' in k_st or 'alphas' in k_st or 'classifier' in k_st:
                continue    # Only load universal layers. Skip proxies, alphas and classifier

            k_tkn = k_st.split('.')
            if k_tkn[-1] == 'num_batches_tracked':
                continue
            if k_tkn[0] == 'conv1':
                k_ckp = 'conv1.{}'.format(k_tkn[-1])
            elif k_tkn[0] == 'bn1':
                k_ckp = 'bn1.{}'.format(k_tkn[-1])
            elif k_tkn[0] == 'layers':
                l = int(k_tkn[1])
                for b, ll in enumerate(self.num_layers):
                    if ll > l:
                        break
                    l -= ll

                if k_tkn[3] == 'conv_res':
                    k_ckp = 'layer{}.{}.downsample.0.{}'.format(b+1, l, k_tkn[-1])
                elif k_tkn[3] == 'bn_res':
                    k_ckp = 'layer{}.{}.downsample.1.{}'.format(b+1, l, k_tkn[-1])
                else:
                    k_tkn2 = k_tkn[0].split('-')
                    k_ckp = 'layer{}.{}.{}.{}'.format(b+1, l, k_tkn[3], k_tkn[-1])
            else:
                raise ValueError('Cannot match weights for universal layer '+k_st)
            
            if in_module:
                k_ckp = "module."+k_ckp
            assert all([s1 == s2 for s1, s2 in zip(checkpoint[k_ckp].size(), state[k_st].size())])
            state[k_st] = checkpoint[k_ckp]
        self.load_state_dict(state)

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if 'proxies' not in name and 'alphas' not in name and 'classifier' not in name:
                param.requires_grad = False

    def expected_complexity(self):
        
        alphas_list, complexity_list = [], []
        global_flops = float(sum([f[0] for f in self.flops[:-1]]))
        for layer, l_flops in zip(self.layers, self.flops[1:-1]):
            alphas_list.append([a for a in layer.alphas()])
            complexity_list.append([f/global_flops for f in l_flops])
        
        # Probability that incoming connections are clear
        incoming_alphas = [torch.prod(1.-torch.stack(alphas)) for alphas in alphas_list]

        # Probability that outgoing connections are clear
        outgoing_alphas = [[] for i in range(len(alphas_list))]
        for i, alphas in enumerate(alphas_list):
            for j in range(len(alphas)):
                src = i if j <= 1 else i-(j-1)
                outgoing_alphas[src].append(alphas[j])

        for i in range(len(outgoing_alphas)):
            outgoing_alphas[i] = torch.prod(1.-torch.stack(outgoing_alphas[i]))

        # Complexity computation
        C = 0.
        for i, (alphas, complixities) in enumerate(zip(alphas_list, complexity_list)):
            for j in range(1, len(alphas)):
                c = complixities[j]
                p = alphas[j]
                p_in = torch.tensor(1., requires_grad=False).to(p.device) if i == 0 else incoming_alphas[i-1]
                p_out = torch.tensor(1., requires_grad=False).to(p.device) if i == len(outgoing_alphas)-1 else outgoing_alphas[i+1]
                c_layer = c * (p - p_in - p_out)
                C += c_layer
        return C

    def expected_data_overhead(self):
        
        alphas_list, overhead_complexity_list = [], []
        global_overhead = float(sum([d[0] for d in self.data_overhead_list[:-1]]))
        for layer, l_overhead in zip(self.layers, self.data_overhead_list[1:-1]):
            alphas_list.append([a for a in layer.alphas()])
            overhead_complexity_list.append([d/global_overhead for d in l_overhead])
        
        # Probability that incoming connections are clear
        incoming_alphas = [torch.prod(1.-torch.stack(alphas)) for alphas in alphas_list]

        # Probability that outgoing connections are clear
        outgoing_alphas = [[] for i in range(len(alphas_list))]
        for i, alphas in enumerate(alphas_list):
            for j in range(len(alphas)):
                src = i if j <= 1 else i-(j-1)
                outgoing_alphas[src].append(alphas[j])

        for i in range(len(outgoing_alphas)):
            outgoing_alphas[i] = torch.prod(1.-torch.stack(outgoing_alphas[i]))

        # Complexity computation
        C = 0.
        for i, (alphas, overhead_complixities) in enumerate(zip(alphas_list, overhead_complexity_list)):
            for j in range(len(alphas)):
                c = overhead_complixities[j]
                p = alphas[j]
                p_in = torch.tensor(1., requires_grad=False).to(p.device) if i == 0 else incoming_alphas[i-1]
                p_out = torch.tensor(1., requires_grad=False).to(p.device) if i == len(outgoing_alphas)-1 else outgoing_alphas[i+1]
                c_layer = c * (p - p_in - p_out)
                C += c_layer
        return C


    def threshold_alphas(self, num_global=None, thr_global=None, percent_global=None, num_proxies=None, thr_proxies=None, percent_proxies=None, only_top=False):
        assert sum([num_global is not None, thr_global is not None, percent_global is not None]) <= 1
        assert sum([num_proxies is not None, thr_proxies is not None, percent_proxies is not None]) <= 1

        alphas_proxies, alphas_layer, keep = [], [], []
        for layer in self.layers:
            aa = layer.alphas().data.cpu().numpy()
            alphas_layer.append(aa[0])
            alphas_proxies.append(aa[1:])
            keep.append([1.]*aa.size)

        # Remove global layers
        alphas_layer = np.array(alphas_layer)
        if num_global is not None:
            to_rm = np.argsort(alphas_layer)[:num_global]
        elif percent_global is not None:
            to_rm = np.argsort(alphas_layer)[:int(len(alphas_layer)*percent_global)]
        elif thr_global is not None:
            to_rm = [i for i, aa in enumerate(alphas_layer) if aa <= thr_global]
        for rm_idx in to_rm:
            keep[rm_idx][0] = 0.

        # Remove task-specific layers
        meta_proxies = [(i, j) for i in range(len(alphas_proxies)) for j in range(alphas_proxies[i].size)]
        alphas_proxies = np.concatenate(alphas_proxies)
        to_rm = []
        if num_proxies is not None:
            to_rm.extend(np.argsort(alphas_proxies)[:num_proxies].tolist())
        elif percent_proxies is not None:
            to_rm.extend(np.argsort(alphas_proxies)[:int(len(alphas_proxies)*percent_proxies)].tolist())
        elif thr_proxies is not None:
            to_rm.extend([i for i, aa in enumerate(alphas_proxies) if aa <= thr_proxies])
        
        if only_top:
            for i in range(len(self.layers)):
                adp_a = [aa for ii, (mm, aa) in enumerate(zip(meta_proxies, alphas_proxies)) if mm[0]==i]
                adp_i = [ii for ii, (mm, aa) in enumerate(zip(meta_proxies, alphas_proxies)) if mm[0]==i]
                to_rm.extend([ii for ii, aa in zip(adp_i, adp_a) if aa/max(adp_a)<0.5])

        for rm_idx in to_rm:
            layer_idx = meta_proxies[rm_idx][0]
            proxies_idx = meta_proxies[rm_idx][1]+1
            keep[layer_idx][proxies_idx] = 0.

        # Update keep variables
        for layer, k in zip(self.layers, keep):
            layer.keep_flag = k[:]

    def get_keep_flags(self):
        return [layer.keep_flag for layer in self.layers]

    def load_keep_flags(self, keep_flags):
        assert len(keep_flags)==len(self.layers)
        for keep, layer in zip(keep_flags, self.layers):
            assert len(keep)==len(layer.keep_flag)
            layer.keep_flag = keep[:]

    def global_params(self):
        global_params = [sum([np.prod(p.size()) for p in self.conv1.parameters()])+sum([np.prod(p.size()) for p in self.bn1.parameters()])]
        for layer in self.layers:
            if layer.keep_flag[0] > 0:
                global_params.append(layer.num_params()[0])
            else:
                global_params.append(0)
        return global_params

    def task_params(self):
        task_params = []
        for layer in self.layers:
            task_params.append([c*float(k) for c, k in zip(layer.num_params()[1:], layer.keep_flag[1:])])
        task_params.append([self.universal_block.expansion * 512 * self.num_classes + self.num_classes])
        return task_params

    def alphas_and_complexities(self):
        alphas = "\n{}   Alphas (Num params, FLOPS)   {}\n".format("="*30, "="*30)
        for layer, l_flops in zip(self.layers, self.flops[1:-1]):
            pp = layer.num_params()
            ff = l_flops
            aa = layer.alphas().data.cpu().numpy()
            kk = layer.keep_flag
            for a, k, p, f in zip(aa, kk, pp, ff):
                alphas += '{} {:.3f} ({:.3f}, {:.3f}) |  '.format('X' if k else ' ', a, p/10**6, f/10**6)
            alphas += '\n'
        return alphas

    def alphas_and_data_overhead(self):
        alphas = "\n{}   Alphas (Num params, data overhead, FLOPS)   {}\n".format("="*30, "="*30)
        for layer, l_flops in zip(self.layers, self.flops[1:-1]):
            pp = layer.num_params()
            ff = l_flops
            aa = layer.alphas().data.cpu().numpy()
            kk = layer.keep_flag
            dd = layer.data_overhead()
            for a, k, p, f, d in zip(aa, kk, pp, ff, dd):
                alphas += '{} {:.3f} ({:.3f}, {:.3f}, {:.3f}) |  '.format('X' if k else ' ', a, p/10**6, d/10**6, f/10**6)
            alphas += '\n'
        return alphas


    def stats(self):
        network_stats = self.alphas_and_complexities()

        network_stats += "\n{}   Parameters   {}\n".format("="*30, "="*30)
        global_params, task_params = 0, 0

        global_params += self.global_params()[0]
        network_stats += "{}\n".format(self.global_params()[0]/10.**6)
        for gp, tp in zip(self.global_params()[1:], self.task_params()[:-1]):
            global_params += gp
            task_params += sum(tp) if isinstance(tp, list) else tp
            p = [gp] + tp if isinstance(tp, list) else [tp]
            network_stats += "{}\n".format(''.join(["{:15}".format(str(pp/10.**6)) for pp in p]))
        task_params += self.task_params()[-1][0]
        network_stats += "{}\n".format(self.task_params()[-1][0]/10.**6)

        network_stats +=  "\nGlobal Parameters {}\n".format(global_params/10.**6)
        network_stats +=  "Task Parameters {}\n".format(task_params/10.**6)
        network_stats +=  "Total {}\n".format(global_params/10.**6+task_params/10.**6)

        network_stats += "\n{}   Data Overhead   {}\n".format("="*30, "="*30)
        global_overhead, task_overhead = 0., 0.
        total_overhead = self.data_overhead_list[0][0]
        network_stats += "{}\n".format(self.data_overhead_list[0][0]/10.**5)
        for l, dd in zip(self.layers,self.data_overhead_list[1:-1]):
            global_overhead += dd[0]
            task_overhead += sum([d if k else 0 for d, k in zip(dd[1:], l.keep_flag[1:])])
            total_overhead += sum([d if k else 0 for d, k in zip(dd, l.keep_flag)])
            network_stats += "{}\n".format(''.join(["{:15}".format(str(d/10.**5 if k else 0.)) for d, k in zip(dd, l.keep_flag)]))
        total_overhead += self.flops[-1][0]
        network_stats += "{}\n".format(self.data_overhead_list[-1][0]/10.**5)
        network_stats +=  "\nTotal {:.4f}, global {:.4f}, task {:.4f}\n".format(
            total_overhead/10.**5, global_overhead/10.**5, task_overhead/10.**5
        )


        network_stats += "\n{}   FLOPS   {}\n".format("="*30, "="*30)
        total_flops = self.flops[0][0]
        network_stats += "{}\n".format(self.flops[0][0]/10.**9)
        for l, ff in zip(self.layers,self.flops[1:-1]):
            total_flops += sum([f if k else 0 for f, k in zip(ff, l.keep_flag)])
            network_stats += "{}\n".format(''.join(["{:15}".format(str(f/10.**9 if k else 0.)) for f, k in zip(ff, l.keep_flag)]))
        total_flops += self.flops[-1][0]
        network_stats += "{}\n".format(self.flops[-1][0]/10.**9)
        network_stats +=  "\nTotal {}\n".format(total_flops/10.**9)

        return network_stats

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


def create_model(backbone, num_classes, max_skip, pretrain_path=None):
    return NetTailor(backbone, num_classes, max_skip=max_skip, pretrain_path=pretrain_path)

##########################################################################################

if __name__ == '__main__':
    device = torch.device("cpu")
    model = create_model('resnet18', 10, max_skip=3)
    model.to(device)

    print('\n'+'='*30+'  Model  '+'='*30)
    print(model)

    print('\n'+'='*30+'  Parameters  '+'='*30)
    for n, p in model.named_parameters():
        print("{:50} | {:10} | {:30} | {:20} | {}".format(
            n, 'Trainable' if p.requires_grad else 'Frozen' , 
            str(p.size()), str(np.prod(p.size())), str(p.type()))
        )

    print(model.stats())
    print(model.expected_complexity())
    
    inp = torch.rand(2, 3, 224, 224)
    inp = inp.to(device)
    print('Input:', inp.shape)
    out, proxies = model(inp)
    for p in proxies:
        print('Proxy:', p.shape)
    print('Output', out.shape)
    labels = torch.zeros(2,).type(torch.LongTensor)
    loss = nn.CrossEntropyLoss()(out, labels)
    loss.backward()


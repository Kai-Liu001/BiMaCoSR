import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class SignSTE_BirealFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # 初始化梯度为零
        grad_input = torch.zeros_like(input)
        
        # 定义各个区间的掩码
        mask1 = input < -1
        mask2 = (input >= -1) & (input < 0)
        mask3 = (input >= 0) & (input < 1)
        mask4 = input >= 1
        
        # 计算各个区间的梯度
        grad_input += (2 * input + 2) * mask2.type_as(input)
        grad_input += (-2 * input + 2) * mask3.type_as(input)
        # 对于 mask1 和 mask4，梯度保持为0
        
        # 将自定义梯度与上游梯度相乘
        return grad_output * grad_input

class SignSTE_Bireal(nn.Module):
    def forward(self, input):
        return SignSTE_BirealFunc.apply(input)

class BinaryConv2d_BiDM(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        super(BinaryConv2d_BiDM, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.channel_threshold = nn.Parameter(torch.zeros((1, in_channels, 1, 1), requires_grad=True))
        self.fliter_k = nn.Conv2d(1, 1, kernel_size, stride=stride, padding=padding, bias=False)
        self.fliter_k.weight.data = torch.full(self.fliter_k.weight.shape, 1/kernel_size/kernel_size)
        self.shortcut_scale = nn.Parameter(torch.ones(1) * 0.3, requires_grad=True)
        self.init_fliter_k = False
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=0)
        self.quantize = SignSTE_Bireal()
    def forward(self, x):
        if not self.init_fliter_k:
            kernel_area = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0] * self.kernel_size[1]
            self.fliter_k.weight.data = torch.full(self.fliter_k.weight.shape, 1/kernel_area).cuda()
            self.init_fliter_k = True

        x_raw = x
        # binary activation
        scaling_a_in = torch.mean(abs(x), dim=1, keepdim=True)
        scaling_a_out = self.fliter_k(scaling_a_in)

        x = x + self.channel_threshold
        x= self.quantize(x)
        # Binary weights
        real_weights = self.weight
        scaling_factor = torch.mean(real_weights.abs(), dim=[1, 2, 3], keepdim=True).detach()
        binary_weights = scaling_factor * torch.sign(real_weights)
        binary_weights = binary_weights.detach() - torch.clamp(real_weights, -1.0, 1.0).detach() + torch.clamp(real_weights, -1.0, 1.0)

        y = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding)
        y = y * scaling_a_out

        # Shortcut connection
        if self.in_channels == self.out_channels:
            if x_raw.shape[-1] < y.shape[-1]:
                shortcut = F.interpolate(x_raw, scale_factor=2, mode="nearest")
            elif x_raw.shape[-1] > y.shape[-1]:
                shortcut = F.avg_pool2d(x_raw, kernel_size=self.stride, stride=self.stride)
            else:
                shortcut = x_raw
        else:
            shortcut = self.shortcut(x_raw)

        return y + shortcut * torch.abs(self.shortcut_scale)

def init_BinaryConv2d_BiDM_from_conv(conv):
    bnn_conv = BinaryConv2d_BiDM(conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.bias is not None)
    bnn_conv.weight=conv.weight
    if conv.bias is not None:
        bnn_conv.bias=conv.bias
    return bnn_conv

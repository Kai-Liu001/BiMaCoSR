import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Any
from torch.nn import Module, Linear, Parameter, Conv2d, ReLU
import torch
from torch import Tensor, FloatTensor
from torch.autograd import Function
from torch.autograd.function import _ContextMethodMixin
from functools import partial
import torch.nn.functional as F
#-------------------reactnet.py-----
#--------------utils----------------
def HORQ_grad(real_weights, N=1):
    residual = real_weights.clone()
    binary_total = torch.zeros_like(real_weights)

    # 根据权重的形状判断是卷积层还是线性层
    if real_weights.ndimension() == 4:
        # 卷积层的情况
        for i in range(N):
            # 计算缩放因子（按通道求平均绝对值）
            scaling_factor = torch.mean(residual.abs(), dim=[ 1,2, 3], keepdim=True).detach()
            # 基于当前残差的符号与缩放因子构造二值化近似
            binary_approx = scaling_factor * torch.sign(residual)
            
            # 累加本阶的二值化结果
            binary_total += binary_approx
            # 更新残差
            residual = residual - binary_approx

    elif real_weights.ndimension() == 2:
        # 线性层的情况
        for i in range(N):
            # 计算缩放因子（按特征求平均绝对值）
            scaling_factor = torch.mean(residual.abs(), dim=1, keepdim=True).detach()
            # 基于当前残差的符号与缩放因子构造二值化近似
            binary_approx = scaling_factor * torch.sign(residual)
            
            # 累加本阶的二值化结果
            binary_total += binary_approx
            # 更新残差
            residual = residual - binary_approx

    else:
        raise ValueError("输入的权重形状不支持，只支持卷积层（4D）和线性层（2D）")

    # 处理二值化总和，并返回
    binary_total = binary_total.detach() - torch.clamp(real_weights, -1.0, 1.0).detach() + torch.clamp(real_weights, -1.0, 1.0)
    return binary_total

class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))
        return x

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

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

#--------CONV2D------------

class BinaryConv2d_reactnet(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=True):
        super(BinaryConv2d_reactnet, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.move0 = LearnableBias(self.in_channels)
        self.binary_activation = SignSTE_Bireal()
        self.relu = RPReLU(self.out_channels)
        self.quantize = SignSTE_Bireal()

    def forward(self, x):
        x_raw = x
        # Activation quantization
        x = self.move0(x)
        x = self.quantize(x)
        # Weight quantization and convolution
        real_weights = self.weight
        binary_weights = HORQ_grad(real_weights, N=1)
        x = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding)
        # shortcut
        if x_raw.size() == x.size():
            x = x + x_raw
        # RPReLU
        x = self.relu(x)
        return x

def init_BinaryConv2d_reactnet_from_conv(conv):
    binary_conv = BinaryConv2d_reactnet(conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.bias is not None)
    binary_conv.weight = conv.weight
    if conv.bias is not None:
        binary_conv.bias = conv.bias
    return binary_conv
#--------LINEAR------------
class BinaryLinear_reactnet(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BinaryLinear_reactnet, self).__init__(in_features, out_features, bias=bias)
        self.binary_act = binary_act
        self.channel_threshold = torch.nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)
        self.quantize = SignSTE_Bireal()
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, input):
        x = input + self.channel_threshold
        input = self.quantize(x)
        real_weights = self.weight
        binary_weights = HORQ_grad(real_weights, N=1)
        output = F.linear(input, binary_weights, self.bias)
        if self.in_features == self.out_features:
            output = output + x
        return output

def init_BinaryLinear_reactnet_from_Linear(linear):
    binary_linear = BinaryLinear_reactnet(linear.in_features, linear.out_features, linear.bias is not None)
    binary_linear.weight = linear.weight
    if linear.bias is not None:
        binary_linear.bias = linear.bias
    return binary_linear




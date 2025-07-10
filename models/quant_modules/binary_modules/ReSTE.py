import logging
import os
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from scipy.stats import ortho_group

#---------------------------utils---------------------------
# ReSTE
class Binary_ReSTE(Function):
    @staticmethod
    def forward(ctx, input, t, o):
        ctx.save_for_backward(input, t, o)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, t, o = ctx.saved_tensors

        interval = 0.1

        tmp = torch.zeros_like(input)
        mask1 = (input <= t) & (input > interval)
        tmp[mask1] = (1 / o) * torch.pow(input[mask1], (1 - o) / o)
        mask2 = (input >= -t) & (input < -interval)
        tmp[mask2] = (1 / o) * torch.pow(-input[mask2], (1 - o) / o)
        tmp[(input <= interval) & (input >= 0)] = approximate_function(interval, o) / interval
        tmp[(input <= 0) & (input >= -interval)] = -approximate_function(-interval, o) / interval

        # calculate the final gradient
        grad_input = tmp * grad_output.clone()

        return grad_input, None, None

def approximate_function(x, o):
    if x >= 0:
        return math.pow(x, 1 / o)
    else:
        return -math.pow(-x, 1 / o)

#--------------------------conv2d----------------------------
class BinaryConv2d_ReSTE(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=True):
        super(BinaryConv2d_ReSTE, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        # ReSTE parameters
        self.t = torch.tensor(1.5).float()
        self.o = torch.tensor(1).float()
        self.t_a = torch.tensor(1.5).float()
        self.o_a = torch.tensor(1).float()

    def forward(self, x):
        a0 = x
        w0 = self.weight
        bw = Binary_ReSTE().apply(w0, self.t.to(w0.device), self.o.to(w0.device))
        ba = Binary_ReSTE().apply(a0, self.t_a.to(w0.device), self.o_a.to(w0.device))
        # scaling factor
        scaler = torch.mean(torch.abs(w0), dim=(1, 2, 3), keepdim=True)
        bw = bw * scaler
        # 1bit conv
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        return output

def init_BinaryConv2d_ReSTE_from_conv(conv):
    binary_conv = BinaryConv2d_ReSTE(conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.bias is not None)
    binary_conv.weight = conv.weight
    if conv.bias is not None:
        binary_conv.bias = conv.bias
    return binary_conv
#---------------------------linear---------------------------

class BinaryLinear_ReSTE(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BinaryLinear_ReSTE, self).__init__(in_features, out_features, bias=bias)
        
        # ReSTE parameters
        self.t = torch.tensor(1.5).float()
        self.o = torch.tensor(1).float()
        self.t_a = torch.tensor(1.5).float()
        self.o_a = torch.tensor(1).float()

    def forward(self, input):
        a0 = input
        w0 = self.weight
        bw = Binary_ReSTE().apply(w0, self.t.to(w0.device), self.o.to(w0.device))
        ba = Binary_ReSTE().apply(a0, self.t_a.to(w0.device), self.o_a.to(w0.device))
        # scaling factor
        scaler = torch.mean(torch.abs(w0), dim=(1), keepdim=True)
        bw = bw * scaler
        # 1bit conv
        output = F.linear(ba, bw, self.bias)
        return output


def init_BinaryLinear_ReSTE_from_Linear(linear):
    binary_linear = BinaryLinear_ReSTE(linear.in_features, linear.out_features, linear.bias is not None)
    binary_linear.weight = linear.weight
    if linear.bias is not None:
        binary_linear.bias = linear.bias
    return binary_linear

if __name__ == "__main__":
    pass
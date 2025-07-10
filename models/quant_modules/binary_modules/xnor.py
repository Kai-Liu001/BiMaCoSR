import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------utils-----------------
class SignSTE_clip_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input
class SignSTE_clip(nn.Module):
    def forward(self, input):
        return SignSTE_clip_Func.apply(input)
#-----------------use HORQ for normal format-----------------
def HORQ_grad(real_weights, N=1):
    residual = real_weights.clone()
    binary_total = torch.zeros_like(real_weights)
    if real_weights.ndimension() == 4:
        for i in range(N):
            scaling_factor = torch.mean(residual.abs(), dim=[ 1,2, 3], keepdim=True).detach()
            binary_approx = scaling_factor * torch.sign(residual)
            binary_total += binary_approx
            residual = residual - binary_approx
    elif real_weights.ndimension() == 2:
        for i in range(N):
            scaling_factor = torch.mean(residual.abs(), dim=1, keepdim=True).detach()
            binary_approx = scaling_factor * torch.sign(residual)
            binary_total += binary_approx
            residual = residual - binary_approx
    else:
        raise ValueError("wrong dim, only support 2D and 4D")
    binary_total = binary_total.detach() - torch.clamp(real_weights, -1.0, 1.0).detach() + torch.clamp(real_weights, -1.0, 1.0)
    return binary_total

def HORQ_grad_activation(activations, N=1, oursign=SignSTE_clip()):
    """
    binary the activation
    :param activations: the activation
    :param N: the number of iteration
    :param oursign: the sign function
    :return: the binary activation
    """
    residual = activations.clone()
    binary_total = torch.zeros_like(activations)

    if activations.ndimension() == 4:
        for _ in range(N):
            scaling_factor = torch.mean(residual.abs(), dim=[1], keepdim=True).detach()
            binary_approx = scaling_factor * oursign(residual)
            binary_total += binary_approx
            residual = residual - binary_approx

    elif activations.ndimension() == 3:
        for _ in range(N):
            scaling_factor = torch.mean(residual.abs(), dim=[2], keepdim=True).detach()
            binary_approx = scaling_factor * oursign(residual)
            binary_total += binary_approx
            residual = residual - binary_approx

    elif activations.ndimension() == 2:
        for _ in range(N):
            scaling_factor = torch.mean(residual.abs(), dim=1, keepdim=True).detach()
            binary_approx = scaling_factor * oursign(residual)
            binary_total += binary_approx
            residual = residual - binary_approx
    else:
        raise ValueError("wrong dim, only support 2D, 3D and 4D")
    return binary_total

#-----------------our version for init conv-----------------
class BinaryConv2d_XNOR(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(BinaryConv2d_XNOR, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.quantize = SignSTE_clip()

    def forward(self, x):
        # Activation quantization
        x=HORQ_grad_activation(x, N=1, oursign=self.quantize)
        # Weight quantization and convolution
        real_weights = self.weight
        binary_weights = HORQ_grad(real_weights, N=1)
        x = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding)
        return x

def init_BinaryConv2d_XNOR_from_conv(conv):
    binary_conv = BinaryConv2d_XNOR(conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.bias is not None)
    binary_conv.weight = conv.weight
    if conv.bias is not None:
        binary_conv.bias = conv.bias
    return binary_conv

#-----------------our version for init linear-----------------
class BinaryLinear_XNOR(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear_XNOR, self).__init__(in_features, out_features, bias=bias)
        self.quantize = SignSTE_clip()
    def forward(self, input):
        # Activation quantization
        input=HORQ_grad_activation(input, N=1, oursign=self.quantize)
        # Weight quantization and linear
        real_weights = self.weight
        binary_weights = HORQ_grad(real_weights, N=1)
        output = F.linear(input, binary_weights, self.bias)
        return output

def init_BinaryLinear_XNOR_from_Linear(linear):
    binary_linear = BinaryLinear_XNOR(linear.in_features, linear.out_features, linear.bias is not None)
    binary_linear.weight = linear.weight
    if linear.bias is not None:
        binary_linear.bias = linear.bias
    return binary_linear


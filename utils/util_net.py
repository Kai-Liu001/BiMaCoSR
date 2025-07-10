#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-11-24 20:29:36

import math
import torch
from pathlib import Path
from collections import OrderedDict
import torch.nn.functional as F


def calculate_parameters(net):
    out = 0
    for param in net.parameters():
        out += param.numel()
    return out

def pad_input(x, mod):
    h, w = x.shape[-2:]
    bottom = int(math.ceil(h/mod)*mod -h)
    right = int(math.ceil(w/mod)*mod - w)
    x_pad = F.pad(x, pad=(0, right, 0, bottom), mode='reflect')
    return x_pad

def forward_chop(net, x, net_kwargs=None, scale=1, shave=10, min_size=160000):
    n_GPUs = 1
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if net_kwargs is None:
                sr_batch = net(lr_batch)
            else:
                sr_batch = net(lr_batch, **net_kwargs)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def measure_time(net, inputs, num_forward=100):
    '''
    Measuring the average runing time (seconds) for pytorch.
    out = net(*inputs)
    '''
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.set_grad_enabled(False):
        for _ in range(num_forward):
            out = net(*inputs)
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / 1000

def reload_model(model, ckpt):
    if list(model.state_dict().keys())[0].startswith('module.'):
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = ckpt
        else:
            ckpt = OrderedDict({f'module.{key}':value for key, value in ckpt.items()})
    else:
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = OrderedDict({key[7:]:value for key, value in ckpt.items()})
        else:
            ckpt = ckpt
    model.load_state_dict(ckpt,strict=False)

def reload_model_print(model, ckpt, output_path='/home/yangkaicheng/Sin-SR-Quant/SinSR-main/load.txt', verbose=False):
    """
    加载权重到模型中，并打印哪些权重被加载到了哪些层上，同时将输出记录到指定文件中。

    参数:
    - model: 要加载权重的模型。
    - ckpt: 包含权重的检查点（state_dict）。
    - output_path: 输出日志文件的路径（默认路径）。
    - verbose: 是否打印加载的权重信息（默认值为 False）。
    """
    # 调整检查点的键以匹配模型的键
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(ckpt.keys())

    if model_keys and model_keys[0].startswith('module.'):
        if not ckpt_keys[0].startswith('module.'):
            # 为检查点的键添加 'module.' 前缀
            ckpt = OrderedDict({f'module.{key}': value for key, value in ckpt.items()})
    else:
        if ckpt_keys and ckpt_keys[0].startswith('module.'):
            # 移除检查点键的 'module.' 前缀
            ckpt = OrderedDict({key[7:]: value for key, value in ckpt.items()})

    # 加载状态字典，使用 strict=False 以便捕获缺失和多余的键
    load_result = model.load_state_dict(ckpt, strict=False)

    # 开始将输出写入指定文件
    with open(output_path, 'w') as f:
        # 打印成功加载的权重
        loaded_keys = set(ckpt.keys()) & set(model.state_dict().keys())
        if loaded_keys:
            f.write("已成功加载的权重:\n")
            if verbose:
                print("\n已成功加载的权重:")
            for key in sorted(loaded_keys):
                line = f"  权重 '{key}' 已加载到模型层 '{key}'\n"
                f.write(line)
                if verbose:
                    print(line.strip())

        # 打印缺失的键（模型中存在但检查点中不存在的键）
        if load_result.missing_keys:
            f.write("\n缺失的键（模型中存在但检查点中不存在）:\n")
            if verbose:
                print("\n缺失的键（模型中存在但检查点中不存在）:")
            for key in load_result.missing_keys:
                line = f"  缺失键: {key}\n"
                f.write(line)
                if verbose:
                    print(line.strip())

        # 打印意外的键（检查点中存在但模型中不存在的键）
        if load_result.unexpected_keys:
            f.write("\n意外的键（检查点中存在但模型中不存在）:\n")
            if verbose:
                print("\n意外的键（检查点中存在但模型中不存在）:")
            for key in load_result.unexpected_keys:
                line = f"  意外键: {key}\n"
                f.write(line)
                if verbose:
                    print(line.strip())

        # 写入和打印加载完成的提示
        f.write("\n权重加载完成。\n")
        if verbose:
            print("\n权重加载完成。")

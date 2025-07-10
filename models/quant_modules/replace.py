import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from .binary_modules import *
from .binary_modules.BiDM import init_BinaryConv2d_BiDM_from_conv
from .binary_modules.ours import *
from .binary_modules.xnor import *
from .binary_modules.HORQ import *
from .binary_modules.ReSTE import *
from .binary_modules.bbcu import *
from .binary_modules.reactnet import *
func={'BinaryConv2d_reactnet':init_BinaryConv2d_reactnet_from_conv,
      'BinaryLinear_reactnet':init_BinaryLinear_reactnet_from_Linear,
      'BinaryConv2d_BiDM':init_BinaryConv2d_BiDM_from_conv,
      'BinaryLinear_BiDM':init_BinaryLinear_reactnet_from_Linear,
      'BinaryConv2d_XNOR':init_BinaryConv2d_XNOR_from_conv,
      'BinaryLinear_XNOR':init_BinaryLinear_XNOR_from_Linear,
      'BinaryConv2d_ours_with_lora':init_BinaryConv2d_ours_from_conv_with_lora,
      'BinaryLinear_ours_with_lora':init_BinaryLinear_ours_from_Linear_with_lora,
      'BinaryConv2d_ours_with_skip':init_BinaryConv2d_ours_from_conv_with_skip,
      'BinaryLinear_ours_with_skip':init_BinaryLinear_ours_from_Linear_with_skip,
      'BinaryConv2d_ours_with_skip_and_lora':init_BinaryConv2d_ours_from_conv_with_skip_and_lora,
      'BinaryLinear_ours_with_skip_and_lora':init_BinaryLinear_ours_from_Linear_with_skip_and_lora,
      'BinaryConv2d_ours_baseline':init_BinaryConv2d_ours_from_conv_with_nothing,
      'BinaryLinear_ours_baseline':init_BinaryLinear_ours_from_Linear_with_nothing,
      'BinaryConv2d_HORQ':init_BinaryConv2d_HORQ_from_conv,
      'BinaryLinear_HORQ':init_BinaryLinear_HORQ_from_Linear,
      'BinaryConv2d_ReSTE':init_BinaryConv2d_ReSTE_from_conv,
      'BinaryLinear_ReSTE':init_BinaryLinear_ReSTE_from_Linear,
      'BinaryConv2d_BBCU':init_BinaryConv2d_BBCU_from_conv,
      'BinaryLinear_BBCU':init_BinaryLinear_BBCU_from_Linear,
      }
def identity(x):
    return x
def replace_conv_layers_allclass(model, 
                                 quant_modules='BinaryConv2d_reactnet',
                                 whitch_fp=[{'skip_keywords':['input_blocks.0.0', 'skip_connection','out.2'],
                                             'param_threshold':0.2,
                                             'use_for_fp':'nn.Conv2d'}],
                                 init_args=None,
                                 init_kwargs=None):
    """
    遍历模型中的所有层，将符合条件的Conv2d层替换为对应的量化层
    quant_modules: str, 量化模块名称
    whitch_fp: list, 量化条件列表，每个元素为一个字典，包含两个键值对，skip_keywords和param_threshold，只有满足某一个字典中的两个条件才不会被量化
    """
    init_conv=identity
    if_illegal=False
    for key, value in func.items():
        if key == quant_modules:
            init_conv = value
            if_illegal=True
        else :
            continue
    if if_illegal is False:
        print("不在列表的量化方法，跳过")
        return model
    module_dict = {}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(m, nn.Conv2d):
            if_fp="No"
            for item in whitch_fp:
                skip_keywords = []
                param_threshold = 0
                use_for_fp = None
                for key, value in item.items():
                    if key == 'skip_keywords':
                        skip_keywords = value
                    elif key == 'param_threshold':
                        param_threshold = value
                    elif key == 'use_for_fp':
                        use_for_fp = value
                total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                if any(keyword in name for keyword in skip_keywords) and total_params<= param_threshold*1e6:
                    if_fp=use_for_fp
                    break
            if if_fp == "No":
                idx = idx + 1 if idx != 0 else idx
                new_m = init_conv(m, *(init_args or ()), **(init_kwargs or {}))
                new_m.to(m.weight.device)
                setattr(father_module, name[idx:], new_m)      
                print(f"replace {name} with {quant_modules}")
            elif if_fp == "nn.Conv2d":
                print(f"skip {name} with nn.Conv2d")
            else:
                init_conv_2=None
                for key, value in func.items():
                    if key == if_fp:
                        init_conv_2 = value
                idx = idx + 1 if idx != 0 else idx
                new_m = init_conv_2(m, *(init_args or ()), **(init_kwargs or {}))
                new_m.to(m.weight.device)
                setattr(father_module, name[idx:], new_m)      
                print(f"replace {name} with {if_fp} as the another choice")
                
    return model

def replace_linear_layers_allclass(model, quant_modules='BinaryLinear_BiLLM',whitch_fp=[],init_args=None,
                                 init_kwargs=None):
    """
    遍历模型中的所有层，将符合条件的linear层替换为对应的量化层
    quant_modules: str, 量化模块名称
    whitch_fp: list, 量化条件列表，每个元素为一个字典，包含两个键值对，skip_keywords和param_threshold，只有满足某一个字典中的两个条件才不会被量化
    """
    if_illegal=False
    for key, value in func.items():
        if key == quant_modules:
            init_linear = value
            if_illegal=True
        else :
            continue
    if if_illegal is False:
        print("不在列表的量化方法，跳过")
        return model
    module_dict = {}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(m, nn.Linear):
            if_fp=False
            for item in whitch_fp:
                skip_keywords = []
                param_threshold = 0
                for key, value in item.items():
                    if key == 'skip_keywords':
                        skip_keywords = value
                    elif key == 'param_threshold':
                        param_threshold = value
                total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                if any(keyword in name for keyword in skip_keywords) and total_params<= param_threshold*1e6:
                    if_fp=True
                    break
            if if_fp is False:
                idx = idx + 1 if idx != 0 else idx
                new_m = init_linear(m, *(init_args or ()), **(init_kwargs or {})).cuda()                    
                setattr(father_module, name[idx:], new_m)      
                print(f"replace {name} with {quant_modules}")
            else :
                print(f"skip {name} with {quant_modules}")
    return model

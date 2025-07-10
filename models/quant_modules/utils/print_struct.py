sum_classes=['Conv2d',
             'BinaryConv2dnew',
             'BinaryConv2dneww1a32',
             'BinaryConv2d_reactnet',
             'BinaryConv2d_BiDM',
             'BinaryConv2d_1x1_BiLLM',
             'BinaryConv2d_reactnet_a32w1',
             'BinaryConv2d_ours',
             'ResidualConnectionWith1x1Conv',
             'Linear',
             'BitLinearNew',
             'BinaryLinear_reactnet',
             'BinaryLinear_BiLLM',
             'BinaryLinear_ours',
] 
def print_structure(model, 
                    path='/home/yangkaicheng/Sin-SR-Quant/SinSR-main/struct.txt', 
                    print_params=True, print_param_size=True, print_class_name=True, 
                    sum_classes=sum_classes, 
                    threshold=10*1e6):
    def n_char_in_str(s, c):
        cnt = 0
        for i in s:
            if i == c:
                cnt += 1
        return cnt

    def format_param_size(num_params):
        return f"{num_params / 1e6:.2f}M"

    if sum_classes is None:
        sum_classes = []

    class_param_sums = {class_name: 0 for class_name in sum_classes}
    print(class_param_sums)
    below_threshold_sums = {class_name: 0 for class_name in sum_classes}
    above_threshold_sums = {class_name: 0 for class_name in sum_classes}
    below_threshold_layers = {class_name: [] for class_name in sum_classes}

    with open(path, 'w') as f:
        # 打印模块结构
        for name, module in model.named_modules():
            name: str
            n_point = n_char_in_str(name, '.')
            class_name = module.__class__.__name__ if print_class_name else ''
            if print_param_size:
                total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                param_size_str = format_param_size(total_params)
                f.write('\t' * n_point + name + f" ({class_name}, {param_size_str})\n")
            else:
                f.write('\t' * n_point + name + f" ({class_name})\n")

            # 如果 print_params 为 True，打印参数键
            if print_params:
                for param_name, param in module.named_parameters(recurse=False):
                    param_n_point = n_point + 1  # 参数键比模块多一个缩进
                    param_full_name = name + '.' + param_name if name else param_name
                    if print_param_size:
                        param_size_str = format_param_size(param.numel())
                        f.write('\t' * param_n_point + param_full_name + f" ({param_size_str})\n")
                    else:
                        f.write('\t' * param_n_point + param_full_name + '\n')

            # 累加指定类的参数量
            if class_name in class_param_sums:
                #print(f'{class_name}: {total_params}')
                class_param_sums[class_name] += total_params
                if threshold is not None:
                    if total_params < threshold:
                        below_threshold_sums[class_name] += total_params
                        below_threshold_layers[class_name].append((name, total_params))
                    else:
                        above_threshold_sums[class_name] += total_params

        # 在文件末尾输出指定类的参数量总和
        if sum_classes:
            f.write("\nSummary of specified classes:\n")
            print("\nSummary of specified classes:")
            for class_name, param_sum in class_param_sums.items():
                param_size_str = format_param_size(param_sum)
                f.write(f"{class_name}: {param_size_str}\n")

        # 在文件末尾输出指定类的参数量阈值统计
        if threshold is not None:
            f.write(f"\nSummary of specified classes with threshold {threshold}:\n")
            print(f"\nSummary of specified classes with threshold {threshold}:")
            for class_name in sum_classes:
                below_param_size_str = format_param_size(below_threshold_sums[class_name])
                above_param_size_str = format_param_size(above_threshold_sums[class_name])
                f.write(f"{class_name} below threshold: {below_param_size_str}\n")
                f.write(f"{class_name} above threshold: {above_param_size_str}\n")

            # 输出低于阈值的层的全名和参数量大小
            f.write("\nLayers below threshold:\n")
            for class_name, layers in below_threshold_layers.items():
                if layers:
                    f.write(f"{class_name}:\n")
                    for layer_name, param_size in layers:
                        param_size_str = format_param_size(param_size)
                        f.write(f"  {layer_name} ({param_size_str})\n")


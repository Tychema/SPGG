if __name__ == '__main__':
    import torch

    # 假设 L_num 是您的张量的大小
    L_num = 4  # 替换为实际的张量大小

    # 创建一个大小为 [1, 1, L_num, L_num] 的随机张量，这只是一个例子，您需要使用您自己的张量
    tensor = torch.randn(1, 1, L_num, L_num)
    print(tensor)
    # 计算每个位置的和
    upper_neighbor = tensor[:, :, :-1, 1:] if L_num > 1 else torch.zeros_like(tensor[:, :, 1:-1, 1:-1])
    left_neighbor = tensor[:, :, 1:, :-1] if L_num > 1 else torch.zeros_like(tensor[:, :, 1:-1, 1:-1])
    lower_neighbor = tensor[:, :, 1:, 1:] if L_num > 1 else torch.zeros_like(tensor[:, :, 1:-1, 1:-1])
    right_neighbor = tensor[:, :, 1:, 1:] if L_num > 1 else torch.zeros_like(tensor[:, :, 1:-1, 1:-1])

    # 计算新值，根据位置的不同，对邻居进行不同的权重相加
    new_values = (
            tensor[:, :, 1:, 1:] +  # 中间位置，四个邻居都有
            (tensor[:, :, :-1, 1:] if L_num > 1 else 0) + (tensor[:, :, 1:, :-1] if L_num > 1 else 0) +  # 第一行、第一列，三个邻居
            (tensor[:, :, :-1, :-1] if L_num > 1 else 0) + (
                tensor[:, :, :-1, 1:] if L_num > 1 else 0) +  # 最后一行、第一列，三个邻居
            (tensor[:, :, :-1, :-1] if L_num > 1 else 0) + (
                tensor[:, :, 1:, :-1] if L_num > 1 else 0) +  # 第一行、最后一列，三个邻居
            (tensor[:, :, 1:, 1:] if L_num > 1 else 0) + (tensor[:, :, 1:, :-1] if L_num > 1 else 0) +  # 最后一行、最后一列，三个邻居
            upper_neighbor + left_neighbor + lower_neighbor + right_neighbor  # 四个角，两个邻居
    )

    # 更新原始张量
    tensor[:, :, 1:-1, 1:-1] = new_values
    print(tensor)
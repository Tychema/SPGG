import torch
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 测试更新
    tensor = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float).to(device)
    Q_tensor = torch.tensor(
        [[[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
         [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
         [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]]],
        dtype=torch.float).to(device)
    epsilon = 0.1  # 设置 epsilon 的值
    one_minus_epsilon = 1 - epsilon
    # 根据 tensor 中的索引获取 Q_tensor 中的概率分布
    indices = tensor.long().flatten()
    print(indices)
    Q_probabilities = Q_tensor[torch.arange(len(indices)), indices]
    print(Q_probabilities)
    # 在 Q_probabilities 中选择最大值索引
    # 找到每个概率分布的最大值
    max_values, _ = torch.max(Q_probabilities, dim=1)
    print(max_values)
    max_tensor = torch.where(Q_probabilities == max_values[:, None], torch.tensor(1.0, device=device),
                             torch.tensor(0.0, device=device))
    print(max_tensor)
    # 生成随机向量
    indices = torch.nonzero(max_tensor == 1, as_tuple=False)
    print(indices)
    #x=[indices[:, 0] == i][:, 1].tolist()) for i in range(Q_tensor.shape[0])
    #这是一个列表解析，它遍历张量的每一行，根据行的第一个索引（向量的索引），选出具有相同索引值的所有行，并提取这些行中1的索引。它返回一个包含每个向量中1的索引的列表。
    #random.choice(...): 在每个列表中选择一个随机索引。这个随机选择是针对每个向量中1的索引列表进行的，以便最终得到一个包含每个向量中随机选择的1的索引的张量。
    random_max_indices = torch.tensor(
        [random.choice(indices[indices[:, 0] == i][:, 1].tolist()) for i in range(Q_tensor.shape[0])]).to(device)
    # print(Q_probabilities)
    # 生成一个随机的0、1、2的值
    random_type = torch.randint(0, 3, (3, 3)).to(device)
    # 生成一个符合 epsilon 概率的随机 mask
    mask = (torch.rand(3, 3) > epsilon).long().to(device)
    # 使用 mask 来决定更新的值
    updated_values = mask.flatten().unsqueeze(1) * random_max_indices.unsqueeze(1) + (
                1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)
    # print(updated_values)
    # 重新组织更新后的 tensor
    updated_tensor = updated_values.view(3, 3)

import torch

# 假设L_num已经给定，这里我们使用一个示例
L_num = 3
# 假设Q_tensor和type_t_matrix也已经给定，这里我们随机生成作为示例
Q_tensor = torch.randn(L_num*L_num, 2, 2)
type_t_matrix = torch.randint(0, 2, (1, L_num*L_num))
print("Q_tensor:")
print(Q_tensor)
print("type_t_matrix:")
print(type_t_matrix)


# 由于type_t_matrix的形状是(1, L_num*L_num)，我们需要稍微调整索引的获取方法
indices = torch.where(type_t_matrix.squeeze() == 1)[0]
print("type_t_matrix.squeeze():")
print(type_t_matrix.squeeze())
print("indices:")
print(indices)

# 使用这些索引从Q_tensor中选取对应的(2,2)子tensor
selected_tensors = Q_tensor[indices]

# 打印结果
print("Selected Tensors:")
print(selected_tensors)

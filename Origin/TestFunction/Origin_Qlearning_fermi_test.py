import torch

# 假设你有一个名为profit_tensor的张量
profit_tensor = torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10]], dtype=torch.float)

# 生成随机张量，与profit_tensor具有相同的形状
random_tensor = torch.rand_like(profit_tensor)

# 将随机张量的每个元素缩放到对应位置profit_tensor的范围内
random_tensor *= profit_tensor

print((profit_tensor>random_tensor)*(torch.zeros(profit_tensor.shape)<random_tensor))
print(profit_tensor)
print(random_tensor)

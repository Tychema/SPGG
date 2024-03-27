import torch
if __name__ == '__main__':
    # 假设你有一个包含向量的二维tensor
    # 这是一个例子
    tensor = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 1]])

    # 生成随机数
    rand_tensor = torch.rand(tensor.size())
    # 将原始tensor中的值为0的位置设为一个较大的负数，以便在后续选取最大值时不考虑这些位置
    masked_tensor = tensor.float() - (1 - tensor.float()) * 1e9
    # 将随机数加到masked_tensor上，使得原始tensor中的1值所在的位置在新的tensor中值最大
    sum_tensor = masked_tensor + rand_tensor
    # 找到每个向量中值为1的位置的索引
    indices = torch.argmax(sum_tensor, dim=1)

    # 生成一个与tensor相同大小的全零tensor，并将对应位置设置为1
    result = torch.zeros_like(tensor)
    result.scatter_(1, indices.unsqueeze(1), 1)
    indices = torch.nonzero(result)[:, 1]

    # 生成一维tensor2
    tensor2 = indices + 1  # 索引从0开始，需要加1对齐题目要求
    print(result)
    print(tensor2)
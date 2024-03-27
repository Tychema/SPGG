if __name__ == '__main__':
    import numpy as np
    import torch

    # 示例数据
    data = torch.tensor(np.random.randint(1,4,(3, 3, 3)),dtype=torch.float16).to("cuda")
    print(data)
    # 计算均值
    # 计算均值
    mean_matrix = torch.mean(data, dim=0)


    # 打印结果
    print(mean_matrix)
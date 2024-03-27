import torch

# 假设 L_num 为你的大小
L_num = 5
if __name__ == '__main__':
    group_5 = torch.ones((1, 1, L_num, L_num))
    group_5[:, :, [0, -1], :] = 0  # 第一行和最后一行置为零
    group_5[:, :, :, [0, -1]] = 0  # 第一列和最后一列置为零
    print(group_5)

    zeros_tensor = torch.zeros((1, 1, L_num, L_num))

    group_4=zeros_tensor
    # 将第一行、第一列、最后一行、最后一列全置为1
    group_4[:, :, [0, -1], :] = 1  # 第一行和最后一行置为1
    group_4[:, :, :, [0, -1]] = 1  # 第一列和最后一列置为1

    # 将四个角位置置为0
    group_4[:, :, [0, -1], [0, -1]] = 0
    group_4[:, :, [0, -1], [-1, 0]] = 0

    print(group_4)

    group_3=zeros_tensor
    group_3[:, :, [0, -1], [0, -1]] = 1
    group_3[:, :, [0, -1], [-1, 0]] = 1

    print(group_3)

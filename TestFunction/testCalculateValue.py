import torch
from torch import tensor
import numpy as np

L_num=5
epoches=10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alpha=0.36
g=5
k=4
r = 2.8
gamma=0.8
eta = 0.8
epsilon = 0.02
u=0.5
neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float).to(device).view(1,1, 3, 3)
w_kernel=torch.tensor([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],dtype=torch.float).to(device).view(1,1, 5, 5)
actions = torch.tensor([0, 1, 2],dtype=torch.float).to(device)
L = np.full((L_num, L_num), 0)
value_matrix = torch.tensor(L, dtype=torch.float).to(device)




def calculation_value(W_matrix: tensor, d_matrix, c_matrix, r_matrix):
    # 投入一次池子贡献1
    # value_matrix=(value_matrix-1)*(r_matrix+c_matrix)+value_matrix*d_matrix
    # 卷积每次博弈的合作＋r的人数
    print(c_matrix+2*r_matrix)
    coorperation_num = torch.nn.functional.conv2d((c_matrix + r_matrix).view(1, 1, L_num, L_num), neibor_kernel,
                                                  bias=None, stride=1, padding=1).to(device)
    # 卷积每次博弈的w值
    w_num = torch.nn.functional.conv2d(W_matrix.view(1, 1, L_num, L_num), neibor_kernel, bias=None, stride=1,
                                       padding=1).to(device)
    # c和r最后的-1是最开始要贡献到池里面的1
    c_profit_matrix = ((coorperation_num + 1) / g * r - 1 + u / k * w_num)
    d_profit_matrix = ((coorperation_num) / g * r)
    r_profit_matrix = (
    ((coorperation_num + 1) / g * r - 1 + u / k * w_num - alpha * u / k * W_matrix * coorperation_num))
    zero_row = torch.zeros_like(c_profit_matrix[:, :, 0, :])
    zero_column = torch.zeros_like(c_profit_matrix[:, :, :, 0:1])
    # 上面的profit
    print(c_profit_matrix)
    print(d_profit_matrix)
    print(r_profit_matrix)
    up_c_profit = c_profit_matrix[:, :, :-1, :].to(device)
    up_c_profit = torch.cat((zero_row.unsqueeze(2), up_c_profit), dim=2).to(device)
    up_d_profit = d_profit_matrix[:, :, :-1, :].to(device)
    up_d_profit = torch.cat((zero_row.unsqueeze(2), up_d_profit), dim=2).to(device)
    up_r_profit = r_profit_matrix[:, :, :-1, :].to(device)
    up_r_profit = torch.cat((zero_row.unsqueeze(2), up_r_profit), dim=2).to(device)
    # 下面的profit
    below_c_profit = c_profit_matrix[:, :, 1:, :].to(device)
    below_c_profit = torch.cat((below_c_profit,zero_row.unsqueeze(2)), dim=2).to(device)
    below_d_profit = d_profit_matrix[:, :, 1:, :].to(device)
    below_d_profit = torch.cat((below_d_profit,zero_row.unsqueeze(2)), dim=2).to(device)
    below_r_profit = r_profit_matrix[:, :, 1:, :].to(device)
    below_r_profit = torch.cat((below_r_profit,zero_row.unsqueeze(2)), dim=2).to(device)
    # 左边的profit
    left_c_profit = c_profit_matrix[:, :, :, :-1].to(device)
    left_c_profit = torch.cat((zero_column, left_c_profit), dim=3).to(device)
    left_d_profit = d_profit_matrix[:, :, :, :-1].to(device)
    left_d_profit = torch.cat((zero_column, left_d_profit), dim=3).to(device)
    left_r_profit = r_profit_matrix[:, :, :, :-1].to(device)
    left_r_profit = torch.cat((zero_column, left_r_profit), dim=3).to(device)
    # 右边的profit
    right_c_profit = c_profit_matrix[:, :, :, 1:].to(device)
    right_c_profit = torch.cat((right_c_profit,zero_column), dim=3).to(device)
    right_d_profit = d_profit_matrix[:, :, :, 1:].to(device)
    right_d_profit = torch.cat((right_d_profit,zero_column), dim=3).to(device)
    right_r_profit = r_profit_matrix[:, :, :, 1:].to(device)
    right_r_profit = torch.cat((right_r_profit,zero_column), dim=3).to(device)
    profit_matrix = c_profit_matrix*c_matrix+up_c_profit * c_matrix + below_c_profit * c_matrix + left_c_profit * c_matrix + right_c_profit * c_matrix +d_profit_matrix*d_matrix+up_d_profit * d_matrix + below_d_profit * d_matrix + left_d_profit * d_matrix + right_d_profit * d_matrix +r_profit_matrix*r_matrix+up_r_profit * r_matrix + below_r_profit * r_matrix + left_r_profit * r_matrix + right_r_profit * r_matrix
    # profit_matrix = (c_matrix * ((coorperation_num + 1) / g * r - 1+ u/k*w_num))+ (d_matrix* (coorperation_num) / g * r) +(r_matrix*((coorperation_num + 1) / g * r - 1+ u/k*w_num-alpha*u/k*W_matrix*coorperation_num))
    W_matrix = torch.where(W_matrix > 0, W_matrix - 1, W_matrix)
    return profit_matrix, W_matrix

def type_matrix_to_three_matrix(type_matrix:tensor):
    # 初始化一个新的张量，其中数值为0的值设为1，为1和2的值设为0
    d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
    c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
    r_matrix = torch.where(type_matrix == 2, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
    return d_matrix,c_matrix,r_matrix

def generated_default_type_matrix():
    probabilities = torch.tensor([1 / 3, 1 / 3, 1 / 3])

    # 生成一个随机张量，其数值分别为0，1，2，根据指定的概率分布
    result_tensor = torch.multinomial(probabilities, L_num * L_num, replacement=True)
    result_tensor = result_tensor.view(L_num, L_num)
    return result_tensor.to(device)

if __name__ == '__main__':
    value_matrix=torch.tensor(L, dtype=torch.float).to(device)
    W_matrix=torch.tensor(L, dtype=torch.float).to(device)
    type_matrix = generated_default_type_matrix()
    d_matrix, c_matrix, r_matrix= type_matrix_to_three_matrix(type_matrix)
    profit_matrix, W_matrix=calculation_value(W_matrix, d_matrix, c_matrix, r_matrix)

# if __name__ == '__main__':
#     import torch
#     # 假设 input_tensor 是你的大小不固定的 tensor
#     input_tensor = torch.randn(1, 1, 5, 5)  # 示例数据
#     print(input_tensor)
#     # 删除第一行
#     input_tensor = input_tensor[:, :, 1:, :]
#     print(input_tensor)
#     # 插入全为0的一行到最后一行
#     zero_row = torch.zeros_like(input_tensor[:, :, -1:, :])
#     print(zero_row)
#     input_tensor = torch.cat((input_tensor, zero_row), dim=2)
#
#     print(input_tensor)






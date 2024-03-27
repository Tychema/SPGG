import torch
from torch import tensor
def type_matrix_to_three_matrix(type_matrix: tensor):
    # 初始化一个新的张量，其中数值为0的值设为1，为1和2的值设为0
    d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(device)
    c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(device)
    l_matrix = torch.where(type_matrix == 2, torch.tensor(1), torch.tensor(0)).to(device)
    return d_matrix, c_matrix, l_matrix

if __name__ == '__main__':
    L_num = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha = 0.8
    r = 6
    gamma = 0.8
    eta = 0.6
    epsilon = 0.02
    neibor_kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float16).to(device).view(1, 1, 3, 3)
    type_t_matrix = torch.tensor([[0, 1, 0],
                                  [1, 2, 1],
                                  [0, 1, 0]]).to(device)
    d_matrix, c_matrix, l_matrix = type_matrix_to_three_matrix(type_t_matrix)
    coorperation_num=torch.tensor([[2, 1, 2], [1, 4, 1], [2, 1, 2]]).to(device)
    g_matrix = torch.nn.functional.conv2d(torch.ones((1, 1, L_num, L_num), dtype=torch.float16).to(device),neibor_kernel,bias=None, stride=1, padding=1).to(device)
    c_profit_matrix = (coorperation_num) / g_matrix * r - 1

    d_profit_matrix = (coorperation_num) / g_matrix * r
    l_profit_matrix = eta * (r - 1)
    # 这里的k不是固定值，周围的player的k可能会有4顶点为3.
    profit_matrix = c_profit_matrix * c_matrix + d_profit_matrix * d_matrix + l_profit_matrix * l_matrix
    print(profit_matrix)
import torch
from torch import tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L_num = 3


# 矩阵向上下左右移动1位，return 4个矩阵
def indices_Matrix_to_Four_Matrix(indices):
    indices_left = torch.roll(indices, 1, 1)
    indices_right = torch.roll(indices, -1, 1)
    indices_up = torch.roll(indices, 1, 0)
    indices_down = torch.roll(indices, -1, 0)
    return indices_left, indices_right, indices_up, indices_down

def updateQMatrix( type_t_matrix: tensor, Q_tensor: tensor, profit_matrix: tensor):
    # 遍历每一个Qtable
    C_indices = torch.arange(type_t_matrix.numel()).to(device)
    # Qtable中选择的行
    A_indices = type_t_matrix.view(-1).long()
    rand_rate=torch.rand_like(profit_matrix.to(torch.float64))
    exp_profit=torch.exp(10 * profit_matrix)
    Q_tensor[C_indices, A_indices, A_indices] = (rand_rate* exp_profit).view(-1).to(torch.float64)
    t_left, t_right, t_up, t_down = indices_Matrix_to_Four_Matrix(type_t_matrix)
    profit_left, profit_right, profit_up, profit_down = indices_Matrix_to_Four_Matrix(profit_matrix)
    # 生成一个矩阵随机决定向哪个方向学习
    learning_direction = torch.randint(0, 4, (L_num, L_num)).to(device)
    learning_type_t = ((learning_direction == 0) * t_left + \
                       (learning_direction == 1) * t_right + \
                       (learning_direction == 2) * t_up + \
                       (learning_direction == 3) * t_down).view(-1).long()
    learning_profit_t = ((learning_direction == 0) * profit_left + \
                         (learning_direction == 1) * profit_right + \
                         (learning_direction == 2) * profit_up + \
                         (learning_direction == 3) * profit_down).view(-1).long()
    rand_rate2=torch.rand_like(learning_profit_t.to(torch.float64))
    exp_learning_profit=torch.exp(10 * learning_profit_t)
    Q_tensor[C_indices, A_indices, learning_type_t] = ( rand_rate2* exp_learning_profit).view(-1).to(torch.float64)
    return Q_tensor

if __name__ == '__main__':
    type_t_matrix = torch.randint(0, 2, (L_num, L_num)).to(device)
    Q_tensor = torch.zeros((L_num * L_num, 2, 2),dtype=torch.float64).to(device)
    profit_matrix = (torch.rand((L_num, L_num))*20).to(device)
    Q_tensor=updateQMatrix(type_t_matrix, Q_tensor, profit_matrix)
    print(Q_tensor.to(torch.float64))
    print(torch.exp(10*torch.tensor([21],dtype=torch.float64)))
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
L_num = 5
neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float64).to(device).view(1,1,3,3)
r=25/9

# 计算利润
def calculation_value(r, type_t_matrix):
    with torch.no_grad():
        # 投入一次池子贡献1
        # value_matrix=(value_matrix-1)*(l_matrix+c_matrix)+value_matrix*d_matrix
        # 卷积每次博弈的合作＋r的人数
        # 获取原始张量的形状
        # 在第0行之前增加最后一行

        pad_tensor = self.pad_matrix(type_t_matrix)
        d_matrix, c_matrix = self.type_matrix_to_three_matrix(pad_tensor)
        coorperation_matrix = c_matrix.view(1, 1, L_num + 2, L_num + 2).to(torch.float64)
        # 下面这个卷积占了一轮的大部分时间约1秒钟，但是其他卷积都是一瞬间完成的，不知道为什么
        coorperation_num = torch.nn.functional.conv2d(coorperation_matrix, neibor_kernel,
                                                      bias=None, stride=1, padding=0).view(L_num, L_num).to(device)
        # c和r最后的-1是最开始要贡献到池里面的1
        c_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r - 1)

        d_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r)
        c_5_profit_matrix = torch.nn.functional.conv2d(c_profit_matrix.view(1, 1, L_num + 2, L_num + 2), neibor_kernel,
                                                       bias=None, stride=1, padding=0).to(torch.float64).to(device)
        d_5_profit_matrix = torch.nn.functional.conv2d(d_profit_matrix.view(1, 1, L_num + 2, L_num + 2), neibor_kernel,
                                                       bias=None, stride=1, padding=0).to(device)
        d_matrix, c_matrix = self.type_matrix_to_three_matrix(type_t_matrix)

        profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
        return profit_matrix.view(L_num, L_num).to(torch.float64)

# 矩阵向上下左右移动1位，return 4个矩阵
def indices_Matrix_to_Four_Matrix(indices):
    indices_left = torch.roll(indices, 1, 1)
    indices_right = torch.roll(indices, -1, 1)
    indices_up = torch.roll(indices, 1, 0)
    indices_down = torch.roll(indices, -1, 0)
    return indices_left, indices_right, indices_up, indices_down

def profit_Matrix_to_Four_Matrix(profit_matrix, K):
    W_left = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, 1, 1)) / K))
    W_right = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, -1, 1)) / K))
    W_up = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, 1, 0)) / K))
    W_down = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, -1, 0)) / K))
    return W_left, W_right, W_up, W_down


def fermiUpdate(type_t_matrix, type_t1_matrix, Q_tensor):
    # 遍历每一个Qtable
    C_indices = torch.arange(type_t_matrix.numel()).to(device)
    # Qtable中选择的行
    A_indices = type_t_matrix.view(-1).long()
    # Qtable中选择的列
    B_indices = type_t1_matrix.view(-1).long()
    profit_matrix = Q_tensor[C_indices, A_indices, B_indices].view(L_num, L_num)
    # 计算费米更新的概率
    W_left, W_right, W_up, W_down = profit_Matrix_to_Four_Matrix(profit_matrix, 0.5)
    indices_left, indices_right, indices_up, indices_down = indices_Matrix_to_Four_Matrix(type_t1_matrix)
    # 生成一个矩阵随机决定向哪个方向学习
    learning_direction = torch.randint(0, 4, (L_num, L_num)).to(device)

    # 生成一个随机矩阵决定是否向他学习
    learning_probabilities = torch.rand(L_num, L_num).to(device)
    # 费米更新
    left_type_t1_matrix = (learning_direction == 0) * (
                (learning_probabilities <= W_left) * indices_left + (learning_probabilities > W_left) * type_t_matrix)
    right_type_t1_matrix = (learning_direction == 1) * ((learning_probabilities <= W_right) * indices_right + (
                learning_probabilities > W_right) * type_t_matrix)
    up_type_t1_matrix = (learning_direction == 2) * (
                (learning_probabilities <= W_up) * indices_up + (learning_probabilities > W_up) * type_t_matrix)
    down_type_t1_matrix = (learning_direction == 3) * (
                (learning_probabilities <= W_down) * indices_down + (learning_probabilities > W_down) * type_t_matrix)
    type_t1_matrix = left_type_t1_matrix + right_type_t1_matrix + up_type_t1_matrix + down_type_t1_matrix
    return type_t1_matrix.view(L_num, L_num)

if __name__ == '__main__':
    type_t_minus_matrix = torch.tensor([[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1]]).to(device)
    type_t_matrix = torch.tensor([[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0]]).to(device)
    profit_matrix=calculation_value(r, type_t_matrix)


import torch
L_num=3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def indices_Matrix_to_Four_Matrix(indices):
    indices_left = torch.roll(indices, 1, 1)
    indices_right = torch.roll(indices, -1, 1)
    indices_up = torch.roll(indices, 1, 0)
    indices_down = torch.roll(indices, -1, 0)
    return indices_left, indices_right, indices_up, indices_down


def neibor_learning(Q_tensor):
    indices = torch.arange(L_num * L_num).view(L_num, L_num).to(device)
    # 计算费米更新的概率
    indices_left, indices_right, indices_up, indices_down = indices_Matrix_to_Four_Matrix(indices)
    # 生成一个矩阵随机决定向哪个方向学习
    learning_direction = torch.randint(0, 4, (L_num, L_num)).to(device)
    # 生成一个随机矩阵决定是否向他学习
    # learning_probabilities=torch.rand(L_num,L_num).to(device)
    # 生成一个随机矩阵决定向他学习哪一行
    learning_action = torch.randint(0, 2, (L_num, L_num)).to(device)
    # 学习周围人的策略
    update_neibor_learning0 = (learning_direction == 0) * Q_tensor[indices_left, learning_action, 0] + \
                              (learning_direction == 1) * Q_tensor[indices_right, learning_action, 0] + \
                              (learning_direction == 2) * Q_tensor[indices_up, learning_action, 0] + \
                              (learning_direction == 3) * Q_tensor[indices_down, learning_action, 0]
    update_neibor_learning1 = (learning_direction == 0) * Q_tensor[indices_left, learning_action, 1] + \
                              (learning_direction == 1) * Q_tensor[indices_right, learning_action, 1] + \
                              (learning_direction == 2) * Q_tensor[indices_up, learning_action, 1] + \
                              (learning_direction == 3) * Q_tensor[indices_down, learning_action, 1]
    update_Q_tensor=Q_tensor.clone()
    update_Q_tensor[indices, learning_action, 0] = update_neibor_learning0
    update_Q_tensor[indices, learning_action, 1] = update_neibor_learning1
    return update_Q_tensor

if __name__ == '__main__':
    Q_tensor = torch.rand(L_num*L_num, 2, 2).to(device)

    update_Q_tensor=neibor_learning(Q_tensor)
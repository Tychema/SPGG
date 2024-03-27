import torch
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    type_t_matrix = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float).to(device)
    Q_matrix = torch.tensor([[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]]], dtype=torch.float).to(device)
    print("Q_matrix")
    print(Q_matrix)
    start_row = torch.tensor(1)
    start_col = torch.tensor(1)
    L_num = 3

    indices = (start_row * L_num + start_col).item()
    print(indices)
    type = type_t_matrix[start_row, start_col].item()
    print(type)
    Q_probabilities = Q_matrix[indices, int(type),:]
    print(Q_probabilities.size())
    max_value= torch.max(Q_probabilities).item()
    print("max_value")
    print(max_value)
    max_indices = torch.nonzero(torch.eq(Q_probabilities, max_value)).view(-1)
    print("所有最大值的索引:", max_indices)
    # 从最大值索引中随机选择一个
    random_max_index = random.choice(max_indices).item()

    print("随机选择的最大值索引:", random_max_index)
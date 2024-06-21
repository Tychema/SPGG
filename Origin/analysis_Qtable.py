import torch
from analysis_FQ import shot_pic1


def profit_Matrix_to_Four_Matrix(profit_matrix, K):
    W_left = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, 1, 1)) / K))
    W_right = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, -1, 1)) / K))
    W_up = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, 1, 0)) / K))
    W_down = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, -1, 0)) / K))
    return W_left, W_right, W_up, W_down

if __name__ == '__main__':
    type_t_matrix=torch.load('/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/shot_pic/r=2.777/two_type/generated1/type_t_matrix/type_t_matrix_r=2.7777777777777777_epoches=50000_L=200_T=50000_第0次实验数据.txt',map_location={'cuda:0': 'cuda:3'}).to('cpu')
    Qtable=torch.load('/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/shot_pic/r=2.777/two_type/generated1/Qtable/Qtable_r=2.7777777777777777_epoches=50000_L=200_T=49999_第0次实验数据.txt',map_location={'cuda:0': 'cuda:3'}).to('cpu')
    #shot_pic1(type_t_matrix,50000)
    i=186
    j1=0
    sub_matrix1 = type_t_matrix[i:i + 5, j1:j1 + 5]
    indices=torch.arange(type_t_matrix.numel()).reshape(200,200)[i:i + 5, j1:j1 + 5].reshape(5*5).to('cpu')
    #shot_pic1(sub_matrix1,50000)
    print(Qtable[indices])
    W_left, W_right, W_up, W_down=profit_Matrix_to_Four_Matrix(Qtable[torch.arange(type_t_matrix.numel()),type_t_matrix.reshape(40000),type_t_matrix.reshape(40000)].reshape(200,200), 0.5)
    print(Qtable[indices,sub_matrix1.reshape(25),sub_matrix1.reshape(25)].reshape(5,5))
    W_left_sub = W_left[i:i + 5, j1:j1 + 5].reshape(5*5)[[7,8]]
    W_right_sub = W_right[i:i + 5, j1:j1 + 5].reshape(5*5)[[7,8]]
    W_up_sub = W_up[i:i + 5, j1:j1 + 5].reshape(5*5)[[7,8]]
    W_down_sub = W_down[i:i + 5, j1:j1 + 5].reshape(5*5)[[7,8]]
    print(W_left_sub)
    print(W_right_sub)
    print(W_up_sub)
    print(W_down_sub)



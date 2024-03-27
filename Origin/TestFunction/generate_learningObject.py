import numpy as np
import torch


def pad_matrix(type_t_matrix):
    # 将第-1行添加到第0行之前
    tensor_matrix = torch.cat((type_t_matrix[-1:], type_t_matrix), dim=0)

    # 将第-1列添加到第0列之前
    tensor_matrix = torch.cat((tensor_matrix[:, [-1]], tensor_matrix), dim=1)

    # 将第1行添加到最后1行
    tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[1:2]), dim=0)

    # 将第1列添加到最后1列
    tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[:, 1:2]), dim=1)

    return tensor_matrix

def generated_default_type_matrix(L_num):
    probabilities = torch.tensor([1 / 2, 1 / 2])
    # 生成一个随机张量，其数值分别为0，1，2，根据指定的概率分布
    result_tensor = torch.multinomial(probabilities, L_num * L_num, replacement=True)
    result_tensor = result_tensor.view(L_num, L_num)
    return result_tensor.to(torch.float16).to("cpu")


def profit_Matrix_to_Four_Matrix(profit_matrix, K):
    left_matrix = torch.roll(profit_matrix, 1, 1)
    right_matrix = torch.roll(profit_matrix, -1, 1)
    up_matrix = torch.roll(profit_matrix, 1, 0)
    down_matrix = torch.roll(profit_matrix, -1, 0)
    print("left_matrix:")
    print(left_matrix)
    print("right_matrix:")
    print(right_matrix)
    print("up_matrix:")
    print(up_matrix)
    print("down_matrix:")
    print(down_matrix)
    W_left=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,1))/K))
    W_right = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, -1, 1)) / K))
    W_up = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, 1, 0)) / K))
    W_down = 1 / (1 + torch.exp((profit_matrix - torch.roll(profit_matrix, -1, 0)) / K))
    return W_left, W_right, W_up, W_down

def np_plus(D,D_F):
    D = D + D_F
    return D

if __name__ == '__main__':
    # #测试pad_matrix
    # type_t_matrix = torch.randint(0,9,(5,5))
    # pad_matrix1=pad_matrix(type_t_matrix)
    # # print(pad_matrix1)
    # # print(pad_matrix1.shape)
    # #测试生成默认的type_matrix
    # print(generated_default_type_matrix(5))
    #测试profit_Matrix_to_Four_Matrix
    # profit_matrix = (torch.rand((5, 5))*10).to("cuda:0").to(torch.float32)
    # print("profit_matrix:")
    # print(profit_matrix)
    K = 0.1
    np1=np.arange(50)
    np2=np.arange(50)
    np3=np_plus(np1,np2)
    print(np3)




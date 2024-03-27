import numpy as np
import torch
from torch import tensor
import random
from datetime import datetime
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
L=300
alpha=0.36
g=5
k=4
r = 2.8
gamma=0.8
eta = 0.8
epsilon = 0.02
u=0.5
neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float).to(device).view(1,1, 3, 3)
actions = torch.tensor([0, 1, 2],dtype=torch.float).to(device)
w_kernel=torch.tensor([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],dtype=torch.float).to(device).view(1,1, 5, 5)

def calculation_coorperation_num_and_defection_num(matrix:tensor):

    x=torch.nn.functional.conv2d(matrix,neibor_kernel, bias=None, stride=1, padding=1)

def updateQMatrix(Q_t_matrix:tensor,Q_t1_matrix:tensor,state:int,action:int,value_matrix:tensor,i:int,j:int,profit_matrix:tensor):
    group_profit_matrix=torch.nn.functional.conv2d(profit_matrix.view(1, 1, L, L),neibor_kernel, bias=None, stride=1, padding=1)
    Q_t_matrix[i*L+j][state][action]=(1-eta)*Q_t_matrix[i*L+j][state][action]+eta*(group_profit_matrix+gamma*torch.max(Q_t_matrix[action]))
    return Q_t_matrix

def calculation_value(W_matrix:tensor,d_matrix,c_matrix,r_matrix):
    coorperation_num=torch.nn.functional.conv2d((c_matrix+r_matrix).view(1, 1, 3, 3),neibor_kernel, bias=None, stride=1, padding=1).to(device)
    w_num = torch.nn.functional.conv2d(W_matrix.view(1, 1, 3, 3), neibor_kernel, bias=None, stride=1, padding=1).to(device)
    value_matrix = c_matrix * ((coorperation_num + 1) / g * r - 1+ u/k*w_num)+ d_matrix* (coorperation_num + 1) / g * r +r_matrix*((coorperation_num + 1) / g * r - 1+ u/k*w_num-alpha*u/k*W_matrix*coorperation_num)
    return value_matrix


def type_matrix_to_three_matrix(type_matrix:tensor):
    # 初始化一个新的张量，其中数值为0的值设为1，为1和2的值设为0
    d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
    c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
    r_matrix = torch.where(type_matrix == 2, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
    return d_matrix,c_matrix,r_matrix


def generated_default_type_matrix():
    probabilities = torch.tensor([1 / 3, 1 / 3, 1 / 3])

    # 生成一个随机张量，其数值分别为0，1，2，根据指定的概率分布
    result_tensor = torch.multinomial(probabilities, L * L, replacement=True)
    result_tensor = result_tensor.view(L, L)
    return result_tensor

def cal_w(type_t_matrix:tensor,type_t1_matrix:tensor, w_matrix:tensor,r_matrix:tensor):
    new_kernel = w_kernel.view(1,1,5,5)  # 适配卷积操作的维度

    # 构造要更新的条件 mask
    update_condition = ((type_t_matrix != type_t1_matrix) & (type_t1_matrix == 0)).float()

    # 使用卷积计算邻域内是否有1，并且r_matrix对应位置是否存在值
    conv_result = torch.nn.functional.conv2d(input=update_condition.view(1,1,3,3),
                                             weight=new_kernel, padding=2,stride=1)
    conv_result = conv_result.squeeze() * r_matrix

    # 更新 w_matrix
    w_matrix = torch.where((conv_result > 0),
                           w_matrix + conv_result, w_matrix)
    w_matrix = torch.where((type_t1_matrix != 2),
                           torch.zeros_like(w_matrix), w_matrix)

    return w_matrix

def update_change():
    #测试更新
    #tensor = torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float).to(device)
    #Q_tensor = torch.tensor([[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]]],dtype=torch.float).to(device)
    tensor=generated_default_type_matrix()
    print(tensor)
    Q = np.zeros((L * L, 3, 3))
    Q_tensor = torch.tensor(Q, dtype=torch.float).to(device)
    epsilon = 0.1  # 设置 epsilon 的值
    one_minus_epsilon = 1 - epsilon

     # 根据 tensor 中的索引获取 Q_tensor 中的概率分布
    indices = tensor.long().flatten()
    Q_probabilities = Q_tensor[torch.arange(len(indices)), indices]
    # 在 Q_probabilities 中选择最大值索引
    # 找到每个概率分布的最大值
    max_values, _ = torch.max(Q_probabilities, dim=1)

    max_tensor = torch.where(Q_probabilities == max_values[:, None], torch.tensor(1.0, device=device),
                             torch.tensor(0.0, device=device))
    # 生成随机向量
    indices = torch.nonzero(max_tensor == 1, as_tuple=False)

    random_max_indices = torch.tensor(
        [random.choice(indices[indices[:, 0] == i][:, 1].tolist()) for i in range(Q_tensor.shape[0])]).to(device)

    #print(Q_probabilities)
    # 生成一个随机的0、1、2的值
    random_indices = torch.randint(0, 3, (L, L)).to(device)

    # 生成一个符合 epsilon 概率的随机 mask
    mask = (torch.rand(L, L) > epsilon).long().to(device)



    # 使用 mask 来决定更新的值
    updated_values =  mask.flatten().unsqueeze(1)* random_max_indices.unsqueeze(1) + (1 - mask.flatten().unsqueeze(1)) *random_indices.flatten().float().unsqueeze(1)
    # print(updated_values)
    # 重新组织更新后的 tensor
    updated_tensor = updated_values.view(L, L)
    print(updated_tensor)


def type_matrix_change(type_matrix: tensor, Q_matrix: tensor):
    epsilon = 0.1  # 设置 epsilon 的值
    indices = type_matrix.long().flatten()
    print(indices)
    Q_probabilities = Q_matrix[torch.arange(len(indices)), indices]
    print(Q_probabilities)
    max_values, _ = torch.max(Q_probabilities, dim=1)
    print(max_values)
    max_tensor = torch.where(Q_probabilities == max_values[:, None], torch.tensor(1.0, device=device),
                             torch.tensor(0.0, device=device))
    print(max_tensor)

    # 生成随机数
    rand_tensor = torch.rand(max_tensor.size()).to(device)
    # 将原始tensor中的值为0的位置设为一个较大的负数，以便在后续选取最大值时不考虑这些位置
    masked_tensor = (max_tensor.float() - (1 - max_tensor.float()) * 1e9).to(device)
    # 将随机数加到masked_tensor上，使得原始tensor中的1值所在的位置在新的tensor中值最大
    sum_tensor = (masked_tensor + rand_tensor).to(device)
    # 找到每个向量中值为1的位置的索引
    indices = torch.argmax(sum_tensor, dim=1).to(device)

    # 生成一个与tensor相同大小的全零tensor，并将对应位置设置为1
    random_max_indices = torch.zeros_like(max_tensor).to(device)
    random_max_indices.scatter_(1, indices.unsqueeze(1), 1)
    random_max_indices = torch.nonzero(random_max_indices)[:, 1]
    print(random_max_indices)


    # indices = torch.nonzero(max_tensor == 1, as_tuple=False)
    # print(indices)
    # # x=[indices[:, 0] == i][:, 1].tolist()) for i in range(Q_tensor.shape[0])
    # # 这是一个列表解析，它遍历张量的每一行，根据行的第一个索引（向量的索引），选出具有相同索引值的所有行，并提取这些行中1的索引。它返回一个包含每个向量中1的索引的列表。
    # # random.choice(...): 在每个列表中选择一个随机索引。这个随机选择是针对每个向量中1的索引列表进行的，以便最终得到一个包含每个向量中随机选择的1的索引的张量。
    #
    # random_max_indices = torch.tensor(
    #     [random.choice(indices[indices[:, 0] == i][:, 1].tolist()) for i in range(Q_matrix.shape[0])]).to(device)
    # print(random_max_indices)
    # 生成一个随机的0、1、2的值
    random_type = torch.randint(0, 3, (3, 3)).to(device)
    # 生成一个符合 epsilon 概率的随机 mask
    mask = (torch.rand(3, 3) > epsilon).long().to(device)
    # 使用 mask 来决定更新的值
    updated_values = mask.flatten().unsqueeze(1) * random_max_indices.unsqueeze(1) + (
            1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)
    # print(updated_values)
    # 重新组织更新后的 tensor
    updated_tensor = updated_values.view(3, 3)

    return updated_tensor

if __name__ == '__main__':
    type_t_matrix=torch.tensor([[2,1,0],[1,2,0],[2,1,0]],dtype=torch.float).to(device)
    w_matrix=torch.tensor([[0,0,0],[0,0,0],[0,0,0]],dtype=torch.float).to(device)
    Q_tensor = torch.tensor([[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,1],[0,1,0]]], dtype=torch.float).to(device)
    type_t1_matrix=torch.tensor([[1, 0, 1], [2, 1, 1], [1, 0, 2]], dtype=torch.float).to(device)
    #type_t1_matrix = type_matrix_change(type_t_matrix,Q_tensor)
    d_matrix,c_matrix,r_matrix=type_matrix_to_three_matrix(type_t1_matrix)
    type_t1_matrix=type_matrix_change(type_t_matrix,Q_tensor)


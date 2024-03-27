import torch
from numpy import ndarray
from torch import tensor
# from torch_geometric.data import Data
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L_num = 4
epoches = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_w = 100
min_w = 0
T = 5

alpha = 0.36
g = 5
r = 2.8
gamma = 0.8
eta = 0.8
eta2= 0.6
epsilon = 0.02

neibor_kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float).to(device).view(1, 1, 3, 3)
w5_kernel = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=torch.float).to(device).view(1, 1, 9, 9)
w4_kernel = torch.tensor([[ 0, 0, 0, 1, 0, 0, 0],
                          [ 0, 0, 1, 1, 1, 0, 0],
                          [ 0, 1, 1, 1, 1, 1, 0],
                          [ 1, 1, 1, 1, 1, 1, 1],
                          [ 0, 1, 1, 1, 1, 1, 0],
                          [ 0, 0, 1, 1, 1, 0, 0],
                          [ 0, 0, 0, 1, 0, 0, 0]], dtype=torch.float).to(device).view(1, 1, 7, 7)
w3_kernel = torch.tensor([[0, 0, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 0, 0]], dtype=torch.float).to(device).view(1, 1, 5, 5)
actions = torch.tensor([0, 1, 2], dtype=torch.float).to(device)
L = np.full((L_num, L_num), 0)
value_matrix = torch.tensor(L, dtype=torch.float).to(device)

# g存在=5，4，3的三种情况，三种情况分开计算
group_5 = torch.ones((1, 1, L_num, L_num)).to(device)
group_5[:, :, [0, -1], :] = 0  # 第一行和最后一行置为零
group_5[:, :, :, [0, -1]] = 0  # 第一列和最后一列置为零

zeros_tensor = torch.zeros((1, 1, L_num, L_num))
group_4 = zeros_tensor.to(device)
group_4[:, :, [0, -1], :] = 1  # 第一行和最后一行置为1
group_4[:, :, :, [0, -1]] = 1  # 第一列和最后一列置为1
group_4[:, :, [0, -1], [0, -1]] = 0
group_4[:, :, [0, -1], [-1, 0]] = 0

group_3 = zeros_tensor.to(device)
group_3[:, :, [0, -1], [0, -1]] = 1
group_3[:, :, [0, -1], [-1, 0]] = 1


def calculation_coorperation_num_and_defection_num(r_matrix: tensor, c_matrix: tensor):
    with torch.no_grad():
        return torch.nn.functional.conv2d((r_matrix + c_matrix).view(1, 1, L_num, L_num), neibor_kernel, bias=None,
                                          stride=1, padding=1)


def updateQMatrix(type_t_matrix: tensor, type_t1_matrix: tensor, Q_tensor: tensor, profit_matrix: tensor,
                  w_matrix: tensor):
    with torch.no_grad():
        d_matrix, c_matrix=type_matrix_to_three_matrix(type_t1_matrix)
        C_indices = torch.arange(type_t_matrix.numel()).to(device)
        A_indices = type_t_matrix.view(-1).long()
        B_indices = type_t1_matrix.view(-1).long()
        # group_profit_matrix = torch.nn.functional.conv2d(profit_matrix.view(1, 1, L_num, L_num), neibor_kernel, bias=None,stride=1, padding=1)
        # print(Q_tensor[C_indices, A_indices])
        # 计算更新值

        w1_set_matrix = torch.where((0 < w_matrix  and w_matrix < (max_w-min_w)/T),
                                    1,0)
        w2_set_matrix = torch.where(((max_w-min_w)/T < w_matrix and w_matrix < (max_w-min_w)*2/T),
                               1,0)
        w3_set_matrix = torch.where(((max_w-min_w)*2/T < w_matrix and w_matrix < (max_w-min_w)*3/T),
                               1,0)
        w4_set_matrix = torch.where(((max_w-min_w)*3/T < w_matrix and w_matrix < (max_w-min_w)*4/T),
                               1,0)
        w5_set_matrix = torch.where(((max_w-min_w)*4/T < w_matrix and w_matrix < (max_w-min_w)*5/T),
                               1,0)
        c_w5_matrix=torch.nn.functional.conv2d(w5_set_matrix*c_matrix,w5_kernel,bias=None, stride=1, padding=4).to(device)
        c_w4_matrix=torch.nn.functional.conv2d(w4_set_matrix*c_matrix,w4_kernel,bias=None, stride=1, padding=4).to(device)
        c_w3_matrix=torch.nn.functional.conv2d(w3_set_matrix*c_matrix,w3_kernel,bias=None, stride=1, padding=4).to(device)
        c_w2_matrix=torch.nn.functional.conv2d(w2_set_matrix*c_matrix,neibor_kernel,bias=None, stride=1, padding=4).to(device)
        c_w1_matrix=torch.nn.functional.conv2d(w1_set_matrix*c_matrix,neibor_kernel,bias=None, stride=1, padding=4).to(device)
        c_w5_num_matrix=torch.nn.functional.conv2d(w5_set_matrix*c_matrix,w5_kernel,bias=None, stride=1, padding=4).to(device)
        c_w4_num_matrix=torch.nn.functional.conv2d(w4_set_matrix*c_matrix,w4_kernel,bias=None, stride=1, padding=4).to(device)
        c_w3_num_matrix=torch.nn.functional.conv2d(w3_set_matrix*c_matrix,w3_kernel,bias=None, stride=1, padding=4).to(device)
        c_w2_num_matrix=torch.nn.functional.conv2d(w2_set_matrix*c_matrix,neibor_kernel,bias=None, stride=1, padding=4).to(device)
        c_w1_num_matrix=torch.nn.functional.conv2d(w1_set_matrix*c_matrix,neibor_kernel,bias=None, stride=1, padding=4).to(device)
        d_w5_matrix=torch.nn.functional.conv2d(w5_set_matrix*c_matrix,w5_kernel,bias=None, stride=1, padding=4).to(device)
        d_w4_matrix=torch.nn.functional.conv2d(w4_set_matrix*c_matrix,w4_kernel,bias=None, stride=1, padding=4).to(device)
        d_w3_matrix=torch.nn.functional.conv2d(w3_set_matrix*c_matrix,w3_kernel,bias=None, stride=1, padding=4).to(device)
        d_w2_matrix=torch.nn.functional.conv2d(w2_set_matrix*c_matrix,neibor_kernel,bias=None, stride=1, padding=4).to(device)
        d_w1_matrix=torch.nn.functional.conv2d(w1_set_matrix*c_matrix,neibor_kernel,bias=None, stride=1, padding=4).to(device)
        d_w5_num_matrix=torch.nn.functional.conv2d(w5_set_matrix*c_matrix,w5_kernel,bias=None, stride=1, padding=4).to(device)
        d_w4_num_matrix=torch.nn.functional.conv2d(w4_set_matrix*c_matrix,w4_kernel,bias=None, stride=1, padding=4).to(device)
        d_w3_num_matrix=torch.nn.functional.conv2d(w3_set_matrix*c_matrix,w3_kernel,bias=None, stride=1, padding=4).to(device)
        d_w2_num_matrix=torch.nn.functional.conv2d(w2_set_matrix*c_matrix,neibor_kernel,bias=None, stride=1, padding=4).to(device)
        d_w1_num_matrix=torch.nn.functional.conv2d(w1_set_matrix*c_matrix,neibor_kernel,bias=None, stride=1, padding=4).to(device)
        w_matrix=(c_w1_matrix*1+c_w2_matrix*2+c_w3_matrix*3+c_w4_matrix*4+c_w5_matrix*5)/(c_w5_num_matrix+c_w4_num_matrix+c_w3_num_matrix+c_w2_num_matrix+c_w1_num_matrix)+(d_w1_matrix*1+d_w2_matrix*2+d_w3_matrix*3+d_w4_matrix*4+d_w5_matrix*5)/(d_w5_num_matrix+d_w4_num_matrix+d_w3_num_matrix+d_w2_num_matrix+d_w1_num_matrix)
        max_values, _ = torch.max(Q_tensor[C_indices, B_indices], dim=1)
        # update_values = (1 - eta) * Q_tensor[C_indices, A_indices, B_indices] + eta * (
        #             group_profit_matrix.view(-1) + gamma * max_values)
        #更新公式
        update_values = (1 - eta) * Q_tensor[C_indices, A_indices, B_indices] + eta * (
                (profit_matrix.view(-1) - r/5+5) / (4 * r/2) + w_matrix / (T/2) + gamma * max_values)

        # print(update_values)
        # 更新 type_t_matrix
        Q_tensor[C_indices, A_indices, B_indices] = update_values
    return Q_tensor


def calculation_value(d_matrix, c_matrix):
    with torch.no_grad():
        # 卷积每次博弈的合作＋r的人数

        # 下面这个卷积占了一轮的大部分时间约1秒钟，但是其他卷积都是一瞬间完成的，不知道为什么
        coorperation_num = torch.nn.functional.conv2d((c_matrix).view(1, 1, L_num, L_num), neibor_kernel,
                                                      bias=None, stride=1, padding=1).to(device)
        c_profit_matrix = ((coorperation_num) / 5 * r - 1) * group_5 + ((coorperation_num) / 4 * r - 1) * group_4 + (
                    (coorperation_num) / 3 * r - 1) * group_3

        d_profit_matrix = ((coorperation_num) / 5 * r) * group_5 + ((coorperation_num) / 4 * r) * group_4 + (
                    (coorperation_num) / 3 * r) * group_3

        c_5_profit_matrix = torch.nn.functional.conv2d(c_profit_matrix.view(1, 1, L_num, L_num), neibor_kernel,
                                                       bias=None, stride=1, padding=1).to(device)
        d_5_profit_matrix = torch.nn.functional.conv2d(d_profit_matrix.view(1, 1, L_num, L_num), neibor_kernel,
                                                       bias=None, stride=1, padding=1).to(device)
        profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
        return profit_matrix


def cal_w(w_matrix: tensor, c_matrix: tensor, d_matrix: tensor):
    with torch.no_grad():
        w_matrix = (w_matrix + 1) * c_matrix + (w_matrix - 1) * d_matrix
        w_matrix = torch.where((w_matrix > 100),
                               torch.tensor(100.0, dtype=torch.float32), w_matrix)
        w_matrix = torch.where((w_matrix < 0),
                               torch.tensor(0.0, dtype=torch.float32), w_matrix)

        return w_matrix


# 一轮博弈只后策略的改变
def type_matrix_change(type_matrix: tensor, Q_matrix: tensor):
    with torch.no_grad():
        # 根据 type_matrix 中的索引获取 Q_tensor 中的概率分布
        indices = type_matrix.long().flatten()
        Q_probabilities = Q_matrix[torch.arange(len(indices)), indices]
        # 在 Q_probabilities 中选择最大值索引
        # 找到每个概率分布的最大值
        max_values, _ = torch.max(Q_probabilities, dim=1)
        max_tensor = torch.where(Q_probabilities == max_values[:, None], torch.tensor(1.0, device=device),
                                 torch.tensor(0.0, device=device))
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
        random_type = torch.randint(0, 3, (L_num, L_num)).to(device)
        # 生成一个符合 epsilon 概率的随机 mask
        mask = (torch.rand(L_num, L_num) > epsilon).long().to(device)
        # 使用 mask 来决定更新的值
        updated_values = mask.flatten().unsqueeze(1) * random_max_indices.unsqueeze(1) + (
                1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)
        # print(updated_values)
        # 重新组织更新后的 tensor
        updated_tensor = updated_values.view(L_num, L_num).to(device)
        return updated_tensor


def type_matrix_to_three_matrix(type_matrix: tensor):
    with torch.no_grad():
        # 初始化一个新的张量，其中数值为0的值设为1，为1和2的值设为0
        d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
        c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
        return d_matrix, c_matrix


def generated_default_type_matrix():
    with torch.no_grad():
        probabilities = torch.tensor([1 / 3, 1 / 3])

        # 生成一个随机张量，其数值分别为0，1，2，根据指定的概率分布
        result_tensor = torch.multinomial(probabilities, L_num * L_num, replacement=True)
        result_tensor = result_tensor.view(L_num, L_num)
        return result_tensor.to(device)


if __name__ == '__main__':
    type_matrix = generated_default_type_matrix()

# if __name__ == '__main__':
#     with torch.no_grad():
#         # node= np.full((L_num,1),1)
#         current_time = datetime.now()
#         milliseconds = current_time.microsecond // 1000
#
#         print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
#         #type_matrix=torch.tensor(node,dtype=torch.int).to(device)
#         type_t_matrix =generated_default_type_matrix()
#
#         value_matrix=torch.tensor(L, dtype=torch.float).to(device)
#         W_matrix=torch.tensor(L, dtype=torch.float).to(device)
#         Q=np.zeros((L_num*L_num,2,2))
#         Q_matrix =torch.tensor(Q, dtype=torch.float).to(device)
#
#         # 初始化图表和数据
#         fig = plt.figure()
#         ax=fig.add_subplot(1,1,1)
#
#         obsX = np.array([])
#         D_Y = np.array([])
#         C_Y = np.array([])
#         R_Y = np.array([])
#         plt.xlabel('迭代次数')
#         plt.ylabel('计数')
#
#
#         for i in tqdm(range(epoches), desc='Processing'):
#             type_file_name = f'type\\type_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
#             W_file_name = f'W\\W_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
#             Q_file_name = f'Q\\Q_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
#             V_file_name = f'V\\V_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
#             print("第",i,"次博弈")
#             print("type_t_matrix:")
#             print(type_t_matrix)
#             #把一个L的三个type分开
#             d_matrix, c_matrix, r_matrix = type_matrix_to_three_matrix(type_t_matrix)
#             #计算此次博弈利润的结果
#             profit_matrix,W_matrix=calculation_value(W_matrix,d_matrix,c_matrix,r_matrix)
#             print("profit_matrix:")
#             print(profit_matrix)
#             print("W_matrix:")
#             print(W_matrix)
#             # 计算得到的价值
#             value_matrix=value_matrix+profit_matrix
#             #博弈演化,type变换，策略传播
#             type_t1_matrix=type_matrix_change(type_t_matrix,Q_matrix).to(device)
#             print("type_t1_matrix:")
#             print(type_t1_matrix)
#             #把一个L的三个type分开，为了方便计算，后面都要用到分开的三个type
#             d_matrix, c_matrix, r_matrix = type_matrix_to_three_matrix(type_t1_matrix)
#             #计算w
#             W_matrix=cal_w(type_t_matrix,type_t1_matrix,W_matrix,r_matrix)
#             print("W_matrix:")
#             print(W_matrix)
#             #Q策略更新
#             Q_matrix=updateQMatrix(type_t_matrix,type_t1_matrix,Q_matrix,profit_matrix)
#             print("Q_matrix:")
#             print(Q_matrix)
#             type_t_matrix=type_t1_matrix
#             count_0 = torch.sum(type_t_matrix == 0).item()
#             count_1 = torch.sum(type_t_matrix == 1).item()
#             count_2 = torch.sum(type_t_matrix == 2).item()
#             obsX=np.append(obsX,i+1)
#             D_Y=np.append(D_Y,count_0)
#             C_Y=np.append(C_Y,count_1)
#             R_Y=np.append(R_Y,count_2)
#             # 清除之前的图表
#             plt.clf()
#
#         plt.plot(obsX, D_Y, 'ro', label='betray')
#         plt.pause(0.001)  # 暂停一小段时间以更新图表
#         plt.plot(obsX, C_Y, 'go', label='cooperation')
#         plt.pause(0.001)  # 暂停一小段时间以更新图表
#         plt.plot(obsX, R_Y, 'yo', label='redistribution')
#         plt.pause(0.001)  # 暂停一小段时间以更新图表
#         current_time = datetime.now()
#         milliseconds = current_time.microsecond // 1000
#
#         print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")

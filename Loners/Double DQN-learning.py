import torch
from numpy import ndarray
from torch import tensor
#from torch_geometric.data import Data
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

epoches=100000
L_num=50
torch.cuda.set_device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alpha=0.8
r = 6
gamma=0.8
delta =0.6
eta = 0.8
eta2= 1
epsilon=0.02
step=20
neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float16).to(device).view(1,1, 3, 3)
actions = torch.tensor([0, 1, 2],dtype=torch.float16).to(device)
L = np.full((L_num, L_num), 0)
value_matrix = torch.tensor(L, dtype=torch.float32).to(device)

zeros_tensor = torch.zeros((1, 1, L_num, L_num),dtype=torch.float16).to(torch.float16)
g_matrix=torch.nn.functional.conv2d(torch.ones((1,1,L_num, L_num),dtype=torch.float16).to(device), neibor_kernel,
                                                      bias=None, stride=1, padding=1).to(device)
l_profit_matrix = torch.full((L_num, L_num),delta*(r-1),dtype=torch.float16).to(device)




def cal_value_bias_mean(profit_matrix: tensor, type_t_matrix: tensor, type_t1_matrix: tensor):
    C_indices = torch.arange(type_t_matrix.numel()).to(device)
    A_indices = type_t_matrix.view(-1).long()
    B_indices = type_t1_matrix.view(-1).long()
    #profit_bias = Q_table[C_indices, A_indices, B_indices].view(L_num, L_num) * 5 - profit_matrix


def updateQMatrix(type_t_matrix: tensor, type_t1_matrix: tensor, Q_table: tensor, Q2_table: tensor,profit_matrix: tensor):
    C_indices = torch.arange(type_t_matrix.numel()).to(device)
    A_indices = type_t_matrix.view(-1).long()
    B_indices = type_t1_matrix.view(-1).long()
    group_profit_matrix = torch.nn.functional.conv2d(profit_matrix.view(1, 1, L_num, L_num), neibor_kernel, bias=None,
                                                     stride=1, padding=1).to(device)
    # print(Q_tensor[C_indices, A_indices])
    # 计算更新值
    Q_probabilities = Q_table[C_indices, B_indices]
    max_values, _ = torch.max(Q_table[C_indices, B_indices], dim=1)
    max_tensor = torch.where(Q_probabilities == max_values[:, None], torch.tensor(1.0, device=device),
                             torch.tensor(0.0, device=device))
    # 生成随机数
    rand_tensor = torch.rand(max_tensor.size()).to(device)
    # 将原始tensor中的值为0的位置设为一个较大的负数，以便在后续选取最大值时不考虑这些位置
    masked_tensor = (max_tensor.float() - (1 - max_tensor.float()) * 1e9).to(device)
    # 将随机数加到masked_tensor上，使得原始tensor中的1值所在的位置在新的tensor中值最大
    sum_tensor = (masked_tensor + rand_tensor).to(device)
    # 找到每个向量中值为1的位置的索引
    max_values_indices = torch.argmax(sum_tensor, dim=1).to(device)

    # max_values, _ = torch.max(Q_tensor[C_indices, B_indices], dim=1)
    #     update_values = (1 - eta) * Q_tensor[C_indices, A_indices, B_indices] + eta * (
    #                 group_profit_matrix.view(-1) + gamma * max_values)
    # update_Q_values = (1 - eta) * Q_table[C_indices, A_indices, B_indices] + eta * (profit_matrix.view(-1) + gamma * Q2_table[C_indices, B_indices, max_values_indices])
    update_Q_values = Q_table[C_indices, A_indices, B_indices] + eta * (
                profit_matrix.view(-1) + 0.8 * Q2_table[C_indices, B_indices, max_values_indices] - Q_table[
            C_indices, A_indices, B_indices])
    # print(update_values)
    # 更新 type_t_matrix
    Q_table[C_indices, A_indices, B_indices] = update_Q_values
    return Q_table


# def updateQ2Matrix(type_t_matrix: tensor, type_t1_matrix: tensor, Q2_table: tensor, profit_matrix: tensor):
#     C_indices = torch.arange(type_t_matrix.numel()).to(device)
#     A_indices = type_t_matrix.view(-1).long()
#     B_indices = type_t1_matrix.view(-1).long()
#     update_values = Q2_table[C_indices, A_indices, B_indices] + eta2 * (
#                 profit_matrix.view(-1) - Q2_table[C_indices, A_indices, B_indices])
#     Q2_table[C_indices, A_indices, B_indices] = update_values
#     return Q2_table
#
def updateQ2Matrix(Q_table: tensor, Q2_table: tensor,i):
    if i%step==0:
        Q2_table=Q_table
    return Q2_table


def calculation_value(d_matrix, c_matrix, l_matrix):
    with torch.no_grad():
        # 投入一次池子贡献1
        # value_matrix=(value_matrix-1)*(l_matrix+c_matrix)+value_matrix*d_matrix
        # 卷积每次博弈的合作＋r的人数
        coorperation_matrix = c_matrix .view(1, 1, L_num, L_num).to(torch.float16)

        # 下面这个卷积占了一轮的大部分时间约1秒钟，但是其他卷积都是一瞬间完成的，不知道为什么
        coorperation_num = torch.nn.functional.conv2d(coorperation_matrix, neibor_kernel,
                                                      bias=None, stride=1, padding=1).to(device)
        # c和r最后的-1是最开始要贡献到池里面的1
        c_profit_matrix = (coorperation_num) / g_matrix * r - 1

        d_profit_matrix = (coorperation_num) / g_matrix * r

        c_profit_matrix = c_profit_matrix - c_profit_matrix * l_matrix

        d_profit_matrix = d_profit_matrix - d_profit_matrix * l_matrix
        l2_profit_matrix = l_profit_matrix - l_profit_matrix * l_matrix

        c_5_profit_matrix = torch.nn.functional.conv2d(c_profit_matrix.view(1, 1, L_num, L_num), neibor_kernel,
                                                       bias=None, stride=1, padding=1).to(torch.float16).to(device)
        d_5_profit_matrix = torch.nn.functional.conv2d(d_profit_matrix.view(1, 1, L_num, L_num), neibor_kernel,
                                                       bias=None, stride=1, padding=1).to(device)
        l_5_profit_matrix = torch.nn.functional.conv2d(l2_profit_matrix.view(1, 1, L_num, L_num), neibor_kernel,
                                                       bias=None, stride=1, padding=1).to(device)
        # 这里的k不是固定值，周围的player的k可能会有4顶点为3.
        profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix + l_5_profit_matrix * l_matrix
        return profit_matrix


# #一轮博弈只后策略的改变
def type_matrix_change(type_matrix: tensor, Q_table: tensor):
    indices = type_matrix.long().flatten()
    Q_probabilities = Q_table[torch.arange(len(indices)), indices]
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

    # 生成一个随机的0、1、2的值
    random_type = torch.randint(0, 3, (L_num, L_num)).to(device)
    # 生成一个符合 epsilon 概率的随机 mask
    mask = (torch.rand(L_num, L_num) > epsilon).long().to(device)

    # 使用 mask 来决定更新的值
    updated_values = mask.flatten().unsqueeze(1) * random_max_indices.unsqueeze(1) + (
            1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)

    # 重新组织更新后的 tensor
    updated_tensor = updated_values.view(L_num, L_num).to(device)

    return updated_tensor


def type_matrix_to_three_matrix(type_matrix: tensor):
    # 初始化一个新的张量，其中数值为0的值设为1，为1和2的值设为0
    d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(device)
    c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(device)
    l_matrix = torch.where(type_matrix == 2, torch.tensor(1), torch.tensor(0)).to(device)
    return d_matrix, c_matrix, l_matrix


def generated_default_type_matrix():
    probabilities = torch.tensor([1 / 3, 1 / 3, 1 / 3])

    # 生成一个随机张量，其数值分别为0，1，2，根据指定的概率分布
    result_tensor = torch.multinomial(probabilities, L_num * L_num, replacement=True)
    result_tensor = result_tensor.view(L_num, L_num)
    return result_tensor.to(torch.float16).to("cpu")


def c_mean_v(value_tensor):
    positive_values = value_tensor[value_tensor > 0.0]
    # 计算大于零的值的平均值
    mean_of_positive = torch.mean(positive_values)
    return mean_of_positive.item() + 1


def c_mean_v2(value_tensor):
    # 创建布尔张量，表示大于零的元素
    positive_num = (value_tensor > 0).to(device)
    negetive_num = (value_tensor < 0).to(device)
    # 计算大于零的元素的均值
    mean_of_positive_elements = (value_tensor.to(torch.float32).sum()) / (positive_num + negetive_num).sum()
    return mean_of_positive_elements.to("cpu")


if __name__ == '__main__':
    # node= np.full((L_num,1),1)
    current_time = datetime.now()
    milliseconds = current_time.microsecond // 1000

    print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
    # type_matrix=torch.tensor(node,dtype=torch.int).to(device)
    type_t_matrix = generated_default_type_matrix().to(device)
    type_t_minus_matrix = torch.zeros((L_num, L_num), dtype=torch.float16).to(device)
    value_matrix = torch.tensor(L, dtype=torch.float16).to(device)
    Q = np.zeros((L_num * L_num, 3, 3))
    Q_table = torch.tensor(Q, dtype=torch.float16).to(device)
    Q2_table = torch.tensor(Q, dtype=torch.float16).to(device)

    # 初始化图表和数据
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    obsX = np.array([])
    D_Y = np.array([])
    C_Y = np.array([])
    L_Y = np.array([])
    D_Value = np.array([])
    C_Value = np.array([])
    L_Value = np.array([])
    D_Profit = np.array([])
    C_Profit = np.array([])
    L_Profit = np.array([])
    plt.xlabel('迭代次数')
    plt.ylabel('计数')

    for i in tqdm(range(epoches), desc='Processing'):
        type_file_name = f'type\\type_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
        W_file_name = f'W\\W_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
        Q_file_name = f'Q\\Q_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
        V_file_name = f'V\\V_{i + 1}.pt'  # 这里使用了 i+1 作为文件名

        # 把一个L的三个type分开
        d_matrix, c_matrix, l_matrix = type_matrix_to_three_matrix(type_t_matrix)
        # 计算此次博弈利润的结果
        profit_matrix = calculation_value(d_matrix, c_matrix, l_matrix)
        # 计算得到的价值
        value_matrix = value_matrix + profit_matrix
        if i != 0:
            # Q策略更新
            Q_table = updateQMatrix(type_t_minus_matrix, type_t_matrix, Q_table,Q2_table, profit_matrix)
            Q2_table=updateQ2Matrix(Q_table, Q2_table,i)
        # 博弈演化,type变换，策略传播
        type_t1_matrix = type_matrix_change(type_t_matrix, Q_table).to(device)
        # 把一个L的三个type分开
        d_matrix, c_matrix, l_matrix = type_matrix_to_three_matrix(type_t1_matrix)
        type_t_minus_matrix = type_t_matrix
        type_t_matrix = type_t1_matrix
        d_value = d_matrix * profit_matrix
        c_value = c_matrix * profit_matrix
        l_value = l_matrix * profit_matrix
        dmean_of_positive = c_mean_v2(d_value)
        cmean_of_positive = c_mean_v2(c_value)
        rmean_of_positive = c_mean_v2(l_value)
        count_0 = torch.sum(type_t_matrix == 0).item()
        count_1 = torch.sum(type_t_matrix == 1).item()
        count_2 = torch.sum(type_t_matrix == 2).item()
        obsX = np.append(obsX, i + 1)
        D_Y = np.append(D_Y, count_0 / (L_num * L_num))
        C_Y = np.append(C_Y, count_1 / (L_num * L_num))
        L_Y = np.append(L_Y, count_2 / (L_num * L_num))
        D_Value = np.append(D_Value, dmean_of_positive)
        C_Value = np.append(C_Value, cmean_of_positive + 0.001)
        L_Value = np.append(L_Value, rmean_of_positive + 0.001)
        if (i + 1) % 10000 == 0:
            if i % 100 == 0:
                epsilon = 0.02
            if i % 5000 == 0:
                epsilon = 0.02
            q_mean_matrix = torch.mean(Q_table, dim=0)
            q2_mean_matrix = torch.mean(Q2_table, dim=0)
            print(q_mean_matrix)
            print(q2_mean_matrix)
            plt.clf()
            # mkdir("/kaggle/working/data")
            plt.plot(obsX, D_Y, 'ro', label='betray', linestyle='-', linewidth=1, markeredgecolor='r', markersize='1',
                     markeredgewidth=1)
            plt.plot(obsX, C_Y, 'bo', label='betray', linestyle='-', linewidth=1, markeredgecolor='b',
                     markersize='1', markeredgewidth=1)
            plt.plot(obsX, L_Y, 'yo', label='loners', linestyle='-', linewidth=1, markeredgecolor='y',
                     markersize='1', markeredgewidth=1)
            plt.xticks([0, 10, 100, 1000, 10000, 100000])
            plt.yticks(
                [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                 0.95, 1])
            plt.ylim([0, 0.8])
            plt.xscale('log')
            # plt.savefig("/kaggle/working/data/betray.png")
            plt.pause(0.001)  # 暂停一小段时间以更新图表

            plt.yticks([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26])
            plt.xticks([0, 100, 1000, 10000, 100000, 100000])
            plt.plot(obsX, D_Value, 'ro', label='betray', linestyle='-', linewidth=1, markeredgecolor='r',
                     markersize='1', markeredgewidth=1)
            plt.plot(obsX, C_Value, 'bo', label='betray', linestyle='-', linewidth=1, markeredgecolor='b',
                     markersize='1', markeredgewidth=1)
            plt.plot(obsX, L_Value, 'yo', label='loners', linestyle='-', linewidth=1, markeredgecolor='y',
                     markersize='1', markeredgewidth=1)
            plt.xscale('log')
            # plt.savefig("/kaggle/working/data/redistribution.png")
            plt.pause(0.001)  # 暂停一小段时间以更新图表
            cmap = plt.get_cmap('Set1', 3)
            # 指定图的大小
            #             plt.figure(figsize=(500, 500))  # 10x10的图
            #             plt.matshow(type_t_matrix.cpu().numpy(), cmap=cmap)
            #             plt.colorbar(ticks=[0, 1, 2], label='Color')
            # 显示图片
            # 定义颜色映射
            color_map = {
                0: (255, 0, 0),  # 蓝色
                1: (0, 0, 255),  # 红色
                2: (0, 255, 0)  # 绿色
            }
            image = np.zeros((L_num, L_num, 3), dtype=np.uint8)
            for label, color in color_map.items():
                image[type_t_matrix.cpu() == label] = color
            plt.imshow(image)
            plt.show()
            print(count_0 / (L_num * L_num))
            print(count_1 / (L_num * L_num))
            print(count_2 / (L_num * L_num))
            print(D_Value[i])
            print(C_Value[i])
            print(L_Value[i])
            #             plt.legend()
            # np.savetxt('/kaggle/working/data/obsX_array.txt', obsX, delimiter='\t')
            # np.savetxt('/kaggle/working/data/D_Y_array.txt', D_Y, delimiter='\t')
            # np.savetxt('/kaggle/working/data/C_Y_array.txt', C_Y, delimiter='\t')
            # np.savetxt('/kaggle/working/data/R_Y_array.txt', R_Y, delimiter='\t')
        # 清除之前的图表
    #         torch.save(type_t_matrix, type_file_name)
    #         torch.save(Q_table, Q_file_name)
    #         torch.save(value_matrix, V_file_name)

    # print(type_t_matrix)
    # print(value_matrix)
    # print(Q_table)
    current_time = datetime.now()
    milliseconds = current_time.microsecond // 1000
    print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
    # D_fig.show()
    # C_fig.show()
    # R_fig.show()
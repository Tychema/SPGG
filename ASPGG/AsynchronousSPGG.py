import torch
from numpy import ndarray
from torch import tensor
#from torch_geometric.data import Data
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation
import os
import cv2

epoches=10000*4000
L_num=200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alpha=0.5
g=5
k=4
r = 2.8
gamma=0.8
eta = 0.8
epsilon=0.02
#epsilon = 0.02
u=0.5
neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float).to(device).view(1,1, 3, 3)
w_kernel=torch.tensor([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],dtype=torch.float).to(device).view(1,1, 5, 5)
actions = torch.tensor([0, 1, 2],dtype=torch.float).to(device)
L = np.full((L_num, L_num), 0)
value_matrix = torch.tensor(L, dtype=torch.float).to(device)

#g存在=5，4，3的三种情况，三种情况分开计算
group_5 = torch.ones((1, 1, L_num, L_num)).to(device)
group_5[:, :, [0, -1], :] = 0  # 第一行和最后一行置为零
group_5[:, :, :, [0, -1]] = 0  # 第一列和最后一列置为零

zeros_tensor = torch.zeros((1, 1, L_num, L_num))
group_4=zeros_tensor.to(device)
group_4[:, :, [0, -1], :] = 1  # 第一行和最后一行置为1
group_4[:, :, :, [0, -1]] = 1  # 第一列和最后一列置为1
group_4[:, :, [0, -1], [0, -1]] = 0
group_4[:, :, [0, -1], [-1, 0]] = 0

group_3=zeros_tensor.to(device)
group_3[:, :, [0, -1], [0, -1]] = 1
group_3[:, :, [0, -1], [-1, 0]] = 1

def random_submatrix(matrix):
    # 在周围扩展一圈，使用4进行填充
    L_num= matrix.shape[0]
    padded_matrix = F.pad(matrix, (1, 1, 1, 1), value=-1)
    start_row = torch.randint(0, L_num - 3 + 1 + 1, (1,))[0]
    start_col = torch.randint(0, L_num - 3 + 1 + 1, (1,))[0]
    submatrix = padded_matrix[start_row:start_row + 3, start_col:start_col + 3].clone()
    return submatrix, start_row, start_col


def insert_submatrix(original_matrix, submatrix, start_row, start_col):
    if(submatrix.size(0)==3):
        original_matrix = F.pad(original_matrix, (1, 1, 1, 1), value=-1)
        original_matrix[start_row:start_row + submatrix.size(0), start_col:start_col + submatrix.size(1)] = submatrix
        return original_matrix[1:-1, 1:-1].to(device)
    else:
        original_matrix = F.pad(original_matrix, (2, 2, 2, 2), value=-1)
        original_matrix[start_row:start_row + submatrix.size(0), start_col:start_col + submatrix.size(1)] = submatrix
        return original_matrix[2:-2, 2:-2].to(device)

def calculation_coorperation_num_and_defection_num(r_matrix: tensor, c_matrix: tensor):
    return torch.nn.functional.conv2d((r_matrix + c_matrix).view(1, 1, L_num, L_num), neibor_kernel, bias=None,
                                      stride=1, padding=1)


def updateQMatrix(type_t_minus_matrix: tensor, type_t_matrix: tensor, Q_tensor: tensor, profit_matrix: tensor,start_row,start_col):
    indices = (int)((start_row * L_num + start_col).item())
    A_indices =(int)(type_t_minus_matrix[start_row, start_col].item())
    B_indices = (int)(type_t_matrix[start_row, start_col].item())
    Q_tensor[indices, A_indices, B_indices] = (1 - eta) * Q_tensor[indices, A_indices, B_indices] + eta * (profit_matrix+ gamma * torch.max(Q_tensor[indices, B_indices, :]))
    return Q_tensor



def calculation_value(W_submatrix: tensor, submatrix):
    #d_num=torch.sum(submatrix == 0).item()
    c_num=torch.sum(submatrix == 1).item()
    r_num=torch.sum(submatrix == 2).item()
    G=torch.sum(torch.where(submatrix>=0,torch.tensor(1),torch.tensor(0))).item()
    w_num=torch.sum(W_submatrix).item()
    D_Profit= ((c_num + r_num) / G * r)
    C_Profit= ((c_num + r_num) / G * r - 1 + u / (G-1) * w_num)
    R_Profit=((c_num + r_num) / G * r - 1 + u / (G-1) * w_num) - alpha * u / (G-1) * w_num * (c_num + r_num)
    W_submatrix = torch.where(W_submatrix*neibor_kernel > 0, W_submatrix - 1, W_submatrix).view(3,3)

    d_profit_matrix = torch.where(submatrix==0,D_Profit,0)
    c_profit_matrix = torch.where(submatrix == 1, C_Profit, 0)
    r_profit_matrix = torch.where(submatrix == 2, R_Profit, 0)
    return d_profit_matrix+c_profit_matrix+r_profit_matrix, W_submatrix


def cal_w(type_t_matrix: tensor, type_t1_matrix: tensor, w_submatrix5: tensor, r_submatrix: tensor,start_row,start_col):
    type0=type_t_matrix[start_row, start_col]
    type1=type_t1_matrix[start_row, start_col]
    if type0!=0 and type1==0:
        w_submatrix5 = (r_submatrix*w_kernel).clone()
    if type0==2 and type1!=2:
        w_submatrix5=w_submatrix5.view(5,5)
        w_submatrix5[2,2]=-w_submatrix5[2,2]
    return w_submatrix5.view(5,5)


# #一轮博弈只后策略的改变
def type_matrix_change(type_t_matrix: tensor, Q_matrix: tensor,start_row,start_col):
    change_type=torch.randint(0,3,(1,1)).to(device)
    random_num=torch.rand(1).to(device)
    type_t1_matrix=type_t_matrix.clone()
    if random_num>=epsilon:
        indices = (start_row * L_num + start_col).item()
        type = type_t_matrix[start_row, start_col].item()
        Q_probabilities = Q_matrix[indices, int(type), :]
        max_value = torch.max(Q_probabilities).item()
        max_indices = torch.nonzero(torch.eq(Q_probabilities, max_value)).view(-1)
        change_type = random.choice(max_indices).item()
    type_t1_matrix[start_row, start_col] = change_type
    return type_t1_matrix


def type_matrix_to_three_matrix(type_matrix: tensor):
    # 初始化一个新的张量，其中数值为0的值设为1，为1和2的值设为0
    d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
    c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
    r_matrix = torch.where(type_matrix == 2, torch.tensor(1), torch.tensor(0)).to(torch.float).to(device)
    return d_matrix, c_matrix, r_matrix


def generated_default_type_matrix():
    probabilities = torch.tensor([1 / 3, 1 / 3, 1 / 3])

    # 生成一个随机张量，其数值分别为0，1，2，根据指定的概率分布
    result_tensor = torch.multinomial(probabilities, L_num * L_num, replacement=True)
    result_tensor = result_tensor.view(L_num, L_num)
    return result_tensor.to("cpu")


def c_mean_v(value_tensor):
    positive_values = value_tensor[value_tensor > 0.0]
    # 计算大于零的值的平均值
    mean_of_positive = torch.mean(positive_values)
    return mean_of_positive.item()


def c_mean_v2(value_tensor):
    # positive_values = value_tensor[value_tensor > 0.0]
    # 计算大于零的值的平均值
    mean_of_positive = torch.mean(value_tensor)
    return mean_of_positive.item()


if __name__ == '__main__':
    # node= np.full((L_num,1),1)
    current_time = datetime.now()
    milliseconds = current_time.microsecond // 1000

    print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
    # type_matrix=torch.tensor(node,dtype=torch.int).to(device)
    type_t_matrix = generated_default_type_matrix().to(device)
    type_t_minus_matrix=torch.zeros((L_num, L_num),dtype=torch.float16).to(device)
    value_matrix = torch.tensor(L, dtype=torch.float).to(device)
    W_matrix = torch.tensor(L, dtype=torch.float).to(device)
    Q = np.zeros((L_num * L_num, 3, 3))
    Q_matrix = torch.tensor(Q, dtype=torch.float).to(device)
    zeros_tensor = torch.zeros(L_num, L_num)
    # 初始化图表和数据
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    obsX = np.array([])
    D_Y = np.array([])
    C_Y = np.array([])
    R_Y = np.array([])
    D_Value = np.array([])
    C_Value = np.array([])
    R_Value = np.array([])
    D_Profit = np.array([])
    C_Profit = np.array([])
    R_Profit = np.array([])
    w_i = np.array([])
    plt.xlabel('迭代次数')
    plt.ylabel('计数')

    for i in tqdm(range(epoches), desc='Processing'):
        type_file_name = f'type\\type_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
        W_file_name = f'W\\W_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
        Q_file_name = f'Q\\Q_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
        V_file_name = f'V\\V_{i + 1}.pt'  # 这里使用了 i+1 作为文件名

        #随机选取一个Agent做博弈
        submatrix, start_row, start_col = random_submatrix(type_t_matrix)
        W_padmatrix=F.pad(W_matrix, (1, 1, 1,1), value=0)
        W_submatrix=W_padmatrix[start_row :start_row + 3, start_col:start_col + 3].view(3,3).clone()
        W_padmatrix5=F.pad(W_matrix, (2, 2, 2,2), value=0)
        W_submatrix5=(W_padmatrix5[start_row :start_row + 5, start_col:start_col + 5]*w_kernel).view(5,5).clone()
        # 计算此次博弈利润的结果
        profit_submatrix, W_submatrix = calculation_value(W_submatrix,submatrix)
        profit_matrix=insert_submatrix(zeros_tensor, profit_submatrix, start_row, start_col)
        W_matrix = insert_submatrix(W_matrix, W_submatrix, start_row, start_col)
        #利润＋持有
        value_matrix= value_matrix + profit_matrix
        if i != 0:
            # Q策略更新
            Q_matrix = updateQMatrix(type_t_minus_matrix, type_t_matrix, Q_matrix, profit_submatrix[1,1],start_row,start_col)
        # 博弈演化,type变换，策略传播
        type_t1_matrix = type_matrix_change(type_t_matrix, Q_matrix,start_row,start_col).to(device)
        # 把一个L的三个type分开
        d_matrix, c_matrix, r_matrix = type_matrix_to_three_matrix(type_t1_matrix)
        # 计算w
        r_padmatrix=F.pad(r_matrix, (2, 2, 2,2), value=0)
        r_submatrix=r_padmatrix[start_row :start_row + 5, start_col:start_col + 5].view(5,5).clone()
        W_submatrix2 = cal_w(type_t_matrix, type_t1_matrix, W_submatrix5, r_submatrix,start_row,start_col)
        w_submatrix2 = insert_submatrix(zeros_tensor, W_submatrix2, start_row, start_col)
        W_matrix=W_matrix+w_submatrix2

        type_t_matrix = type_t1_matrix
        d_value = d_matrix * profit_matrix
        c_value = c_matrix * profit_matrix
        r_value = r_matrix * profit_matrix
        dmean_of_positive = c_mean_v2(d_value)
        cmean_of_positive = c_mean_v2(c_value)
        rmean_of_positive = c_mean_v2(r_value)
        wmean_of_positive = c_mean_v2(W_matrix)
        count_0 = torch.sum(type_t_matrix == 0).item()
        count_1 = torch.sum(type_t_matrix == 1).item()
        count_2 = torch.sum(type_t_matrix == 2).item()
        obsX = np.append(obsX, i + 1)
        D_Y = np.append(D_Y, count_0 / (L_num * L_num))
        C_Y = np.append(C_Y, count_1 / (L_num * L_num))
        R_Y = np.append(R_Y, count_2 / (L_num * L_num))
        D_Value = np.append(D_Value, dmean_of_positive)
        C_Value = np.append(C_Value, cmean_of_positive)
        R_Value = np.append(R_Value, rmean_of_positive)
        w_i = np.append(w_i, wmean_of_positive)
        if (i + 1) % 10000 == 0:
            q_mean_matrix = torch.mean(Q_matrix, dim=0)
            print(q_mean_matrix)
            epsilon = 0.02
            plt.clf()
            #mkdir("/kaggle/working/data")
            plt.xticks([0, 10, 100, 1000, 10000])
            plt.yticks(
                [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                 0.95, 1])
            plt.plot(obsX, D_Y, 'bo', label='betray', linestyle='-', linewidth=1, markeredgecolor='b', markersize='1',
                     markeredgewidth=1)
            plt.xscale('log')
            # plt.savefig("/kaggle/working/data/betray.png")
            plt.pause(0.001)  # 暂停一小段时间以更新图表
            plt.xticks([0, 10, 100, 1000, 10000])
            plt.yticks(
                [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                 0.95, 1])
            plt.plot(obsX, C_Y, 'ro', label='cooperation', linestyle='-', linewidth=1, markeredgecolor='r',
                     markersize='1', markeredgewidth=1)
            plt.xscale('log')
            # plt.savefig("/kaggle/working/data/cooperation.png")
            plt.pause(0.1)  # 暂停一小段时间以更新图表
            plt.xticks([0, 10, 100, 1000, 10000])
            plt.yticks(
                [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                 0.95, 1])
            plt.plot(obsX, R_Y, 'go', label='redistribution', linestyle='-', linewidth=1, markeredgecolor='g',
                     markersize='1', markeredgewidth=1)
            plt.xscale('log')
            # plt.savefig("/kaggle/working/data/redistribution.png")
            plt.pause(0.001)  # 暂停一小段时间以更新图表
            plt.yticks([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26])
            plt.xticks([0, 100, 1000, 10000, 100000])
            plt.plot(obsX, D_Value, 'bo', label='redistribution', linestyle='-', linewidth=1, markeredgecolor='b',
                     markersize='1', markeredgewidth=1)
            plt.xscale('log')
            # plt.savefig("/kaggle/working/data/redistribution.png")
            plt.pause(0.001)  # 暂停一小段时间以更新图表
            plt.yticks([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26])
            plt.xticks([0, 100, 1000, 10000, 100000])
            plt.plot(obsX, C_Value, 'ro', label='redistribution', linestyle='-', linewidth=1, markeredgecolor='r',
                     markersize='1', markeredgewidth=1)
            plt.xscale('log')
            # plt.savefig("/kaggle/working/data/redistribution.png")
            plt.pause(0.001)  # 暂停一小段时间以更新图表
            plt.yticks([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26])
            plt.xticks([0, 100, 1000, 5000, 10000, 100000])
            plt.plot(obsX, R_Value, 'go', label='redistribution', linestyle='-', linewidth=1, markeredgecolor='g',
                     markersize='1', markeredgewidth=1)
            plt.xscale('log')
            # plt.savefig("/kaggle/working/data/redistribution.png")
            plt.pause(0.001)  # 暂停一小段时间以更新图表
            plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            plt.xticks([0, 100, 1000, 5000, 10000, 100000])
            plt.plot(obsX, w_i, 'bo', label='redistribution', linestyle='-', linewidth=1, markeredgecolor='b',
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
            print(D_Value[i])
            print(C_Value[i])
            print(R_Value[i])
            print(w_i[i])
            #             plt.legend()
            np.savetxt('/kaggle/working/data/obsX_array.txt', obsX, delimiter='\t')
            np.savetxt('/kaggle/working/data/D_Y_array.txt', D_Y, delimiter='\t')
            np.savetxt('/kaggle/working/data/C_Y_array.txt', C_Y, delimiter='\t')
            np.savetxt('/kaggle/working/data/R_Y_array.txt', R_Y, delimiter='\t')
        # 清除之前的图表
    #         torch.save(type_t_matrix, type_file_name)
    #         torch.save(W_matrix, W_file_name)
    #         torch.save(Q_matrix, Q_file_name)
    #         torch.save(value_matrix, V_file_name)

    # print(type_t_matrix)
    # print(value_matrix)
    # print(W_matrix)
    # print(Q_matrix)
    current_time = datetime.now()
    milliseconds = current_time.microsecond // 1000
    print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
    # D_fig.show()
    # C_fig.show()
    # R_fig.show()

    #
    # def random_submatrix2(matrix):
    #     #probability = random.uniform(0, 1)
    #     probability=0.45
    #     print("probability")
    #     print(probability)
    #     L_num= matrix.shape[0]
    #     # 根据概率值生成相应的数字
    #     if probability < 4 / (L_num * L_num):
    #         random_number = random.choice([0, 1, 2, 3])
    #         #左上角
    #         if  random_number==0:
    #             submatrix = matrix[0:2, 0:2].clone()
    #             return submatrix, 0, 0,0
    #         #右上角
    #         elif random_number==1:
    #             submatrix = matrix[0:2, L_num - 2:L_num].clone()
    #             return submatrix, 0, L_num - 2,0
    #         #左下角
    #         elif random_number==2:
    #             submatrix = matrix[L_num - 2:L_num, 0:2].clone()
    #             return submatrix, L_num - 2, 0,0
    #         #右下角
    #         else:
    #             submatrix = matrix[L_num - 2:L_num, L_num - 2:L_num].clone()
    #             return submatrix, L_num - 2, L_num - 2,0
    #     elif probability < (4 * (L_num - 2)) / (L_num * L_num) and probability >= 4 / (L_num * L_num):
    #         random_number = random.choice([0, 1, 2, 3])
    #         #上
    #         if  random_number==0:
    #             random_col = random.randint(1, L_num - 1)
    #             submatrix = matrix[0:2, random_col-1:random_col+2].clone()
    #             return submatrix, 0, random_col-1,1
    #         #左
    #         elif random_number==1:
    #             random_row = random.randint(1, L_num - 2)
    #             submatrix = matrix[random_row-1:random_row+2, 0:2].clone()
    #             return submatrix, 0, random_row-1,1
    #         #下
    #         elif random_number==2:
    #             random_col = random.randint(1, L_num - 1)
    #             submatrix = matrix[L_num-2:L_num,random_col-1:random_col+2].clone()
    #             return submatrix, L_num-2, random_col-1,1
    #         #右
    #         else:
    #             random_row = random.randint(1, L_num - 1)
    #             submatrix = matrix[random_row-1:random_row+2, L_num-2:L_num].clone()
    #             return submatrix, random_row-1, L_num-2,1
    #     else:
    #         start_row = torch.randint(0, L_num - 3 + 1, (1,))[0]
    #         start_col = torch.randint(0, L_num - 3 + 1, (1,))[0]
    #         submatrix = matrix[start_row:start_row + 3, start_col:start_col +3].clone()
    #         return submatrix, start_row, start_col,2
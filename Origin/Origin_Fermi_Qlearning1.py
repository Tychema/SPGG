#用Fermi实现Qlearning博弈
#Fermi的pi_y和pi_x用Q表的值来代替


import torch

from torch import tensor
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap

L_num=200
torch.cuda.set_device("cuda:3" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
alpha=0.8
gamma=0.8
epsilon=-1
neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float64).to(device).view(1,1,3,3)
actions = torch.tensor([0, 1],dtype=torch.float64).to(device)
L = np.full((L_num, L_num), 0)
value_matrix = torch.tensor(L, dtype=torch.float64).to(device)

zeros_tensor = torch.zeros((1, 1, L_num, L_num),dtype=torch.float64).to(torch.float64)
g_matrix=torch.nn.functional.conv2d(torch.ones((1,1,L_num, L_num),dtype=torch.float64).to(device), neibor_kernel,
                                                      bias=None, stride=1, padding=1).to(device)
xticks=[0, 10, 100, 1000, 10000, 100000]
fra_yticks=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95, 1.00]
profite_yticks=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

class SPGG_Qlearning(nn.Module):
    def __init__(self,L_num,device,alpha,gamma,epsilon,r,epoches,lr=0.2,eta=0.8,count=0,cal_transfer=False):
        super(SPGG_Qlearning, self).__init__()
        self.epoches=epoches
        self.L_num=L_num
        self.device=device
        self.alpha=alpha
        self.r=r
        self.gamma=gamma
        self.epsilon=epsilon
        self.cal_transfer=cal_transfer
        self.lr=lr
        self.eta=eta
        self.count=count

    #矩阵向上下左右移动1位，return 4个矩阵
    def indices_Matrix_to_Four_Matrix(self,indices):
        indices_left=torch.roll(indices,1,1)
        indices_right=torch.roll(indices,-1,1)
        indices_up=torch.roll(indices,1,0)
        indices_down=torch.roll(indices,-1,0)
        return indices_left,indices_right,indices_up,indices_down

    #Qtable更新，一个是自己经验学习，一个是邻居经验学习
    def updateQMatrix(self,alpha,gamma,type_t_matrix: tensor, type_t1_matrix: tensor, Q_tensor: tensor, profit_matrix: tensor):
        #遍历每一个Qtable
        C_indices = torch.arange(type_t_matrix.numel()).to(device)
        #Qtable中选择的行
        A_indices = type_t_matrix.view(-1).long()
        #Qtable中选择的列
        B_indices = type_t1_matrix.view(-1).long()
        # 计算更新值
        max_values, _ = torch.max(Q_tensor[C_indices, B_indices], dim=1)
        #更新公式
        update_values = Q_tensor[C_indices, A_indices, B_indices] + alpha * (profit_matrix.view(-1) + gamma * max_values - Q_tensor[C_indices, A_indices, B_indices])
        #update_values = (1 - self.eta) * Q_tensor[C_indices, A_indices, B_indices] + self.eta * (profit_matrix.view(-1) + gamma * max_values)
        # print(update_values)
        # 更新 type_t_matrix
        #更新Qtable
        Q_tensor[C_indices, A_indices, B_indices] = update_values
        return Q_tensor

    def profit_Matrix_to_Four_Matrix(self,profit_matrix,K):
        W_left=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,1))/K))
        W_right=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,1))/K))
        W_up=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,0))/K))
        W_down=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,0))/K))
        return W_left,W_right,W_up,W_down

    def fermiUpdate(self,type_t_matrix,type_t1_matrix,Q_tensor):
        #遍历每一个Qtable
        C_indices = torch.arange(type_t_matrix.numel()).to(device)
        #Qtable中选择的行
        A_indices = type_t_matrix.view(-1).long()
        #Qtable中选择的列
        B_indices = type_t1_matrix.view(-1).long()
        profit_matrix = Q_tensor[C_indices, A_indices, B_indices].view(L_num, L_num)
        #计算费米更新的概率
        W_left,W_right,W_up,W_down=self.profit_Matrix_to_Four_Matrix(profit_matrix,0.5)
        indices_left, indices_right, indices_up, indices_down = self.indices_Matrix_to_Four_Matrix(type_t1_matrix)
        #生成一个矩阵随机决定向哪个方向学习
        learning_direction=torch.randint(0,4,(L_num,L_num)).to(device)
        #生成一个随机矩阵决定是否向他学习
        learning_probabilities=torch.rand(L_num,L_num).to(device)
        #费米更新
        type_t1_matrix=(learning_direction==0)*((learning_probabilities<=W_left)*indices_left+(learning_probabilities>W_left)*type_t_matrix) +\
                          (learning_direction==1)*((learning_probabilities<=W_right)*indices_right+(learning_probabilities>W_right)*type_t_matrix) +\
                            (learning_direction==2)*((learning_probabilities<=W_up)*indices_up+(learning_probabilities>W_up)*type_t_matrix) +\
                                (learning_direction==3)*((learning_probabilities<=W_down)*indices_down+(learning_probabilities>W_down)*type_t_matrix)
        return type_t1_matrix.view(L_num,L_num)

    #将最后一行和最后一列添加到第一行和第一列之前，将第一行和第一列添加到最后一行和最后一列之后
    #为了将边缘的两行两列相连
    def pad_matrix(self,type_t_matrix):
        # 将第-1行添加到第0行之前
        tensor_matrix = torch.cat((type_t_matrix[-1:], type_t_matrix), dim=0)
        # 将第-1列添加到第0列之前
        tensor_matrix = torch.cat((tensor_matrix[:, [-1]], tensor_matrix), dim=1)
        # 将第1行添加到最后1行
        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[1:2]), dim=0)
        # 将第1列添加到最后1列
        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[:, 1:2]), dim=1)
        return tensor_matrix


    #计算利润
    def calculation_value(self,r,type_t_matrix):
        with torch.no_grad():
            # 投入一次池子贡献1
            # value_matrix=(value_matrix-1)*(l_matrix+c_matrix)+value_matrix*d_matrix
            # 卷积每次博弈的合作＋r的人数
            # 获取原始张量的形状
            # 在第0行之前增加最后一行

            pad_tensor = self.pad_matrix(type_t_matrix)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(pad_tensor)
            coorperation_matrix = c_matrix .view(1, 1, L_num+2, L_num+2).to(torch.float64)
            # 下面这个卷积占了一轮的大部分时间约1秒钟，但是其他卷积都是一瞬间完成的，不知道为什么
            coorperation_num = torch.nn.functional.conv2d(coorperation_matrix, neibor_kernel,
                                                          bias=None, stride=1, padding=0).view(L_num,L_num).to(device)
            # c和r最后的-1是最开始要贡献到池里面的1
            c_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r - 1)

            d_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r)
            c_5_profit_matrix = torch.nn.functional.conv2d(c_profit_matrix.view(1, 1, L_num+2, L_num+2), neibor_kernel,
                                                           bias=None, stride=1, padding=0).to(torch.float64).to(device)
            d_5_profit_matrix = torch.nn.functional.conv2d(d_profit_matrix.view(1, 1, L_num+2, L_num+2), neibor_kernel,
                                                           bias=None, stride=1, padding=0).to(device)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(type_t_matrix)
            # 这里的k不是固定值，周围的player的k可能会有4顶点为3.
            profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
            return profit_matrix.view(L_num, L_num).to(torch.float64)


    # #一轮博弈只后策略的改变
    def type_matrix_change(self,epsilon,type_matrix: tensor, Q_matrix: tensor):
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
        # random_max_indices = torch.zeros_like(max_tensor).to(device)
        # random_max_indices.scatter_(1, indices.unsqueeze(1), 1)
        # random_max_indices = torch.nonzero(random_max_indices)[:, 1]

        # 生成一个随机的0、1、2的值
        random_type = torch.randint(0,2, (L_num, L_num)).to(device)
        # 生成一个符合 epsilon 概率的随机 mask
        mask = (torch.rand(L_num, L_num) >= epsilon).long().to(device)

        # 使用 mask 来决定更新的值
        # updated_values = mask.flatten().unsqueeze(1) * random_max_indices.unsqueeze(1) + (
        #         1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)
        updated_values = mask.flatten().unsqueeze(1) * indices.unsqueeze(1) + (1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)

        # 重新组织更新后的 tensor
        updated_tensor = updated_values.view(L_num, L_num).to(device)
        return updated_tensor

    #CD矩阵分开，之前主要是用于3策略的
    def type_matrix_to_three_matrix(self,type_matrix: tensor):
        # 初始化一个新的张量，其中数值为0的值设为1，为1和2的值设为0
        d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(device)
        c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(device)
        return d_matrix, c_matrix

    #生成初始矩阵
    def generated_default_type_matrix(self):
        probabilities = torch.tensor([1 /2, 1 / 2])

        # 生成一个随机张量，其数值分别为0，1，2，根据指定的概率分布
        result_tensor = torch.multinomial(probabilities, L_num * L_num, replacement=True)
        result_tensor = result_tensor.view(L_num, L_num)
        return result_tensor.to(torch.float64).to("cpu")

    def generated_default_type_matrix2(self):
        tensor = torch.zeros(L_num, L_num)
        # 计算上半部分和下半部分的分界线（中间行）
        mid_row = L_num // 2
        # 将下半部分的元素设置为1
        tensor[mid_row:, :] = 1
        return tensor

    def generated_default_type_matrix3(self):
        tensor = torch.zeros(L_num, L_num)
        # 计算上半部分和下半部分的分界线（中间行）
        return tensor

    #求利润均值_version_one
    def c_mean_v(self,value_tensor):
        positive_values = value_tensor[value_tensor > 0.0]
        # 计算大于零的值的平均值
        mean_of_positive = torch.mean(positive_values)
        return mean_of_positive.item() + 1

    #求利润均值_version_two
    def c_mean_v2(self,value_tensor):
        # 创建布尔张量，表示大于零的元素
        positive_num = (value_tensor > 0).to(device)
        negetive_num = (value_tensor < 0).to(device)
        # 计算大于零的元素的均值
        mean_of_positive_elements = (value_tensor.to(torch.float64).sum()) / ((positive_num + negetive_num).sum())
        return mean_of_positive_elements.to("cpu")

    #画折线图
    def draw_line_pic(self,D_Y,C_Y,xticks,yticks,r,ylim=(0,1),epoches=10000,type="line1",xlable='T',ylabel='fraction'):
        plt.clf()
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(np.arange(D_Y.shape[0]), D_Y, 'ro', label='betray', linestyle='-', linewidth=1, markeredgecolor='r', markersize=1,
                 markeredgewidth=1)
        plt.plot(np.arange(C_Y.shape[0]), C_Y, 'bo', label='cooperation', linestyle='-', linewidth=1, markeredgecolor='b',
                 markersize=1, markeredgewidth=1)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.ylim(ylim)
        if(type=="line1"):
            plt.xscale('log')
        plt.ylabel(ylabel)
        plt.xlabel(xlable)
        plt.title('Q_learning:'+' L='+str(self.L_num)+' r='+str(r)+' T='+str(epoches))
        plt.pause(0.001)
        plt.clf()
        plt.close("all")

    #画转移图
    def draw_transfer_pic(self,  CC_data, DD_data, CD_data, DC_data, xticks, yticks, r, ylim=(0, 1), epoches=10000):
        plt.clf()    # #一轮博弈只后策略的改变
    def type_matrix_change(self,epsilon,type_matrix: tensor, Q_matrix: tensor):
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
        # random_max_indices = torch.zeros_like(max_tensor).to(device)
        # random_max_indices.scatter_(1, indices.unsqueeze(1), 1)
        # random_max_indices = torch.nonzero(random_max_indices)[:, 1]

        # 生成一个随机的0、1、2的值
        random_type = torch.randint(0,2, (L_num, L_num)).to(device)
        # 生成一个符合 epsilon 概率的随机 mask
        mask = (torch.rand(L_num, L_num) >= epsilon).long().to(device)

        # 使用 mask 来决定更新的值
        # updated_values = mask.flatten().unsqueeze(1) * random_max_indices.unsqueeze(1) + (
        #         1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)
        updated_values = mask.flatten().unsqueeze(1) * indices.unsqueeze(1) + (1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)

        # 重新组织更新后的 tensor
        updated_tensor = updated_values.view(L_num, L_num).to(device)
        return updated_tensor
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(np.arange(DD_data.shape[0]), DD_data, color='red',marker='o', label='DD', linestyle='-', linewidth=1, markeredgecolor='red', markersize=1,
                 markeredgewidth=1)
        plt.plot(np.arange(CC_data.shape[0]), CC_data, color='blue',marker='*', label='CC', linestyle='-', linewidth=1, markeredgecolor='blue',
                 markersize=1, markeredgewidth=1)
        plt.plot(np.arange(CD_data.shape[0]), CD_data, color='black',marker='o', label='CD', linestyle='-', linewidth=1, markeredgecolor='black', markersize=1,markeredgewidth=1)
        plt.plot(np.arange(DC_data.shape[0]), DC_data,color='gold',marker='o', label='DC', linestyle='-', linewidth=1, markeredgecolor='gold',
                 markersize=1, markeredgewidth=1)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.ylim(ylim)
        plt.xscale('log')
        plt.ylabel('fraction')
        plt.xlabel('T')
        plt.title('Q_learning:'+'L'+str(self.L_num)+' r='+str(r)+' T='+str(epoches))
        plt.legend()
        plt.pause(0.001)
        plt.clf()
        plt.close("all")

    #画快照
    def shot_pic(self,type_t_matrix: tensor,i,r):

        plt.clf()
        plt.close("all")
        # 初始化图表和数据
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(type_t_matrix.cpu().numpy(), cmap='binary_r', vmin=0, vmax=1, interpolation='None')
        self.mkdir('data/Origin_Fermi_Qlearning1/shot_pic/r={}/two_type/generated1'.format(r))
        plt.savefig('data/Origin_Fermi_Qlearning1/shot_pic/r={}/two_type/generated1/t={}.png'.format(r,i))
        self.mkdir('data/Origin_Fermi_Qlearning1/shot_pic/r={}/two_type/generated1/type_t_matrix'.format(r))
        np.savetxt('data/Origin_Fermi_Qlearning1/shot_pic/r={}/two_type/generated1/type_t_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(r),"type_t_matrix",str(r),str(self.epoches),str(self.L_num),str(i),str(self.count)),type_t_matrix.cpu().numpy())
        #plt.show()
        plt.clf()
        plt.close("all")

    def shot_pic2(self,type_t_matrix: tensor,i,r):
        plt.clf()
        plt.close("all")
        # 初始化图表和数据
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 4)
        # 指定图的大小
        #             plt.figure(figsize=(500, 500))  # 10x10的图
        #             plt.matshow(type_t_matrix.cpu().numpy(), cmap=cmap)
        #             plt.colorbar(ticks=[0, 1, 2], label='Color')
        # 显示图片
        # 定义颜色映射
        color_map = {
            #0设置为灰色
            0:(128, 128, 128),  # 灰色 DD
            #1设置为白色
            1:(255, 255, 255),  # 白色 CC
            #2设置为黑色
            2:(0, 0, 0),  # 黑色 CDC
            #3设置为深蓝色
            3:(31,119,180) # 深蓝色 StickStrategy
        }
        image = np.zeros((L_num, L_num, 3), dtype=np.uint8)
        for label, color in color_map.items():
            image[type_t_matrix.cpu() == label] = color
        plt.title('Qlearning: '+f"T:{i}")
        plt.imshow(image,interpolation='None')
        self.mkdir('data/Origin_Fermi_Qlearning1/shot_pic/r={}/four_type/generated1'.format(r))
        plt.savefig('data/Origin_Fermi_Qlearning1/shot_pic/r={}/four_type/generated1/t={}.png'.format(r,i))
        self.mkdir('data/Origin_Fermi_Qlearning1/shot_pic/r={}/four_type/generated1/type_t_matrix'.format(r))
        np.savetxt('data/Origin_Fermi_Qlearning1/shot_pic/r={}/four_type/generated1/type_t_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(r), "type_t_matrix", str(r), str(self.epoches), str(self.L_num),str(i), str(self.count)), type_t_matrix.cpu().numpy())

        #plt.show()
        plt.clf()
        plt.close("all")

    #计算CD比例和利润
    def cal_fra_and_value(self, D_Y, C_Y, D_Value, C_Value,all_value, type_t_minus_matrix,type_t_matrix, d_matrix, c_matrix, profit_matrix,i):
        # 初始化图表和数据

        d_value = d_matrix * profit_matrix
        c_value = c_matrix * profit_matrix
        dmean_of_positive = self.c_mean_v2(d_value)
        cmean_of_positive = self.c_mean_v2(c_value)
        count_0 = torch.sum(type_t_matrix == 0).item()
        count_1 = torch.sum(type_t_matrix == 1).item()
        D_Y = np.append(D_Y, count_0 / (L_num * L_num))
        C_Y = np.append(C_Y, count_1 / (L_num * L_num))
        D_Value = np.append(D_Value, dmean_of_positive)
        C_Value = np.append(C_Value, cmean_of_positive)
        all_value = np.append(all_value, profit_matrix.sum().item())
        CC, DD, CD, DC = self.cal_transfer_num(type_t_minus_matrix,type_t_matrix)
        return  D_Y, C_Y, D_Value, C_Value,all_value, count_0, count_1, CC, DD, CD, DC

    #计算转移的比例
    def cal_transfer_num(self,type_t_matrix,type_t1_matrix):
        CC=(torch.where((type_t_matrix==1)&(type_t1_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        DD=(torch.where((type_t_matrix==0)&(type_t1_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        CD=(torch.where((type_t_matrix==1)&(type_t1_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        DC=(torch.where((type_t_matrix==0)&(type_t1_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        return CC,DD,CD,DC

    #提取Qtable
    def extract_Qtable(self,Q_tensor, type_t_matrix):
        C_indices = torch.where(type_t_matrix.squeeze() == 1)[0]
        D_indices = torch.where(type_t_matrix.squeeze() == 0)[0]
        C_Q_table = Q_tensor[C_indices]
        D_indices = Q_tensor[D_indices]
        C_q_mean_matrix = torch.mean(C_Q_table, dim=0)
        D_q_mean_matrix = torch.mean(D_indices, dim=0)
        return D_q_mean_matrix.cpu().numpy(), C_q_mean_matrix.cpu().numpy()

    def split_four_policy_type(self,Q_matrix):
        CC = torch.where((Q_matrix[:, 1, 1] > Q_matrix[:, 1, 0]) & (
                Q_matrix[:, 0, 0] <= Q_matrix[:, 0, 1]), torch.tensor(1), torch.tensor(0))
        DD = torch.where((Q_matrix[:, 0, 0] > Q_matrix[:, 0, 1]) & (
                    Q_matrix[:, 1, 1] <= Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0))
        CDC = torch.where((Q_matrix[:, 0, 0] < Q_matrix[:, 0, 1]) & (Q_matrix[:, 1, 1] < Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0))
        StickStrategy=torch.where((Q_matrix[:,0,0]>Q_matrix[:,0,1])&(Q_matrix[:,1,1]>Q_matrix[:,1,0]),torch.tensor(1),torch.tensor(0))
        return DD.view((L_num,L_num)),CC.view((L_num,L_num)), CDC.view((L_num,L_num)), StickStrategy.view((L_num,L_num))

    def split_five_policy_type(self,Q_matrix,type_t_matrix):
        CC = torch.where((Q_matrix[:, 1, 1] > Q_matrix[:, 1, 0]) & (
                Q_matrix[:, 0, 0] <= Q_matrix[:, 0, 1]), torch.tensor(1), torch.tensor(0)).view((L_num,L_num))
        DD = torch.where((Q_matrix[:, 0, 0] > Q_matrix[:, 0, 1]) & (
                    Q_matrix[:, 1, 1] <= Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0)).view((L_num,L_num))
        CDC = torch.where((Q_matrix[:, 0, 0] < Q_matrix[:, 0, 1]) & (Q_matrix[:, 1, 1] < Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0)).view((L_num,L_num))
        StickStrategy=torch.where((Q_matrix[:,0,0]>Q_matrix[:,0,1])&(Q_matrix[:,1,1]>Q_matrix[:,1,0]),torch.tensor(1),torch.tensor(0)).view((L_num,L_num))
        CDC_C=CDC*torch.where(type_t_matrix==1,torch.tensor(1),torch.tensor(0))
        CDC_D=CDC*torch.where(type_t_matrix==0,torch.tensor(1),torch.tensor(0))
        CDC_neibor_num=0
        other_neibor_num=0
        CDC_neibor_DD, CDC_neibor_CC=torch.zeros((L_num,L_num)).to(device),torch.zeros((L_num,L_num)).to(device)
        if CDC.sum().item()!=0:
            CDC_neibor_matrix=self.pad_matrix(CDC.to(torch.float64).to(device))
            CDC_neibor_conv2d = torch.nn.functional.conv2d(CDC_neibor_matrix.view(1,1,L_num+2,L_num+2), neibor_kernel,
                                                          bias=None, stride=1, padding=0).view(L_num,L_num).to(device)
            CDC_neibor_num=(CDC_neibor_conv2d*CDC).sum().item()/CDC.sum().item()
            other_neibor_num = (CDC_neibor_conv2d * (1-CDC)).sum().item() / (1-CDC).sum().item()
            CDC_neibor_DD=torch.where(CDC_neibor_conv2d*(1-CDC)>0,torch.tensor(1),torch.tensor(0))*DD
            CDC_neibor_CC=torch.where(CDC_neibor_conv2d*(1-CDC)>0,torch.tensor(1),torch.tensor(0))*CC
        return DD,CC, CDC, StickStrategy,CDC_D,CDC_C,CDC_neibor_num,other_neibor_num,CDC_neibor_DD,CDC_neibor_CC

    def cal_four_type_value(self,DD,CC,CDC,StickStrategy,profit_matrix):
        CC_value = profit_matrix * CC
        DD_value = profit_matrix * DD
        CDC_value = profit_matrix * CDC
        StickStrategy_value = profit_matrix * StickStrategy
        return  DD_value,CC_value, CDC_value, StickStrategy_value

    def cal_five_type_value(self,DD,CC,CDC,StickStrategy,CDC_D,CDC_C,CDC_neibor_DD,CDC_neibor_CC,profit_matrix):
        CC_value = profit_matrix * CC
        DD_value = profit_matrix * DD
        CDC_value = profit_matrix * CDC
        StickStrategy_value = profit_matrix * StickStrategy
        CDC_C_value = profit_matrix * CDC_C
        CDC_D_value = profit_matrix * CDC_D
        CDC_neibor_DD_value = profit_matrix * CDC_neibor_DD
        CDC_neibor_CC_value = profit_matrix * CDC_neibor_CC
        return  DD_value,CC_value, CDC_value, StickStrategy_value,CDC_D_value,CDC_C_value,CDC_neibor_DD_value,CDC_neibor_CC_value


    def run(self,r,alpha,gamma,epsilon,epoches, L_num, device,type):
        # node= np.full((L_num,1),1)
        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000

        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
        # type_matrix=torch.tensor(node,dtype=torch.int).to(device)
        type_t_matrix = self.generated_default_type_matrix().to(device)
        type_t_minus_matrix = torch.zeros((L_num, L_num), dtype=torch.float64).to(device)
        type_t1_matrix = type_t_matrix
        value_matrix = torch.tensor(L, dtype=torch.float64).to(device)
        Q = np.zeros((L_num * L_num, 2, 2))
        Q_matrix = torch.tensor(Q, dtype=torch.float64).to(device)
        count_0=torch.where(type_t_matrix == 0, torch.tensor(1), torch.tensor(0)).sum().item()/ (L_num * L_num)
        count_1=1-count_0


        D_Y,C_Y = np.array([]),np.array([])
        CC_Y,DD_Y,CDC_Y,StickStrategy_Y=np.array([]),np.array([]),np.array([]),np.array([])
        D_Value,C_Value,all_value = np.array([]),np.array([]),np.array([])
        CC_value_np, DD_value_np, CDC_value_np, StickStrategy_value_np,CDC_D_value_np,CDC_C_value_np = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        CDC_neibor_num_np, other_neibor_num_np,CDC_neibor_DD_value_np,CDC_neibor_CC_value_np= np.array([]), np.array([]), np.array([]), np.array([])
        Q_C_DD,Q_C_DC,Q_C_CD,Q_C_CC=np.array([]),np.array([]),np.array([]),np.array([])
        Q_D_DD,Q_D_DC,Q_D_CD,Q_D_CC=np.array([]),np.array([]),np.array([]),np.array([])
        CC_data,DD_data,CD_data,DC_data=np.array([]),np.array([]),np.array([]),np.array([])
        D_Y= np.append(D_Y, count_0 )
        C_Y = np.append(C_Y, count_1 )


        for i in tqdm(range(epoches), desc='Processing'):
            # type_file_name = f'type\\type_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
            # Q_file_name = f'Q\\Q_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
            # V_file_name = f'V\\V_{i + 1}.pt'  # 这里使用了 i+1 作为文件名

            # 把一个L的三个type分开
            d_matrix, c_matrix = self.type_matrix_to_three_matrix(type_t_matrix)
            # 计算此次博弈利润的结果
            profit_matrix = self.calculation_value(r,type_t_matrix)
            # 计算得到的价值
            value_matrix = value_matrix + profit_matrix
            if i!=0:
                Q_matrix = self.updateQMatrix(alpha, gamma, type_t_minus_matrix, type_t_matrix, Q_matrix, profit_matrix)
                # 博弈演化,type变换，策略传播
                type_t1_matrix = self.fermiUpdate(type_t_minus_matrix,type_t_matrix,Q_matrix).to(device)
            # 把一个L的三个type分开
            d_matrix, c_matrix = self.type_matrix_to_three_matrix(type_t1_matrix)

            type_t_minus_matrix = type_t_matrix
            type_t_matrix = type_t1_matrix



            D_Y, C_Y, D_Value, C_Value,all_value, count_0, count_1, CC, DD, CD, DC = self.cal_fra_and_value( D_Y, C_Y,D_Value,C_Value,all_value,type_t_minus_matrix,type_t_matrix,d_matrix,c_matrix,profit_matrix,i)

            #DD,CC, CDC, StickStrategy = self.split_four_policy_type(Q_matrix)
            DD, CC, CDC, StickStrategy,CDC_D,CDC_C,CDC_neibor_num,other_neibor_num,CDC_neibor_DD,CDC_neibor_CC= self.split_five_policy_type(Q_matrix,type_t_minus_matrix)
            #DD_value,CC_value, CDC_value, StickStrategy_value = self.cal_four_type_value( DD,CC, CDC, StickStrategy,profit_matrix)
            DD_value,CC_value, CDC_value, StickStrategy_value,CDC_D_value,CDC_C_value,CDC_neibor_DD_value,CDC_neibor_CC_value = self.cal_five_type_value( DD,CC, CDC, StickStrategy,CDC_D,CDC_C,CDC_neibor_DD,CDC_neibor_CC,profit_matrix)
            Q_D_DD_data, Q_D_DC_data, Q_D_CD_data, Q_D_CC_data=0 if d_matrix.sum()==0 else ((Q_matrix[:,0,0]*(d_matrix.view(-1))).sum()/d_matrix.sum()).item(),0 if d_matrix.sum()==0 else ((Q_matrix[:,0,1]*(d_matrix.view(-1))).sum()/d_matrix.sum()).item(),0 if d_matrix.sum()==0 else ((Q_matrix[:,1,0]*(d_matrix.view(-1))).sum()/d_matrix.sum()).item(),0 if d_matrix.sum()==0 else ((Q_matrix[:,1,1]*(d_matrix.view(-1))).sum()/d_matrix.sum()).item()
            Q_C_DD_data, Q_C_DC_data, Q_C_CD_data, Q_C_CC_data=0 if c_matrix.sum()==0 else ((Q_matrix[:,0,0]*(c_matrix.view(-1))).sum()/c_matrix.sum()).item(),0 if c_matrix.sum()==0 else ((Q_matrix[:,0,1]*(c_matrix.view(-1))).sum()/c_matrix.sum()).item(),0 if c_matrix.sum()==0 else ((Q_matrix[:,1,0]*(c_matrix.view(-1))).sum()/c_matrix.sum()).item(),0 if c_matrix.sum()==0 else ((Q_matrix[:,1,1]*(c_matrix.view(-1))).sum()/c_matrix.sum()).item()
            CC_data = np.append(CC_data, CC.sum().item()/ (L_num * L_num))
            DD_data = np.append(DD_data, DD.sum().item()/ (L_num * L_num))
            CD_data = np.append(CD_data, CDC.sum().item()/ (L_num * L_num))
            DC_data = np.append(DC_data, StickStrategy.sum().item()/ (L_num * L_num))
            CC_Y = np.append(CC_Y, CC.sum().item()/ (L_num * L_num))
            DD_Y = np.append(DD_Y, DD.sum().item()/ (L_num * L_num))
            CDC_Y = np.append(CDC_Y, CDC.sum().item()/ (L_num * L_num))
            StickStrategy_Y = np.append(StickStrategy_Y, StickStrategy.sum().item()/ (L_num * L_num))
            CC_value_np=np.append(CC_value_np,0 if CC.sum().item()==0 else CC_value.sum().item()/(CC.sum().item()))
            DD_value_np=np.append(DD_value_np,0 if DD.sum().item()==0 else DD_value.sum().item()/(DD.sum().item()))
            CDC_value_np=np.append(CDC_value_np,0 if CDC.sum().item()==0 else CDC_value.sum().item()/(CDC.sum().item()))
            StickStrategy_value_np=np.append(StickStrategy_value_np,0 if StickStrategy.sum().item()==0 else StickStrategy_value.sum().item()/(StickStrategy.sum().item()))
            CDC_C_value_np=np.append(CDC_C_value_np,0 if CDC_C.sum().item()==0 else CDC_C_value.sum().item()/(CDC_C.sum().item()))
            CDC_D_value_np=np.append(CDC_D_value_np,0 if CDC_D.sum().item()==0 else CDC_D_value.sum().item()/(CDC_D.sum().item()))
            CDC_neibor_num_np=np.append(CDC_neibor_num_np,CDC_neibor_num)
            other_neibor_num_np=np.append(other_neibor_num_np,other_neibor_num)
            CDC_neibor_DD_value_np=np.append(CDC_neibor_DD_value_np,0 if CDC_neibor_DD.sum().item()==0 else CDC_neibor_DD_value.sum().item()/(CDC_neibor_DD.sum().item()))
            CDC_neibor_CC_value_np=np.append(CDC_neibor_CC_value_np,0 if CDC_neibor_CC.sum().item()==0 else CDC_neibor_CC_value.sum().item()/(CDC_neibor_CC.sum().item()))
            four_type_matrix = (CC*1+CDC*2+StickStrategy*3).view((L_num,L_num))
            Q_D_DD, Q_D_DC, Q_D_CD, Q_D_CC = np.append(Q_D_DD, Q_D_DD_data), np.append(Q_D_DC, Q_D_DC_data), np.append(Q_D_CD, Q_D_CD_data), np.append(Q_D_CC, Q_D_CC_data)
            Q_C_DD, Q_C_DC, Q_C_CD, Q_C_CC = np.append(Q_C_DD, Q_C_DD_data), np.append(Q_C_DC, Q_C_DC_data), np.append(Q_C_CD, Q_C_CD_data), np.append(Q_C_CC, Q_C_CC_data)



            if i==0:
                self.shot_pic(type_t_minus_matrix,i,r)
            if i==0  or i==9 or i==10 or i==11 or i==12 or i==13 or i==14 or i==15 or i==16 or i==17 or i==18 or i==19 or i==49 or i==99 or i==299 or i==499 or i==799 or i==999 or i==4999 or i==9999 or i==19999 or i==29999 or i==39999 or i==49999:
                self.shot_pic(type_t_matrix,i+1,r)
            if i==0  or i==9 or i==10 or i==11 or i==12 or i==13 or i==14 or i==15 or i==16 or i==17 or i==18 or i==19 or i==49 or i==99 or i==299 or i==499 or i==799 or i==999 or i==4999 or i==9999 or i==19999 or i==29999 or i==39999 or i==49999:
                self.shot_pic2(four_type_matrix,i+1,r)

        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000
        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
        if (type == "line2"):
            return D_Y[-1], C_Y[-1]
        elif (type == "line1"):
            return D_Y, C_Y, D_Value, C_Value, all_value, Q_matrix, type_t_matrix, count_0, count_1, \
                   CC_data, DD_data, CD_data, DC_data, DD_Y, CC_Y, CDC_Y, StickStrategy_Y, DD_value_np, CC_value_np, CDC_value_np, StickStrategy_value_np, \
                   Q_D_DD, Q_D_DC, Q_D_CD, Q_D_CC, Q_C_DD, Q_C_DC, Q_C_CD, Q_C_CC
                   # CDC_D_value_np, CDC_C_value_np, CDC_neibor_num_np, other_neibor_num_np, CDC_neibor_DD_value_np, CDC_neibor_CC_value_np
        elif (type == "Qtable"):
            return Q_matrix,type_t_matrix

    def mkdir(self,path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

    def save_data(self,type,name,r,count,data):
        self.mkdir('data/Origin_Fermi_Qlearning1/generated1/'+str(type))
        np.savetxt('data/Origin_Fermi_Qlearning1/generated1/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(type), name,str(r),str(self.epoches),str(self.L_num),str(count)), data)
        # try:
        #     np.savetxt('data/Origin_Qlearning_NeiborLearning/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(type), name, str(r),str(self.epoches),str(self.L_num), str(count)),data)
        # except:
        #     print("Save failed")

    def run_line2_pic(self, loop_num1= 51, loop_num2 = 10):
        r=29
        for j in range(loop_num1):
            for i in range(loop_num2):
                r1=r/10
                print("loop_num1: "+str(j)+" loop_num2: "+str(i)+" r="+str(r1))
                self.count=i
                D_Y, C_Y, D_Value, C_Value, all_value, Q_matrix, type_t_matrix, count_0, count_1, CC_data, DD_data, CD_data, DC_data, \
                DD_Y, CC_Y, CDC_Y, StickStrategy_Y, DD_value_np, CC_value_np, CDC_value_np, StickStrategy_value_np, \
                Q_D_DD, Q_D_DC, Q_D_CD, Q_D_CC, Q_C_DD, Q_C_DC, Q_C_CD, Q_C_CC = self.run(r1, self.alpha,self.gamma, self.epsilon,self.epoches, self.L_num,self.device, type="line1")
                self.save_data('D_fra', 'D_fra', r1, i, D_Y)
                self.save_data('C_fra', 'C_fra', r1, i, C_Y)
                self.save_data('C_value', 'C_value', r1, i, C_Value)
                self.save_data('D_value', 'D_value', r1, i, D_Value)
                self.save_data('all_value','all_value',r1,i,all_value)
                self.save_data('CC_fra', 'CC_fra', r1, i, CC_data)
                self.save_data('DD_fra', 'DD_fra', r1, i, DD_data)
                self.save_data('CD_fra', 'CD_fra', r1, i, CD_data)
                self.save_data('DC_fra', 'DC_fra', r1, i, DC_data)
                self.save_data('DD_Y', 'DD_Y', r1, i, DD_Y)
                self.save_data('CC_Y', 'CC_Y', r1, i, CC_Y)
                self.save_data('CDC_Y', 'CDC_Y', r1, i, CDC_Y)
                self.save_data('StickStrategy_Y', 'StickStrategy_Y', r1, i, StickStrategy_Y)
                self.save_data('DD_value_np', 'DD_value_np', r1, i, DD_value_np)
                self.save_data('CC_value_np', 'CC_value_np', r1, i, CC_value_np)
                self.save_data('CDC_value_np', 'CDC_value_np', r1, i, CDC_value_np)
                self.save_data('StickStrategy_value_np', 'StickStrategy_value_np', r1, i, StickStrategy_value_np)
                self.save_data('Q_D_DD', 'Q_D_DD', r1, i, Q_D_DD)
                self.save_data('Q_D_DC', 'Q_D_DC', r1, i, Q_D_DC)
                self.save_data('Q_D_CD', 'Q_D_CD', r1, i, Q_D_CD)
                self.save_data('Q_D_CC', 'Q_D_CC', r1, i, Q_D_CC)
                self.save_data('Q_C_DD', 'Q_C_DD', r1, i, Q_C_DD)
                self.save_data('Q_C_DC', 'Q_C_DC', r1, i, Q_C_DC)
                self.save_data('Q_C_CD', 'Q_C_CD', r1, i, Q_C_CD)
                self.save_data('Q_C_CC', 'Q_C_CC', r1, i, Q_C_CC)
                # self.save_data('CDC_D_value_np', 'CDC_D_value_np', r1, i, CDC_D_value_np)
                # self.save_data('CDC_C_value_np', 'CDC_C_value_np', r1, i, CDC_C_value_np)
                # self.save_data('CDC_neibor_num_np', 'CDC_neibor_num_np', r1, i, CDC_neibor_num_np)
                # self.save_data('other_neibor_num_np', 'other_neibor_num_np', r1, i, other_neibor_num_np)
                # self.save_data('CDC_neibor_DD_value_np', 'CDC_neibor_DD_value_np', r1, i, CDC_neibor_DD_value_np)
                # self.save_data('CDC_neibor_CC_value_np', 'CDC_neibor_CC_value_np', r1, i, CDC_neibor_CC_value_np)
            r=r+1



    # def cal_transfer_pic(self):
    #     D_Y, C_Y, D_Value, C_Value,all_value, Q_matrix, type_t_matrix, count_0, count_1,  CC_data, DD_data, CD_data, DC_data,DD_Y,CC_Y,CDC_Y\
    #         ,StickStrategy_Y,DD_value_np,CC_value_np,CDC_value_np,StickStrategy_value_np,CDC_D_value_np,CDC_C_value_np=\
    #         self.run(self.r, self.alpha,self.gamma,self.epsilon,self.epoches, self.L_num,self.device,type="line1")
    #     self.draw_transfer_pic(CC_data, DD_data,CD_data,DC_data, xticks, fra_yticks,r=self.r,epoches=self.epoches)

    def extra_Q_table(self,loop_num):
        for i in range(loop_num):
            Q_matrix,type_t_matrix = self.run(self.r, self.alpha,self.gamma,self.epsilon,self.epoches, self.L_num,self.device,type="Qtable")
            D_q_mean_matrix, C_q_mean_matrix = self.extract_Qtable(Q_matrix, type_t_matrix)
            print(D_q_mean_matrix,C_q_mean_matrix)
            self.save_data('D_Qtable', 'D_Qtable',self.r, str(i), D_q_mean_matrix)
            self.save_data('C_Qtable', 'C_Qtable',self.r, str(i), C_q_mean_matrix)

    # def hot_pic(self,loop_num1=50,loop_num2 = 50,L_num=100):
    #     alpha=0
    #     gamma=0
    #     for j in range(loop_num1):
    #         for i in range(loop_num2):
    #             r1 = r / 10
    #             print("loop_num1: " + str(j) + " loop_num2: " + str(i) + " r=" + str(r1))
    #             D_Y, C_Y, D_Value, C_Value,all_value, Q_matrix, type_t_matrix, count_0, count_1,  CC_data, DD_data, CD_data, DC_data,\
    #             DD_Y,CC_Y,CDC_Y,StickStrategy_Y,DD_value_np,CC_value_np,CDC_value_np,StickStrategy_value_np,CDC_D_value_np,CDC_C_value_np = self.run(
    #                 r1, alpha, gamma, self.epsilon, self.epoches, self.L_num, self.device, type="line1")
    #             self.save_data('C_fra', 'C_fra', r1, i, C_Y)
    #             self.save_data('D_fra', 'D_fra', r1, i, D_Y)
    #             self.save_data('C_value', 'C_value', r1, i, C_Value)
    #             self.save_data('D_value', 'D_value', r1, i, D_Value)
    #             self.save_data('CC_fra', 'CC_fra', r1, i, CC_data)
    #             self.save_data('DD_fra', 'DD_fra', r1, i, DD_data)
    #             self.save_data('CD_fra', 'CD_fra', r1, i, CD_data)
    #             self.save_data('DC_fra', 'DC_fra', r1, i, DC_data)
    #         r = r + 1

    def line1_pic(self, r):
        loop_num = 1
        for i in range(loop_num):
            print("第i轮:", i)
            self.count = i
            D_Y, C_Y, D_Value, C_Value, all_value, Q_matrix, type_t_matrix, count_0, count_1,CC_data, DD_data, CD_data, DC_data, \
            DD_Y, CC_Y, CDC_Y, StickStrategy_Y, DD_value_np, CC_value_np, CDC_value_np, StickStrategy_value_np, \
            Q_D_DD, Q_D_DC, Q_D_CD, Q_D_CC,Q_C_DD, Q_C_DC, Q_C_CD, Q_C_CC = self.run(self.r, self.alpha, self.gamma, self.epsilon, self.epoches, self.L_num, self.device, type="line1")
            self.save_data('D_fra', 'D_fra', r, i, D_Y)
            self.save_data('C_fra', 'C_fra', r, i, C_Y)
            self.save_data('C_value', 'C_value', r, i, C_Value)
            self.save_data('D_value', 'D_value', r, i, D_Value)
            self.save_data('all_value', 'all_value', r, i, all_value)
            self.save_data('CC_fra', 'CC_fra', r, i, CC_data)
            self.save_data('DD_fra', 'DD_fra', r, i, DD_data)
            self.save_data('CD_fra', 'CD_fra', r, i, CD_data)
            self.save_data('DC_fra', 'DC_fra', r, i, DC_data)
            self.save_data('DD_Y', 'DD_Y', r, i, DD_Y)
            self.save_data('CC_Y', 'CC_Y', r, i, CC_Y)
            self.save_data('CDC_Y', 'CDC_Y', r, i, CDC_Y)
            self.save_data('StickStrategy_Y', 'StickStrategy_Y', r, i, StickStrategy_Y)
            self.save_data('DD_value_np', 'DD_value_np', r, i, DD_value_np)
            self.save_data('CC_value_np', 'CC_value_np', r, i, CC_value_np)
            self.save_data('CDC_value_np', 'CDC_value_np', r, i, CDC_value_np)
            self.save_data('StickStrategy_value_np', 'StickStrategy_value_np', r, i, StickStrategy_value_np)
            self.save_data('Q_D_DD', 'Q_D_DD', r, i, Q_D_DD)
            self.save_data('Q_D_DC', 'Q_D_DC', r, i, Q_D_DC)
            self.save_data('Q_D_CD', 'Q_D_CD', r, i, Q_D_CD)
            self.save_data('Q_D_CC', 'Q_D_CC', r, i, Q_D_CC)
            self.save_data('Q_C_DD', 'Q_C_DD', r, i, Q_C_DD)
            self.save_data('Q_C_DC', 'Q_C_DC', r, i, Q_C_DC)
            self.save_data('Q_C_CD', 'Q_C_CD', r, i, Q_C_CD)
            self.save_data('Q_C_CC', 'Q_C_CC', r, i, Q_C_CC)


def draw_shot():
    r_list=[2.5,2.9,3.3]
    for r in r_list:
        SPGG = SPGG_Qlearning(L_num, device, alpha, gamma, epsilon,lr=0.2, r=r, epoches=20000,eta=0.8,cal_transfer=True)
        SPGG.line1_pic(r)

if __name__ == '__main__':
    r=3.8
    SPGG=SPGG_Qlearning(L_num,device,alpha,gamma,epsilon,r=r,epoches=20000,lr=0.2,eta=0.8,cal_transfer=True)
    SPGG.run_line2_pic(loop_num1=51,loop_num2 = 10)
    # SPGG.extra_Q_table(10)
    #SPGG=SPGG_Qlearning(L_num,device,alpha,gamma,epsilon,r=r,epoches=10000,cal_transfer=True)
    #SPGG.run_line2_pic(loop_num1=51,loop_num2 = 10)
    #SPGG.line1_pic(r)
    #SPGG.cal_transfer_pic()
    #SPGG.extra_Q_table(10)
    #draw_shot()

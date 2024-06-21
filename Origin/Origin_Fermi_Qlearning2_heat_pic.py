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

L_num=50
torch.cuda.set_device("cuda:4" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
alpha=0.8
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
        #update_values = Q_tensor[C_indices, A_indices, B_indices] + alpha * (profit_matrix.view(-1) + gamma * max_values - Q_tensor[C_indices, A_indices, B_indices])
        update_values = (1 - self.eta) * Q_tensor[C_indices, A_indices, B_indices] + self.eta * (profit_matrix.view(-1) + gamma * max_values)
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

    def fermiUpdate(self,type_t_minus_matrix,type_t_matrix,Q_tensor):
        #遍历每一个Qtable
        C_indices = torch.arange(type_t_minus_matrix.numel()).to(device)
        #Qtable中选择的行
        A_indices = type_t_minus_matrix.view(-1).long()
        #Qtable中选择的列
        B_indices = type_t_matrix.view(-1).long()
        profit_matrix = Q_tensor[C_indices, A_indices, B_indices].view(L_num, L_num)
        #计算费米更新的概率
        W_left,W_right,W_up,W_down=self.profit_Matrix_to_Four_Matrix(profit_matrix,0.5)
        indices_left, indices_right, indices_up, indices_down = self.indices_Matrix_to_Four_Matrix(type_t_matrix)
        #生成一个矩阵随机决定向哪个方向学习
        learning_direction=torch.randint(0,4,(L_num,L_num)).to(device)

        #生成一个随机矩阵决定是否向他学习
        learning_probabilities=torch.rand(L_num,L_num).to(device)
        #费米更新
        left_type_t1_matrix=(learning_direction==0)*((learning_probabilities<=W_left)*indices_left+(learning_probabilities>W_left)*type_t_matrix)
        right_type_t1_matrix=(learning_direction==1)*((learning_probabilities<=W_right)*indices_right+(learning_probabilities>W_right)*type_t_matrix)
        up_type_t1_matrix=(learning_direction==2)*((learning_probabilities<=W_up)*indices_up+(learning_probabilities>W_up)*type_t_matrix)
        down_type_t1_matrix=(learning_direction==3)*((learning_probabilities<=W_down)*indices_down+(learning_probabilities>W_down)*type_t_matrix)
        type_t1_matrix= left_type_t1_matrix+right_type_t1_matrix+up_type_t1_matrix+down_type_t1_matrix

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

            profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
            return profit_matrix.view(L_num, L_num).to(torch.float64)

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

    #求利润均值_version_two
    def c_mean_v2(self,value_tensor):
        # 创建布尔张量，表示大于零的元素
        positive_num = (value_tensor > 0).to(device)
        negetive_num = (value_tensor < 0).to(device)
        # 计算大于零的元素的均值
        mean_of_positive_elements = (value_tensor.to(torch.float64).sum()) / ((positive_num + negetive_num).sum())
        return mean_of_positive_elements.to("cpu")

    #计算CD比例和利润
    def cal_fra_and_value(self, D_Y, C_Y, D_Value, C_Value,type_t_matrix, d_matrix, c_matrix, profit_matrix):
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
        return  D_Y, C_Y, D_Value, C_Value, count_0, count_1


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
        D_Value,C_Value = np.array([]),np.array([])

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

            D_Y, C_Y, D_Value, C_Value, count_0, count_1 = self.cal_fra_and_value(D_Y, C_Y, D_Value, C_Value,type_t_matrix, d_matrix, c_matrix, profit_matrix)


        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000
        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
        return D_Y, C_Y, D_Value, C_Value


    def mkdir(self,path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

    def save_data(self,type,name,r,count,data):
        self.mkdir('data/Origin_Fermi_Qlearning2/heat_pic_r=2.777/eta={}/gammm={}/'.format(str(self.eta),str(self.gamma))+str(type))
        np.savetxt('data/Origin_Fermi_Qlearning2/heat_pic_r=2.777/eta={}/gammm={}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(self.eta),str(self.gamma),str(type), name,str(r),str(self.epoches),str(self.L_num),str(count)), data)


    def line1_pic(self, r,i):
        self.count = i
        D_Y, C_Y, D_Value, C_Value= self.run(self.r, self.alpha, self.gamma, self.epsilon, self.epoches, self.L_num, self.device, type="line1")
        self.save_data('D_fra', 'D_fra', r, i, D_Y)
        self.save_data('C_fra', 'C_fra', r, i, C_Y)
        self.save_data('C_value', 'C_value', r, i, C_Value)
        self.save_data('D_value', 'D_value', r, i, D_Value)


def draw_heat_pic():
    r=25/9
    loop_num1=1
    loop_num_eta=101
    loop_num_gamma=101
    start_eta=0
    start_gamma=0
    for i in range(loop_num1):
        for eta1 in range(loop_num_eta):
            for gamma1 in range(loop_num_gamma):
                eta2=(start_eta+eta1)/100
                gamma2=(start_gamma+gamma1)/100
                print(f"r={r:.3f}","eta=", eta2," gamma=", gamma2, " 第i轮: ", i)
                SPGG = SPGG_Qlearning(50, device, alpha, gamma2, epsilon, lr=0.2, r=r, epoches=10000, eta=eta2, cal_transfer=True)
                SPGG.line1_pic(r,i)

if __name__ == '__main__':
    draw_heat_pic()

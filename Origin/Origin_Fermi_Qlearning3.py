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
torch.cuda.set_device("cuda:4" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
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
    #画快照
    def shot_pic(self,type_t_matrix: tensor,i,r):
        plt.clf()
        plt.close("all")
        # 初始化图表和数据
        fig = plt.figure(figsize=(40,40))
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)
        # 指定图的大小
        #             plt.figure(figsize=(500, 500))  # 10x10的图
        #             plt.matshow(type_t_matrix.cpu().numpy(), cmap=cmap)
        #             plt.colorbar(ticks=[0, 1, 2], label='Color')
        # 显示图片
        # 定义颜色映射
        color_map = {
            #0设置为黑色
            0: (0, 0, 0),  # 黑色
            #1设置为白色
            1: (255, 255, 255),  # 白色
        }
        image = np.zeros((L_num, L_num, 3), dtype=np.uint8)
        for label, color in color_map.items():
            image[type_t_matrix.cpu() == label] = color
        #plt.title('Qlearning: '+f"T:{i}")
        # 隐藏坐标轴刻度标签
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['bottom'].set_linewidth(3)  # 设置x轴底部线条宽度
        ax.spines['left'].set_linewidth(3)  # 设置y轴左侧线条宽度
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        plt.imshow(image,interpolation='None')

        self.mkdir('data/Origin_Fermi_Qlearning3/shot_pic/r={}/two_type/generated1'.format(r))
        plt.savefig('data/Origin_Fermi_Qlearning3/shot_pic/r={}/two_type/generated1/t={}.png'.format(r, i))
        #plt.show()
        plt.clf()
        plt.close("all")


    def shot_pic2(self,type_t_matrix: tensor,i,r):
        plt.clf()
        plt.close("all")
        # 初始化图表和数据
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 4)
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
        self.mkdir('data/Origin_Fermi_Qlearning3/shot_pic/r={}/four_type/generated1'.format(r))
        plt.savefig('data/Origin_Fermi_Qlearning3/shot_pic/r={}/four_type/generated1/t={}.png'.format(r,i))
        self.mkdir('data/Origin_Fermi_Qlearning3/shot_pic/r={}/four_type/generated1/type_t_matrix'.format(r))
        torch.save(type_t_matrix.int(), 'data/Origin_Fermi_Qlearning3/shot_pic/r={}/four_type/generated1/type_t_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(r), "type_t_matrix", str(r), str(self.epoches), str(self.L_num),str(i), str(self.count)))
        #np.savetxt('data/Origin_Fermi_Qlearning3/shot_pic/r={}/four_type/generated1/type_t_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(r), "type_t_matrix", str(r), str(self.epoches), str(self.L_num),str(i), str(self.count)), type_t_matrix.int().cpu().numpy())

        #plt.show()
        plt.clf()
        plt.close("all")

    def shot_save_data(self,type_t_minus_matrix: tensor,type_t_matrix: tensor,i,r,profit_matrix,Q_matrix):
        # 遍历每一个Qtable
        C_indices = torch.arange(type_t_matrix.numel()).to(device)
        # Qtable中选择的行
        A_indices = type_t_minus_matrix.view(-1).long()
        # Qtable中选择的列
        B_indices = type_t_matrix.view(-1).long()
        Q_sa_matrix = Q_matrix[C_indices, A_indices, B_indices].view(L_num, L_num)
        self.mkdir('data/Origin_Fermi_Qlearning3/shot_pic/r={}/two_type/generated1/type_t_matrix'.format(r))
        np.savetxt('data/Origin_Fermi_Qlearning3/shot_pic/r={}/two_type/generated1/type_t_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(r), "type_t_matrix", str(r), str(self.epoches), str(self.L_num), str(i), str(self.count)),type_t_matrix.cpu().numpy())
        self.mkdir('data/Origin_Fermi_Qlearning3/shot_pic/r={}/two_type/generated1/profit_matrix'.format(r))
        np.savetxt('data/Origin_Fermi_Qlearning3/shot_pic/r={}/two_type/generated1/profit_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(r), "profit_matrix", str(r), str(self.epoches), str(self.L_num), str(i), str(self.count)),profit_matrix.cpu().numpy())
        self.mkdir('data/Origin_Fermi_Qlearning3/shot_pic/r={}/two_type/generated1/Q_sa_matrix'.format(r))
        np.savetxt('data/Origin_Fermi_Qlearning3/shot_pic/r={}/two_type/generated1/Q_sa_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(r), "Q_sa_matrix", str(r), str(self.epoches), str(self.L_num), str(i), str(self.count)),Q_sa_matrix.cpu().numpy())
        self.mkdir('data/Origin_Fermi_Qlearning3/shot_pic/r={}/two_type/generated1/Q_matrix'.format(r))
        torch.save(Q_matrix,'data/Origin_Fermi_Qlearning3/shot_pic/r={}/two_type/generated1/Q_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(r), "Q_matrix", str(r), str(self.epoches), str(self.L_num), str(i), str(self.count)))


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


    def run(self,r,alpha,gamma,epsilon,epoches, L_num, device,type):
        # node= np.full((L_num,1),1)
        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000

        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
        # type_matrix=torch.tensor(node,dtype=torch.int).to(device)
        type_t_matrix = self.generated_default_type_matrix().to(device)
        type_t_minus_matrix = type_t_matrix.clone().detach().to(device)
        type_t1_matrix = type_t_matrix.clone().detach().to(device)
        value_matrix = torch.zeros((L_num, L_num), dtype=torch.float64).to(device)
        Q_matrix = torch.zeros((L_num * L_num, 2, 2), dtype=torch.float64).to(device)
        count_0=torch.where(type_t_matrix == 0, torch.tensor(1), torch.tensor(0)).sum().item()/ (L_num * L_num)
        count_1=1-count_0


        D_Y,C_Y = np.array([]),np.array([])
        D_Value,C_Value,all_value = np.array([]),np.array([]),np.array([])

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
            Q_matrix = self.updateQMatrix(alpha, gamma, type_t_minus_matrix, type_t_matrix, Q_matrix, profit_matrix)
            # 博弈演化,type变换，策略传播
            type_t1_matrix = self.fermiUpdate(type_t_minus_matrix,type_t_matrix,Q_matrix).to(device)
            if i == 0 or i == 9 or i == 10 or i == 11 or i == 99 or i == 299 or i == 499 or i == 799 or i == 999 or i == 4999 or i == 9999 or i == 19999 or i == 29999 or i == 39999 or i == 49998 or i == 49999 or i == 99999 or i == 199999 or i == 299999 or i == 399999 or i == 499999 or i == 599999 or i == 699999 or i == 799999 or i == 899999 or i == 999999:
                self.shot_save_data(type_t_minus_matrix, type_t_matrix, i, r, profit_matrix, Q_matrix)
            # 把一个L的三个type分开
            d_matrix, c_matrix = self.type_matrix_to_three_matrix(type_t1_matrix)

            type_t_minus_matrix = type_t_matrix
            type_t_matrix = type_t1_matrix



            D_Y, C_Y, D_Value, C_Value,all_value, count_0, count_1, CC, DD, CD, DC = self.cal_fra_and_value( D_Y, C_Y,D_Value,C_Value,all_value,type_t_minus_matrix,type_t_matrix,d_matrix,c_matrix,profit_matrix,i)


            # if i==0:
            #     self.shot_pic(type_t_minus_matrix,i,r)
            # if i==0  or i==9 or i==10 or i==11 or i==99 or i==299 or i==499 or i==799 or i==999 or i==4999 or i==9999 or i==19999 or i==29999 or i==39999 or i==49998 or i==49999 or i==99999 or i==199999 or i==299999 or i==399999 or i==499999 or i==599999 or i==699999 or i==799999 or i==899999 or i==999999:
            #     self.shot_pic(type_t_matrix,i+1,r)
            # if i==0 or i==10 or (i>=99 and i<=110) or (i>=180 and i<=220) or (i>=800 and i<=900) or i==999 or (i>=4980 and i<=5020) or i==9999:
            #     self.shot_pic(type_t_minus_matrix,i,r,profit_matrix)

        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000
        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
        return D_Y, C_Y, D_Value, C_Value,all_value, Q_matrix, type_t_matrix, count_0, count_1

    def mkdir(self,path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

    def save_data(self,type,name,r,count,data):
        self.mkdir('data/Origin_Fermi_Qlearning3/'+str(type))
        np.savetxt('data/Origin_Fermi_Qlearning3/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(type), name,str(r),str(self.epoches),str(self.L_num),str(count)), data)


    def run_line2_pic(self, loop_num1= 51, loop_num2 = 10):
        r=27
        for j in range(27,loop_num1):
            for i in range(loop_num2):
                r1=r/10
                print("loop_num1: "+str(j)+" loop_num2: "+str(i)+" r="+str(r1))
                self.count=i
                D_Y, C_Y, D_Value, C_Value, all_value, Q_matrix, type_t_matrix, count_0, count_1= self.run(r1, self.alpha, self.gamma, self.epsilon, self.epoches, self.L_num, self.device, type="line1")
                self.save_data('D_fra', 'D_fra', r1, i, D_Y)
                self.save_data('C_fra', 'C_fra', r1, i, C_Y)
                self.save_data('C_value', 'C_value', r1, i, C_Value)
                self.save_data('D_value', 'D_value', r1, i, D_Value)
                self.save_data('all_value','all_value',r1,i,all_value)
            r=r+1


    def extra_Q_table(self,loop_num):
        for i in range(loop_num):
            Q_matrix,type_t_matrix = self.run(self.r, self.alpha,self.gamma,self.epsilon,self.epoches, self.L_num,self.device,type="Qtable")
            D_q_mean_matrix, C_q_mean_matrix = self.extract_Qtable(Q_matrix, type_t_matrix)
            print(D_q_mean_matrix,C_q_mean_matrix)
            self.save_data('D_Qtable', 'D_Qtable',self.r, str(i), D_q_mean_matrix)
            self.save_data('C_Qtable', 'C_Qtable',self.r, str(i), C_q_mean_matrix)

    def line1_pic(self, r):
        loop_num = 10
        for i in range(loop_num):
            print("第i轮:", i)
            self.count = i
            D_Y, C_Y, D_Value, C_Value, all_value, Q_matrix, type_t_matrix, count_0, count_1 = self.run(r, self.alpha,self.gamma,self.epsilon,self.epoches,self.L_num,self.device,type="line1")
            self.save_data('D_fra', 'D_fra', r, i, D_Y)
            self.save_data('C_fra', 'C_fra', r, i, C_Y)
            self.save_data('C_value', 'C_value', r, i, C_Value)
            self.save_data('D_value', 'D_value', r, i, D_Value)
            self.save_data('all_value', 'all_value', r, i, all_value)
            #self.save_data('D_fra', 'D_fra', r, i, D_Y)
            #self.save_data('C_fra', 'C_fra', r, i, C_Y)
            #self.save_data('C_value', 'C_value', r, i, C_Value)
            #self.save_data('D_value', 'D_value', r, i, D_Value)
            #self.save_data('all_value', 'all_value', r, i, all_value)
            #self.save_data('CC_fra', 'CC_fra', r, i, CC_data)
            #self.save_data('DD_fra', 'DD_fra', r, i, DD_data)
            #self.save_data('CD_fra', 'CD_fra', r, i, CD_data)
            #self.save_data('DC_fra', 'DC_fra', r, i, DC_data)
            #self.save_data('Q_D_DD', 'Q_D_DD', r, i, Q_D_DD)
            #self.save_data('Q_D_DC', 'Q_D_DC', r, i, Q_D_DC)
            #self.save_data('Q_D_CD', 'Q_D_CD', r, i, Q_D_CD)
            #self.save_data('Q_D_CC', 'Q_D_CC', r, i, Q_D_CC)
            #self.save_data('Q_C_DD', 'Q_C_DD', r, i, Q_C_DD)
            #self.save_data('Q_C_DC', 'Q_C_DC', r, i, Q_C_DC)
            #self.save_data('Q_C_CD', 'Q_C_CD', r, i, Q_C_CD)
            #self.save_data('Q_C_CC', 'Q_C_CC', r, i, Q_C_CC)


def draw_shot():
    r_list=[25/9]
    for r in r_list:
        SPGG = SPGG_Qlearning(L_num, device, alpha, gamma, epsilon,lr=0.2, r=r, epoches=10000,eta=0.8,cal_transfer=True)
        SPGG.line1_pic(r)



if __name__ == '__main__':
    r=0
    SPGG=SPGG_Qlearning(L_num,device,alpha,gamma,epsilon,r=r,epoches=100000,lr=0.2,eta=0.8,cal_transfer=True)
    #draw_shot()
    SPGG.run_line2_pic(loop_num1=36,loop_num2 = 10)
    #SPGG.extra_Q_table(10)
    #SPGG=SPGG_Qlearning(L_num,device,alpha,gamma,epsilon,r=r,epoches=10000,cal_transfer=True)
    #SPGG.run_line2_pic(loop_num1=51,loop_num2 = 10)
    #SPGG.line1_pic(r)
    #SPGG.cal_transfer_pic()
    #SPGG.extra_Q_table(10)

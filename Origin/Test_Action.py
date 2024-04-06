import torch

from torch import tensor
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

L_num=3
mid=int((L_num*L_num-1)/2)
torch.cuda.set_device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
alpha=0.8
gamma=0.8
epsilon=0.02
neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float32).to(device).view(1,1,3,3)
actions = torch.tensor([0, 1],dtype=torch.float32).to(device)
L = np.full((L_num, L_num), 0)
value_matrix = torch.tensor(L, dtype=torch.float32).to(device)

zeros_tensor = torch.zeros((1, 1, L_num, L_num),dtype=torch.float32).to(torch.float32)
g_matrix=torch.nn.functional.conv2d(torch.ones((1,1,L_num, L_num),dtype=torch.float32).to(device), neibor_kernel,
                                                      bias=None, stride=1, padding=1).to(device)
xticks=[0, 10, 100, 1000, 10000, 100000]
fra_yticks=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95, 1.00]
Q_xticks=[0,1,2,3,4,5,6,7,8,9,10,11]
Q_yticks=[-10,-5,0,5,10,15,20, 25, 30, 35, 40, 45, 50,60]
profite_yticks=[-3,-1,0,1,3,5,7]
C_env = torch.zeros((L_num, L_num), dtype=torch.float32).to(device)
C_env[1][1] = 1

class SPGG_Qlearning(nn.Module):
    def __init__(self,L_num,device,alpha,gamma,epsilon,r,epoches,cal_transfer=False):
        super(SPGG_Qlearning, self).__init__()
        self.epoches=epoches
        self.L_num=L_num
        self.device=device
        self.alpha=alpha
        self.r=r
        self.gamma=gamma
        self.epsilon=epsilon
        self.cal_transfer=cal_transfer

    #Qtable更新
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
        # print(update_values)
        # 更新 type_t_matrix
        #更新Qtable
        Q_tensor[C_indices, A_indices, B_indices] = update_values
        return Q_tensor

    def updateQMatrix2(self,alpha,gamma,type_t_matrix: tensor, type_t1_matrix: tensor, Q_tensor: tensor, profit_matrix: tensor):
        #遍历每一个Qtable
        C_indices = mid
        #Qtable中选择的行
        A_indices = type_t_matrix.view(-1).long()[mid]
        #Qtable中选择的列
        B_indices = type_t1_matrix.view(-1).long()[mid]
        # 计算更新值
        max_values = torch.max(Q_tensor[C_indices, B_indices])
        #更新公式
        update_values = Q_tensor[C_indices, A_indices, B_indices] + alpha * (profit_matrix.view(-1)[mid] + gamma * max_values - Q_tensor[C_indices, A_indices, B_indices])
        # print(update_values)
        # 更新 type_t_matrix
        #更新Qtable
        Q_tensor[C_indices, A_indices, B_indices] = update_values
        return Q_tensor

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



    def calculation_value(self,r,type_t_matrix):
        with torch.no_grad():
            # 投入一次池子贡献1
            # value_matrix=(value_matrix-1)*(l_matrix+c_matrix)+value_matrix*d_matrix
            # 卷积每次博弈的合作＋r的人数
            # 获取原始张量的形状
            # 在第0行之前增加最后一行

            pad_tensor = self.pad_matrix(type_t_matrix)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(pad_tensor)
            coorperation_matrix = c_matrix .view(1, 1, L_num+2, L_num+2).to(torch.float32)
            # 下面这个卷积占了一轮的大部分时间约1秒钟，但是其他卷积都是一瞬间完成的，不知道为什么
            coorperation_num = torch.nn.functional.conv2d(coorperation_matrix, neibor_kernel,
                                                          bias=None, stride=1, padding=0).view(L_num,L_num).to(device)
            # c和r最后的-1是最开始要贡献到池里面的1
            c_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r - 1)

            d_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r)
            c_5_profit_matrix = torch.nn.functional.conv2d(c_profit_matrix.view(1, 1, L_num+2, L_num+2), neibor_kernel,
                                                           bias=None, stride=1, padding=0).to(torch.float32).to(device)
            d_5_profit_matrix = torch.nn.functional.conv2d(d_profit_matrix.view(1, 1, L_num+2, L_num+2), neibor_kernel,
                                                           bias=None, stride=1, padding=0).to(device)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(type_t_matrix)
            # 这里的k不是固定值，周围的player的k可能会有4顶点为3.
            profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
            return profit_matrix.view(L_num, L_num)


    # #一轮博弈只后策略的改变
    def type_matrix_change(self,epsilon,type_matrix: tensor, Q_matrix: tensor):
        type_t1_matrix= type_matrix.clone()
        indices = type_matrix.long().flatten()[mid]
        Q_probabilities = Q_matrix[mid, indices]
        # 在 Q_probabilities 中选择最大值索引
        # 找到每个概率分布的最大值
        if Q_probabilities[0] > Q_probabilities[1]:
            max_index = 0
        elif Q_probabilities[0] < Q_probabilities[1]:
            max_index = 1
        else:
            max_index = torch.randint(0, 2, (1,)).item()
        # 生成一个随机数
        random_number = torch.rand(1).item()
        # 如果随机数小于 epsilon，则随机选择一个动作
        if random_number < epsilon:
            updated_values = torch.randint(0, 2, (1,)).to(device)
        else:
            updated_values = torch.tensor(max_index).to(device)
        # 选择最大值的索引

        # 重新组织更新后的 tensor
        type_t1_matrix[1][1] = updated_values

        return type_t1_matrix

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
        return result_tensor.to(torch.float32).to("cpu")

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
        mean_of_positive_elements = (value_tensor.to(torch.float32).sum()) / ((positive_num + negetive_num).sum())
        return mean_of_positive_elements.to("cpu")

    #画折线图
    def draw_line_pic(self,obsX,D_Y,C_Y,xticks,yticks,r,ylim=(0,1),epoches=10000,type="line1",xlable='step',ylabel='fraction'):
        plt.clf()
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(obsX, D_Y, 'ro', label='betray', linestyle='-', linewidth=1, markeredgecolor='r', markersize=1,
                 markeredgewidth=1)
        plt.plot(obsX, C_Y, 'bo', label='cooperation', linestyle='-', linewidth=1, markeredgecolor='b',
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
    def draw_transfer_pic(self, obsX, CC_data, DD_data, CD_data, DC_data, xticks, yticks, r, ylim=(0, 1), epoches=10000):
        plt.clf()
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(obsX, DD_data, color='red',marker='o', label='DD', linestyle='-', linewidth=1, markeredgecolor='red', markersize=1,
                 markeredgewidth=1)
        plt.plot(obsX, CC_data, color='blue',marker='*', label='CC', linestyle='-', linewidth=1, markeredgecolor='blue',
                 markersize=1, markeredgewidth=1)
        plt.plot(obsX, CD_data, color='black',marker='o', label='CD', linestyle='-', linewidth=1, markeredgecolor='black', markersize=1,markeredgewidth=1)
        plt.plot(obsX, DC_data,color='green',marker='o', label='DC', linestyle='-', linewidth=1, markeredgecolor='green',
                 markersize=1, markeredgewidth=1)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.ylim(ylim)
        plt.xscale('log')
        plt.ylabel('fraction')
        plt.xlabel('step')
        plt.title('Q_learning:'+'L'+str(self.L_num)+' r='+str(r)+' T='+str(epoches))
        plt.legend()
        plt.pause(0.001)
        plt.clf()
        plt.close("all")

    #画快照
    def shot_pic(self,type_t_matrix: tensor):
        plt.clf()
        plt.close("all")
        # 初始化图表和数据
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)
        # 指定图的大小
        #             plt.figure(figsize=(500, 500))  # 10x10的图
        #             plt.matshow(type_t_matrix.cpu().numpy(), cmap=cmap)
        #             plt.colorbar(ticks=[0, 1, 2], label='Color')
        # 显示图片
        # 定义颜色映射
        color_map = {
            0: (255, 0, 0),  # 蓝色
            1: (0, 0, 255),  # 红色
        }
        image = np.zeros((L_num, L_num, 3), dtype=np.uint8)
        for label, color in color_map.items():
            image[type_t_matrix.cpu() == label] = color
        plt.imshow(image)
        plt.show()
        plt.clf()
        plt.close("all")

    #计算CD比例和利润
    def cal_fra_and_value(self,obsX, D_Y, C_Y, D_Value, C_Value, type_t_minus_matrix,type_t_matrix, d_matrix, c_matrix, profit_matrix,i):
        # 初始化图表和数据

        d_value = d_matrix * profit_matrix
        c_value = c_matrix * profit_matrix
        dmean_of_positive = self.c_mean_v2(d_value)
        cmean_of_positive = self.c_mean_v2(c_value)
        count_0 = torch.sum(type_t_matrix == 0).item()
        count_1 = torch.sum(type_t_matrix == 1).item()
        obsX = np.append(obsX, i + 1)
        D_Y = np.append(D_Y, count_0 / (L_num * L_num))
        C_Y = np.append(C_Y, count_1 / (L_num * L_num))
        D_Value = np.append(D_Value, dmean_of_positive)
        C_Value = np.append(C_Value, cmean_of_positive)
        CC, DD, CD, DC = self.cal_transfer_num(type_t_minus_matrix,type_t_matrix)
        return obsX, D_Y, C_Y, D_Value, C_Value, count_0, count_1, CC, DD, CD, DC

    #计算转移的比例
    def cal_transfer_num(self,type_t_matrix,type_t1_matrix):
        CC=(torch.where((type_t_matrix==1)&(type_t1_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        DD=(torch.where((type_t_matrix==0)&(type_t1_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        CD=(torch.where((type_t_matrix==1)&(type_t1_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        DC=(torch.where((type_t_matrix==0)&(type_t1_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        return CC,DD,CD,DC

    def extract_Qtable(self,Q_tensor, type_t_matrix):
        C_indices = torch.where(type_t_matrix.squeeze() == 1)[0]
        D_indices = torch.where(type_t_matrix.squeeze() == 0)[0]
        C_Q_table = Q_tensor[C_indices]
        D_indices = Q_tensor[D_indices]
        C_q_mean_matrix = torch.mean(C_Q_table, dim=0)
        D_q_mean_matrix = torch.mean(D_indices, dim=0)
        return D_q_mean_matrix.cpu().numpy(), C_q_mean_matrix.cpu().numpy()

    def run(self,type_t_matrix,Q_matrix,r,alpha,gamma,epsilon,epoches, L_num, device,type):
        # node= np.full((L_num,1),1)
        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000

        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
        # type_matrix=torch.tensor(node,dtype=torch.int).to(device)
        type_t_minus_matrix = torch.zeros((L_num, L_num), dtype=torch.float32).to(device)
        value_matrix = torch.tensor(L, dtype=torch.float32).to(device)
        # Q = np.zeros((L_num * L_num, 2, 2))
        # Q_matrix = torch.tensor(Q, dtype=torch.float32).to(device)
        count_0=torch.where(type_t_matrix == 0, torch.tensor(1), torch.tensor(0)).sum().item()/ (L_num * L_num)
        count_1=1-count_0


        Qtable_Loop=[]
        Qtable_Loop.append(Q_matrix[mid].cpu().numpy())
        type_t_matrix_Loop=[]
        type_t_matrix_Loop.append(type_t_matrix.cpu().numpy())
        obsX = np.array([])
        D_Y = np.array([])
        C_Y = np.array([])
        C_state= np.array([])
        D_Value = np.array([])
        C_Value = np.array([])
        CC_data,DD_data,CD_data,DC_data=np.array([]),np.array([]),np.array([]),np.array([])
        D_Y= np.append(D_Y, count_0 )
        C_Y = np.append(C_Y, count_1 )
        self.shot_pic(type_t_matrix)
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
            if i != 0:
                # Q策略更新
                Q_matrix = self.updateQMatrix2(alpha,gamma,type_t_minus_matrix, type_t_matrix, Q_matrix, profit_matrix)

            # 博弈演化,type变换，策略传播
            type_t1_matrix = self.type_matrix_change(epsilon,type_t_matrix, Q_matrix).to(device)
            # 把一个L的三个type分开
            d_matrix, c_matrix = self.type_matrix_to_three_matrix(type_t1_matrix)

            type_t_minus_matrix = type_t_matrix
            type_t_matrix = type_t1_matrix

            C_Value=np.append(C_Value,profit_matrix[1][1].item())
            Qtable_Loop.append(Q_matrix[mid].cpu().numpy())
            type_t_matrix_Loop.append(type_t_matrix.cpu().numpy())
            # print("Q_matrix[mid]:")
            # print(Q_matrix[mid])
            # obsX, D_Y, C_Y, D_Value, C_Value, count_0, count_1, CC, DD, CD, DC = self.cal_fra_and_value(obsX, D_Y, C_Y,D_Value,C_Value,type_t_minus_matrix,type_t_matrix,d_matrix,c_matrix,profit_matrix,i)
            # CC_data = np.append(CC_data, CC)
            # DD_data = np.append(DD_data, DD)
            # CD_data = np.append(CD_data, CD)
            # DC_data = np.append(DC_data, DC)
            #self.shot_pic(type_t_matrix)
        Qtable_Loop=np.array(Qtable_Loop)
        type_t_matrix_Loop=np.array(type_t_matrix_Loop)
        # print("C_Value")
        # print(C_Value)
        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000
        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
        if (type == "line2"):
            return D_Y[-1], C_Y[-1]
        elif (type == "line1"):
            return D_Y, C_Y, D_Value, C_Value, Q_matrix, type_t_matrix, count_0, count_1, obsX, CC_data, DD_data, CD_data, DC_data
        elif (type == "Qtable"):
            return Qtable_Loop,type_t_matrix_Loop,C_Value

    def mkdir(self,path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

    def save_data(self,type,name,r,count,data):
        self.mkdir('data/Origin_Qlearning/'+str(type))
        try:
            np.savetxt('data/Origin_Qlearning/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(type), name, str(r),str(self.epoches),str(self.L_num), str(count)),data)
        except:
            print("Save failed")

    def run_line2_pic(self,loop_num1=50,loop_num2 = 10):
        r=0
        for j in range(loop_num1):
            for i in range(loop_num2):
                r1=r/10
                print("loop_num1: "+str(j)+" loop_num2: "+str(i)+" r="+str(r1))
                D_Y, C_Y, D_Value, C_Value, Q_matrix, type_t_matrix, count_0, count_1, obsX, CC_data, DD_data, CD_data, DC_data = self.run(
                    r1, self.alpha, self.gamma, self.epsilon, self.epoches, self.L_num, self.device, type="line1")
                self.save_data('C_fra', 'C_fra',r1, i, C_Y)
                self.save_data('D_fra', 'D_fra',r1, i, D_Y)
                self.save_data('C_value', 'C_value',r1, i, C_Value)
                self.save_data('D_value', 'D_value',r1, i, D_Value)
                self.save_data('CC_fra', 'CC_fra',r1, i, CC_data)
                self.save_data('DD_fra', 'DD_fra',r1, i, DD_data)
                self.save_data('CD_fra', 'CD_fra',r1, i, CD_data)
                self.save_data('DC_fra', 'DC_fra',r1, i, DC_data)
            r=r+1

    def cal_transfer_pic(self):
        CC_data, DD_data, CD_data, DC_data=self.run(self.r, self.alpha,self.gamma,self.epsilon,self.epoches, self.L_num,self.device,type="line1")
        self.draw_transfer_pic(np.arange(self.epoches), CC_data, DD_data,CD_data,DC_data, xticks, fra_yticks,r=self.r,epoches=self.epoches)

    def extra_Q_table(self,loop_num):
        for i in range(loop_num):
            Q_matrix,type_t_matrix = self.run(self.r, self.alpha,self.gamma,self.epsilon,self.epoches, self.L_num,self.device,type="Qtable")
            D_q_mean_matrix, C_q_mean_matrix = self.extract_Qtable(Q_matrix, type_t_matrix)
            self.save_data('D_Qtable', 'D_Qtable',self.r, str(i), D_q_mean_matrix)
            self.save_data('C_Qtable', 'C_Qtable',self.r, str(i), C_q_mean_matrix)

    def read_data(self,path):
        data = np.loadtxt(path)
        return data

    def read_Qtable(self,updateMethod,type,name,r,count):
        final_Qtable=np.zeros((2,2))
        for i in range(count):
            data = self.read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),str(type), name, str(r),str(10000),str(200), str(i)))
            final_Qtable=final_Qtable+data
        final_Qtable=final_Qtable/count
        return final_Qtable

    def mock_C_env(self):
        type_t_matrix=torch.zeros((self.L_num,self.L_num),dtype=torch.float32).to(self.device)
        type_t_matrix[1][1]=1
        type_t_matrix[0][1]=1
        type_t_matrix[1][0]=1
        type_t_matrix[1][2]=1
        C_Qtable=self.read_Qtable('Origin_Qlearning','C_Qtable','C_Qtable',self.r,10)
        print(C_Qtable)
        Q_matrix= torch.zeros(L_num*L_num, 2, 2).to(torch.float32).to(self.device)
        Q_matrix[mid]=torch.tensor(C_Qtable,dtype=torch.float32).to(self.device)
        Qtable_Loop,type_t_matrix_Loop,C_Value=self.run(type_t_matrix,Q_matrix,self.r, self.alpha,self.gamma,self.epsilon,self.epoches, self.L_num,self.device,type="Qtable")
        self.draw_line_pic(np.arange(self.epoches+1), type_t_matrix_Loop[:,1,1], type_t_matrix_Loop[:,1,1], xticks, [-1,0,1,2], r=self.r, epoches=self.epoches, type="line1", ylabel='state', xlable='step',ylim=(-1,2))
        self.draw_line_pic(np.arange(self.epoches), C_Value, C_Value, Q_xticks, profite_yticks, r=self.r, epoches=self.epoches, type="line1", ylabel='value', xlable='step',ylim=(-3,7))

        Qtable_CC=Qtable_Loop[:,1,1]
        Qtable_DD=Qtable_Loop[:,0,0]
        Qtable_CD=Qtable_Loop[:,1,0]
        Qtable_DC=Qtable_Loop[:,0,1]
        # print(Qtable_Loop[-1])
        self.draw_transfer_pic(np.arange(self.epoches+1), Qtable_CC, Qtable_DD, Qtable_CD, Qtable_DC, Q_xticks, Q_yticks, r=self.r, epoches=self.epoches,ylim=(-10,60))


    def mock_D_env(self):
        type_t_matrix=torch.zeros((self.L_num,self.L_num),dtype=torch.float32).to(self.device)
        type_t_matrix[1][1]=0
        D_Qtable=self.read_Qtable('Origin_Qlearning','D_Qtable','D_Qtable',self.r,0)
        Q_matrix=torch.tensor((self.L_num*self.L_num,2,2),dtype=torch.float32).to(self.device)
        Q_matrix[mid]=torch.tensor(D_Qtable,dtype=torch.float32).to(self.device)
        D_Y, C_Y, D_Value, C_Value, Q_matrix, type_t_matrix, count_0, count_1, obsX, CC_data, DD_data, CD_data, DC_data=self.run(type_t_matrix,Q_matrix,self.r, self.alpha,self.gamma,self.epsilon,self.epoches, self.L_num,self.device,type="Qtable")
        self.draw_line_pic(obsX, D_Y, C_Y, xticks, fra_yticks, r=self.r, epoches=self.epoches, type="line1", ylabel='fraction', xlable='step')
        self.draw_line_pic(obsX, D_Value, C_Value, xticks, profite_yticks, r=self.r, epoches=self.epoches, type="line1", ylabel='fraction', xlable='step')


    def draw_transfer_pic(self,obsX, CC_data, DD_data, CD_data, DC_data, xticks, yticks, r, ylim=(0, 1), epoches=10000):
        plt.clf()
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(obsX, DD_data, color='red', marker='o', label='DD', linestyle='-', linewidth=1, markeredgecolor='red',
                 markersize=1,
                 markeredgewidth=1)
        plt.plot(obsX, CC_data, color='blue', marker='*', label='CC', linestyle='-', linewidth=1,
                 markeredgecolor='blue',
                 markersize=1, markeredgewidth=1)
        plt.plot(obsX, CD_data, color='black', marker='o', label='CD', linestyle='-', linewidth=1,
                 markeredgecolor='black', markersize=1, markeredgewidth=1)
        plt.plot(obsX, DC_data, color='gold', marker='o', label='DC', linestyle='-', linewidth=1,
                 markeredgecolor='gold',
                 markersize=1, markeredgewidth=1)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.ylim(ylim)
        plt.xscale('log')
        plt.ylabel('fraction')
        plt.xlabel('step')
        plt.title('Q_learning:' + 'L' + str(L_num) + ' r=' + str(r) + ' T=' + str(epoches))
        plt.legend()
        plt.pause(0.001)
        plt.clf()
        plt.close("all")


if __name__ == '__main__':
    SPGG=SPGG_Qlearning(L_num,device,alpha,gamma,epsilon,r=4.9,epoches=1000,cal_transfer=True)
    SPGG.mock_C_env()
    # SPGG.run_line2_pic(loop_num1=50,loop_num2 = 10)
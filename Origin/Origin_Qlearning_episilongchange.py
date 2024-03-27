import torch

from torch import tensor
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
#变化的epsilon对Qlearning的影响
L_num=200
torch.cuda.set_device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
alpha=0.8
gamma=0.8
# epsilon=0.02
neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float32).to(device).view(1,1,3,3)
actions = torch.tensor([0, 1],dtype=torch.float32).to(device)
L = np.full((L_num, L_num), 0)
value_matrix = torch.tensor(L, dtype=torch.float32).to(device)

zeros_tensor = torch.zeros((1, 1, L_num, L_num),dtype=torch.float32).to(torch.float32)
g_matrix=torch.nn.functional.conv2d(torch.ones((1,1,L_num, L_num),dtype=torch.float32).to(device), neibor_kernel,
                                                      bias=None, stride=1, padding=1).to(device)
xticks=[0, 10, 100, 1000, 10000, 100000]
fra_yticks=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95, 1.00]
profite_yticks=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

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

    def updateQMatrix(self,alpha,gamma,type_t_matrix: tensor, type_t1_matrix: tensor, Q_tensor: tensor, profit_matrix: tensor):
        C_indices = torch.arange(type_t_matrix.numel()).to(device)
        A_indices = type_t_matrix.view(-1).long()
        B_indices = type_t1_matrix.view(-1).long()
        # 计算更新值
        max_values, _ = torch.max(Q_tensor[C_indices, B_indices], dim=1)
        update_values = Q_tensor[C_indices, A_indices, B_indices] + alpha * (profit_matrix.view(-1) + gamma * max_values - Q_tensor[C_indices, A_indices, B_indices])
        # print(update_values)
        # 更新 type_t_matrix
        Q_tensor[C_indices, A_indices, B_indices] = update_values
        return Q_tensor


    # def calculation_value(self,r,d_matrix, c_matrix):
    #     with torch.no_grad():
    #         # 投入一次池子贡献1
    #         # value_matrix=(value_matrix-1)*(l_matrix+c_matrix)+value_matrix*d_matrix
    #         # 卷积每次博弈的合作＋r的人数
    #         coorperation_matrix = c_matrix .view(1, 1, self.L_num, self.L_num).to(torch.float32)
    #         # 下面这个卷积占了一轮的大部分时间约1秒钟，但是其他卷积都是一瞬间完成的，不知道为什么
    #         coorperation_num = torch.nn.functional.conv2d(coorperation_matrix, neibor_kernel,
    #                                                       bias=None, stride=1, padding=1).to(device)
    #         # c和r最后的-1是最开始要贡献到池里面的1
    #         c_profit_matrix = (coorperation_num) / g_matrix * r - 1
    #
    #         d_profit_matrix = (coorperation_num) / g_matrix * r
    #
    #         c_5_profit_matrix = torch.nn.functional.conv2d(c_profit_matrix.view(1, 1, L_num, L_num), neibor_kernel,
    #                                                        bias=None, stride=1, padding=1).to(torch.float32).to(device)
    #         d_5_profit_matrix = torch.nn.functional.conv2d(d_profit_matrix.view(1, 1, L_num, L_num), neibor_kernel,
    #                                                        bias=None, stride=1, padding=1).to(device)
    #         # 这里的k不是固定值，周围的player的k可能会有4顶点为3.
    #         profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
    #         return profit_matrix
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

    # def calculation_value(self, r, d_matrix, c_matrix):
    #     with torch.no_grad():
    #         # 卷积每次博弈的合作＋r的人数
    #         coorperation_matrix = c_matrix.view(1, 1, self.L_num, self.L_num).to(torch.float32)
    #         # 下面这个卷积占了一轮的大部分时间约1秒钟，但是其他卷积都是一瞬间完成的，不知道为什么
    #
    #         coorperation_num = torch.nn.functional.conv2d(coorperation_matrix, neibor_kernel,
    #                                                       bias=None, stride=1, padding=1).to(device)
    #         # c和r最后的-1是最开始要贡献到池里面的1
    #         c_profit_matrix = (coorperation_num) / 5 * r - 1
    #         d_profit_matrix = (coorperation_num) / 5 * r
    #         c_5_profit_matrix = torch.nn.functional.conv2d(c_profit_matrix.view(1, 1, L_num, L_num), neibor_kernel,
    #                                                        bias=None, stride=1, padding=1).to(torch.float32).to(
    #             device)
    #         d_5_profit_matrix = torch.nn.functional.conv2d(d_profit_matrix.view(1, 1, L_num, L_num), neibor_kernel,
    #                                                        bias=None, stride=1, padding=1).to(device)
    #         # 这里的k不是固定值，周围的player的k可能会有4顶点为3.
    #         profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
    #         return profit_matrix

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
        mask = (torch.rand(L_num, L_num) > epsilon).long().to(device)

        # 使用 mask 来决定更新的值
        # updated_values = mask.flatten().unsqueeze(1) * random_max_indices.unsqueeze(1) + (
        #         1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)
        updated_values = mask.flatten().unsqueeze(1) * indices.unsqueeze(1) + (1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)

        # 重新组织更新后的 tensor
        updated_tensor = updated_values.view(L_num, L_num).to(device)

        return updated_tensor


    def type_matrix_to_three_matrix(self,type_matrix: tensor):
        # 初始化一个新的张量，其中数值为0的值设为1，为1和2的值设为0
        d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(device)
        c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(device)
        return d_matrix, c_matrix


    def generated_default_type_matrix(self):
        probabilities = torch.tensor([1 /2, 1 / 2])

        # 生成一个随机张量，其数值分别为0，1，2，根据指定的概率分布
        result_tensor = torch.multinomial(probabilities, L_num * L_num, replacement=True)
        result_tensor = result_tensor.view(L_num, L_num)
        return result_tensor.to(torch.float32).to("cpu")


    def c_mean_v(self,value_tensor):
        positive_values = value_tensor[value_tensor > 0.0]
        # 计算大于零的值的平均值
        mean_of_positive = torch.mean(positive_values)
        return mean_of_positive.item() + 1


    def c_mean_v2(self,value_tensor):
        # 创建布尔张量，表示大于零的元素
        positive_num = (value_tensor > 0).to(device)
        negetive_num = (value_tensor < 0).to(device)
        # 计算大于零的元素的均值
        mean_of_positive_elements = (value_tensor.to(torch.float32).sum()) / ((positive_num + negetive_num).sum())
        return mean_of_positive_elements.to("cpu")

    def draw_line_pic(self,obsX,D_Y,C_Y,xticks,yticks,r,ylim=(0,1),epoches=10000,type="line1",xlable='step',ylabel='fractions'):
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
        plt.title('Q_learning:'+'L='+str(self.L_num)+' r='+str(r)+' n_iter='+str(epoches))
        plt.pause(0.001)
        plt.clf()
        plt.close("all")


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
        plt.plot(obsX, DC_data,color='gold',marker='o', label='DC', linestyle='-', linewidth=1, markeredgecolor='gold',
                 markersize=1, markeredgewidth=1)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.ylim(ylim)
        plt.xscale('log')
        plt.ylabel('fractions')
        plt.xlabel('step')
        plt.title('Q_learning:'+'L'+str(self.L_num)+' r='+str(r)+' n_iter='+str(epoches))
        plt.legend()
        plt.pause(0.001)
        plt.clf()
        plt.close("all")

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
        image = np.zeros((type_t_matrix.size(0), type_t_matrix.size(1), 3), dtype=np.uint8)
        for label, color in color_map.items():
            image[type_t_matrix.cpu() == label] = color
        plt.imshow(image)
        plt.show()
        plt.clf()
        plt.close("all")

    def cal_transfer_num(self,type_t_minus_matrix,type_t_matrix):
        CC=(torch.where((type_t_minus_matrix==1)&(type_t_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        DD=(torch.where((type_t_minus_matrix==0)&(type_t_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        CD=(torch.where((type_t_minus_matrix==1)&(type_t_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        DC=(torch.where((type_t_minus_matrix==0)&(type_t_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        return CC,DD,CD,DC

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



    def run(self,r,alpha,gamma,epsilon,epoches, L_num, device,type):
        # node= np.full((L_num,1),1)
        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000

        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
        # type_matrix=torch.tensor(node,dtype=torch.int).to(device)
        type_t_matrix = self.generated_default_type_matrix().to(device)
        type_t_minus_matrix = torch.zeros((L_num, L_num), dtype=torch.float32).to(device)
        value_matrix = torch.tensor(L, dtype=torch.float32).to(device)
        Q = np.zeros((L_num * L_num, 2, 2))
        Q_matrix = torch.tensor(Q, dtype=torch.float32).to(device)
        count_0=torch.where(type_t_matrix == 0, torch.tensor(1), torch.tensor(0)).sum().item()/ (L_num * L_num)
        count_1=1-count_0


        obsX = np.array([])
        D_Y = np.array([])
        C_Y = np.array([])
        D_Value = np.array([])
        C_Value = np.array([])
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
            if i != 0:
                # Q策略更新
                Q_matrix = self.updateQMatrix(alpha,gamma,type_t_minus_matrix, type_t_matrix, Q_matrix, profit_matrix)
            if (1-(i+20)/100)<0.02:
                epsilon=0.02
            else:
                epsilon=1-(i+20)/100
            # 博弈演化,type变换，策略传播
            type_t1_matrix = self.type_matrix_change(epsilon,type_t_matrix, Q_matrix).to(device)
            # 把一个L的三个type分开
            d_matrix, c_matrix = self.type_matrix_to_three_matrix(type_t1_matrix)

            type_t_minus_matrix = type_t_matrix
            type_t_matrix = type_t1_matrix

            obsX, D_Y, C_Y, D_Value, C_Value, count_0, count_1,CC,DD,CD,DC = self.cal_fra_and_value(obsX, D_Y, C_Y, D_Value, C_Value, type_t_minus_matrix,type_t_matrix, d_matrix, c_matrix, profit_matrix,i)
            CC_data = np.append(CC_data, CC)
            DD_data = np.append(DD_data, DD)
            CD_data = np.append(CD_data, CD)
            DC_data = np.append(DC_data, DC)

        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000
        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
        if(type=="line2"):
            print(D_Y[-1],C_Y[-1])
            return D_Y[-1],C_Y[-1]
        elif(type=="line1"):
            return D_Y,C_Y,D_Value,C_Value,Q_matrix,type_t_matrix,count_0,count_1,obsX,CC_data,DD_data,CD_data,DC_data

    def mkdir(self,path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

    def save_data(self,type,name,r,count,data):
        self.mkdir('data/Origin_Qlearning_epsilongchange/'+str(type))
        try:
            np.savetxt('data/Origin_Fermi/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(type), name, str(r), str(self.epoches), str(self.L_num),str(count)), data)
            # np.savetxt(str(type)+'/'+name + '第' +" epoches=10000 L=200"+ str(count) + '次实验数据.txt',
            #            C_data)
        except:
            print("Save failed")

    def run_line2_pic(self,loop_num1=50,loop_num2 = 10):
        r=0
        for j in range(loop_num1):
            for i in range(loop_num2):
                r1=r/10
                print("loop_num1: "+str(j)+" loop_num2: "+str(i)+"r="+str(r1))
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
            r=r+0.1

if __name__ == '__main__':
    SPGG=SPGG_Qlearning(L_num,device,alpha,gamma,epsilon=0.02,r=4,epoches=10000,cal_transfer=True)
    SPGG.run_line2_pic(loop_num1=50,loop_num2 = 10)

#def line2_pic(self, alpha,gamma,epsilon,loop_num1=10,loop_num2 = 50):
    #     D_Final_fra = np.zeros(loop_num2)
    #     C_Final_fra = np.zeros(loop_num2)
    #     for j in range(loop_num1):
    #         r=0
    #         D_Loop_fra=np.array([])
    #         C_Loop_fra=np.array([])
    #         for i in range(loop_num2):
    #             print("loop_num1: "+str(j)+" loop_num2: "+str(i))
    #             r=r+0.1
    #             D_Y,C_Y=self.run(r, alpha,gamma,epsilon,self.epoches, self.L_num,self.device,type="line2")
    #             D_Loop_fra=np.append(D_Loop_fra,D_Y)
    #             C_Loop_fra=np.append(C_Loop_fra,C_Y)
    #         D_Final_fra=D_Final_fra+D_Loop_fra
    #         C_Final_fra=C_Final_fra+C_Loop_fra
    #         print(D_Loop_fra)
    #         print(C_Loop_fra)
    #         print(D_Final_fra)
    #         print(C_Final_fra)
    #     D_Final_fra=D_Final_fra/loop_num1
    #     C_Final_fra=C_Final_fra/loop_num1
    #     self.draw_line_pic(np.arange(loop_num2)/10, D_Final_fra, C_Final_fra, np.arange(loop_num2/10), fra_yticks,r='0-5',epoches=self.epoches,type="line2",ylabel='fractions',xlable='r')
    #     self.save_data(1, C_Final_fra, D_Final_fra, "r=0-5 epoches=10000 L=200")
    #
    # def line1_pic(self,r):
    #     epoches=self.epoches
    #     D_Y_ave, C_Y_ave, D_Value_ave, C_Value_ave, count_0_ave, \
    #     count_1_ave, CC_data_ave, DD_data_ave, CD_data_ave, DC_data_ave = \
    #         np.zeros(epoches+1), np.zeros(epoches+1), np.zeros(epoches), np.zeros(epoches), \
    #         0, 0, np.zeros(epoches), np.zeros(epoches),np.zeros(epoches), np.zeros(epoches)
    #     Q_matrix_ave=torch.zeros((L_num * L_num, 2, 2)).to(device)
    #     loop_num=10
    #     for i in range(loop_num):
    #         D_Y, C_Y, D_Value, C_Value, Q_matrix, type_t_matrix, count_0, count_1, obsX, CC_data, DD_data, CD_data, DC_data = self.run(
    #             self.r, self.alpha, self.gamma, self.epsilon, self.epoches, self.L_num, self.device, type="line1")
    #         D_Y_ave = D_Y_ave + D_Y
    #         C_Y_ave = C_Y_ave + C_Y
    #         D_Value_ave = D_Value_ave + D_Value
    #         C_Value_ave = C_Value_ave + C_Value
    #         Q_matrix_ave = Q_matrix_ave + Q_matrix
    #         count_0_ave = count_0_ave + count_0
    #         count_1_ave = count_1_ave + count_1
    #         CC_data_ave = CC_data_ave + CC_data
    #         DD_data_ave = DD_data_ave + DD_data
    #         CD_data_ave = CD_data_ave + CD_data
    #         DC_data_ave = DC_data_ave + DC_data
    #     D_Y_ave = D_Y_ave / loop_num
    #     C_Y_ave = C_Y_ave / loop_num
    #     D_Value_ave = D_Value_ave / loop_num
    #     C_Value_ave = C_Value_ave / loop_num
    #     Q_matrix_ave = Q_matrix_ave / loop_num
    #     count_0_ave = count_0_ave / loop_num
    #     count_1_ave = count_1_ave / loop_num
    #     CC_data_ave = CC_data_ave / loop_num
    #     DD_data_ave = DD_data_ave / loop_num
    #     CD_data_ave = CD_data_ave / loop_num
    #     DC_data_ave = DC_data_ave / loop_num
    #
    #     q_mean_matrix = torch.mean(Q_matrix_ave, dim=0)
    #     print(q_mean_matrix)
    #
    #     self.draw_line_pic(np.arange(epoches+1), D_Y_ave, C_Y_ave, xticks, fra_yticks, r=r, epoches=epoches)
    #     self.draw_line_pic(obsX, D_Value_ave, C_Value_ave, xticks, profite_yticks, ylim=(0, 15), r=r, epoches=epoches,
    #                        xlable='step', ylabel='value')
    #     if (self.cal_transfer == True):
    #         self.draw_transfer_pic(obsX, CC_data_ave, DD_data_ave, CD_data_ave, DC_data_ave, xticks,
    #                                r=self.r, epoches=self.epoches)
    #     self.shot_pic(type_t_matrix)
    #     print(count_0_ave / (L_num * L_num))
    #     print(count_1_ave / (L_num * L_num))
    #     print(D_Value_ave[-1])
    #     print(C_Value_ave[-1])

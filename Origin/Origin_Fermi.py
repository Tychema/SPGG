import torch

from torch import tensor
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap

# 定义两个线性分段的颜色映射
colors = [(48, 83, 133), (218, 160, 90), (253, 243, 197)]  # R -> G -> B
colors = [(color[0] / 255, color[1] / 255, color[2] / 255) for color in colors]
cmap_mma = LinearSegmentedColormap.from_list("mma", colors, N=256)

# 定义另一个颜色映射
colors = ["#eeeeee", "#111111", "#787ac0", ]
cmap = mpl.colors.ListedColormap(colors, N=3)

epoches=10000
L_num=200
torch.cuda.set_device("cuda:4" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float32).to(device).view(1,1,3,3)
actions = torch.tensor([0, 1],dtype=torch.float32).to(device)
L = np.full((L_num, L_num), 0)
value_matrix = torch.tensor(L, dtype=torch.float32).to(device)
zeros_tensor = torch.zeros((1, 1, L_num, L_num),dtype=torch.float32).to(torch.float32)
xticks=[0, 10, 100, 1000, 10000, 100000]
fra_yticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,0.95, 1,1.05]
profite_yticks=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,17,20]

class SPGG_Fermi(nn.Module):
    def __init__(self,epoches,L_num,device,r,K=0.1,count=0,cal_transfer=False):
        super(SPGG_Fermi, self).__init__()
        self.epoches=epoches
        self.L_num=L_num
        self.device=device
        self.r=r
        self.neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float32).to(device).view(1,1,3,3)
        self.K=K
        self.cal_transfer=cal_transfer
        self.count=count


    def generated_default_type_matrix(self,L_num):
        probabilities = torch.tensor([1 / 2, 1 / 2])

        # 生成一个随机张量，其数值分别为0，1，2，根据指定的概率分布
        result_tensor = torch.multinomial(probabilities, L_num * L_num, replacement=True)
        result_tensor = result_tensor.view(L_num, L_num)
        return result_tensor.to(torch.float32).to(device)

    def generated_default_type_matrix2(self,L_num):
        tensor = torch.zeros(L_num, L_num)
        # 计算上半部分和下半部分的分界线（中间行）
        mid_row = L_num // 2
        # 将下半部分的元素设置为1
        tensor[mid_row:, :] = 1
        return tensor


    def profit_Matrix_to_Four_Matrix(self,profit_matrix,K):
        W_left=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,1))/K))
        W_right=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,1))/K))
        W_up=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,0))/K))
        W_down=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,0))/K))
        return W_left,W_right,W_up,W_down

    def fermiUpdate(self,type_t_matrix,profit_matrix,K):
        #计算费米更新的概率
        W_left,W_right,W_up,W_down=self.profit_Matrix_to_Four_Matrix(profit_matrix,0.1)
        #生成一个矩阵随机决定向哪个方向学习
        learning_direction=torch.randint(0,4,(L_num,L_num)).to(device)
        #生成一个随机矩阵决定是否向他学习
        learning_probabilities=torch.rand(L_num,L_num).to(device)
        #费米更新
        type_t1_matrix=(learning_direction==0)*((learning_probabilities<=W_left)*torch.roll(type_t_matrix,1,1)+(learning_probabilities>W_left)*type_t_matrix) +\
                          (learning_direction==1)*((learning_probabilities<=W_right)*torch.roll(type_t_matrix,-1,1)+(learning_probabilities>W_right)*type_t_matrix) +\
                            (learning_direction==2)*((learning_probabilities<=W_up)*torch.roll(type_t_matrix,1,0)+(learning_probabilities>W_up)*type_t_matrix) +\
                                (learning_direction==3)*((learning_probabilities<=W_down)*torch.roll(type_t_matrix,-1,0)+(learning_probabilities>W_down)*type_t_matrix)
        return type_t1_matrix.view(L_num,L_num)

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



    def type_matrix_to_three_matrix(self,type_matrix: tensor):
        # 初始化一个新的张量，其中数值为0的值设为1，为1和2的值设为0
        d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(device)
        c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(device)
        return d_matrix, c_matrix


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

    def draw_line_pic(self, D_Y, C_Y, xticks, yticks, r, epoches,ylim=(0, 1.05), type="line1",xlabel="T",ylabel="fraction"):
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
        if (type == "line1"):
            plt.xscale('log')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title('Fermi:'+'L='+str(self.L_num)+' r='+str(r)+' T='+str(epoches))
        plt.legend()
        plt.pause(0.001)
        plt.clf()
        plt.close("all")

    def shot_pic(self,type_t_matrix: tensor,i,r):
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
            #0设置为黑色
            0: (0, 0, 0),  # 黑色
            #1设置为白色
            1: (255, 255, 255),  # 白色
        }
        image = np.zeros((L_num, L_num, 3), dtype=np.uint8)
        for label, color in color_map.items():
            image[type_t_matrix.cpu() == label] = color
        plt.title('Fermi: '+'T='+str(i))
        plt.imshow(image,interpolation='None')
        self.mkdir('data/Origin_Fermi/shot_pic/r={}/generated1'.format(r))
        plt.savefig('data/Origin_Fermi/shot_pic/r={}/generated1/t={}.png'.format(r,i))
        self.mkdir('data/Origin_Fermi/shot_pic/r={}/generated1/type_t_matrix'.format(r))
        np.savetxt('data/Origin_Fermi/shot_pic/r={}/generated1/type_t_matrix/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(r),"type_t_matrix",str(r),str(self.epoches),str(self.L_num),str(self.count)),type_t_matrix.cpu().numpy())
        #plt.show()
        plt.clf()
        plt.close("all")

    def draw_transfer_pic(self, CC_data, DD_data, CD_data, DC_data, xticks, yticks, r, ylim=(0, 1), epoches=10000):
        plt.clf()
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
        plt.title('Fermi:'+'L'+str(self.L_num)+' r='+str(r)+' T='+str(epoches))
        plt.legend()
        plt.pause(0.001)
        plt.clf()
        plt.close("all")

    def cal_fra_and_value(self, D_Y, C_Y, D_Value, C_Value,all_value, type_t_matrix, d_matrix, c_matrix, profit_matrix,i):
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
        all_value = np.append(all_value,profit_matrix.sum().item())
        return  D_Y, C_Y, D_Value, C_Value, count_0, count_1, all_value

    def cal_transfer_num(self,type_t_matrix,type_t1_matrix):
        CC=(torch.where((type_t_matrix==1)&(type_t1_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        DD=(torch.where((type_t_matrix==0)&(type_t1_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        CD=(torch.where((type_t_matrix==1)&(type_t1_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        DC=(torch.where((type_t_matrix==0)&(type_t1_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        return CC,DD,CD,DC

    def run(self,r,type):
        epoches =self.epoches
        L_num=self.L_num
        device=self.device
        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000
        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")


        type_t_matrix = self.generated_default_type_matrix(L_num).to(device)
        value_matrix = torch.tensor(L, dtype=torch.float32).to(device)
        count_0=torch.where(type_t_matrix == 0, torch.tensor(1), torch.tensor(0)).sum().item()/ (L_num * L_num)
        count_1=1-count_0


        D_Y = np.array([])
        C_Y = np.array([])
        D_Value = np.array([])
        C_Value = np.array([])
        all_value=np.array([])
        D_Y= np.append(D_Y, count_0 )
        C_Y = np.append(C_Y, count_1 )

        CC_data,DD_data,CD_data,DC_data=np.array([]),np.array([]),np.array([]),np.array([])
        self.shot_pic(type_t_matrix, 0,r)
        for i in tqdm(range(epoches), desc='Processing'):
            # type_file_name = f'type\\type_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
            # Q_file_name = f'Q\\Q_{i + 1}.pt'  # 这里使用了 i+1 作为文件名
            # V_file_name = f'V\\V_{i + 1}.pt'  # 这里使用了 i+1 作为文件名

            # 计算此次博弈利润的结果
            profit_matrix = self.calculation_value(r,type_t_matrix)
            # 计算得到的价值
            value_matrix = value_matrix + profit_matrix
            #费米更新
            type_t1_matrix=self.fermiUpdate(type_t_matrix,profit_matrix,self.K)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(type_t1_matrix)
            #快照
            if i==0  or i==9 or i==49 or i==99 or i==299 or i==499 or i==799 or i==999 or i==4999 or i==9999 or i==19999:
                self.shot_pic(type_t_matrix,i+1,r)
            #计算比例转换
            if self.cal_transfer == True:
                CC, DD, CD, DC = self.cal_transfer_num(type_t_matrix, type_t1_matrix)
                CC_data = np.append(CC_data, CC)
                DD_data = np.append(DD_data, DD)
                CD_data = np.append(CD_data, CD)
                DC_data = np.append(DC_data, DC)
            elif type == "line1" or type == "line2":
                 D_Y, C_Y, D_Value, C_Value, count_0, count_1,all_value = self.cal_fra_and_value( D_Y, C_Y, D_Value, C_Value,all_value,
                                                                                   type_t1_matrix, d_matrix, c_matrix,
                                                                                   profit_matrix,i)
            type_t_matrix = type_t1_matrix
        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000
        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")

        if self.cal_transfer == True:
            return CC_data, DD_data, CD_data, DC_data
        elif (type == "line2"):
            return D_Y[-1], C_Y[-1]
        elif (type == "line1"):
            return D_Y, C_Y, D_Value, C_Value,type_t_matrix, count_0, count_1, all_value

    def mkdir(self,path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

    def save_data(self,type,name,r,count,data):
        self.mkdir('data/Origin_Fermi/generated1/'+str(type))
        try:
            np.savetxt('data/Origin_Fermi/generated1/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(type), name, str(r), str(self.epoches), str(self.L_num),str(count)), data)
            # np.savetxt(str(type)+'/'+name + '第' +" epoches=10000 L=200"+ str(count) + '次实验数据.txt',
            #            C_data)
        except:
            print("Save failed")

    def run_line2_pic(self, loop_num1=51, loop_num2=10):
        r = 0
        for j in range(loop_num1):
            for i in range(loop_num2):
                r1=r/10
                print("loop_num1: " + str(j) + " loop_num2: " + str(i) + "r=" + str(r1))
                D_Y, C_Y, D_Value, C_Value, type_t_matrix, count_0, count_1, all_value= self.run(r1, type="line1")
                self.save_data('C_fra', 'C_fra', r1, i, C_Y)
                self.save_data('D_fra', 'D_fra', r1, i, D_Y)
                self.save_data('C_value', 'C_value', r1, i, C_Value)
                self.save_data('D_value', 'D_value', r1, i, D_Value)
                self.save_data('all_value', 'all_value', r1, i, all_value)
            r = r + 1

    def line2_pic(self, loop_num1=10, loop_num2=51):
        D_Final_fra = np.zeros(loop_num2)
        C_Final_fra = np.array(loop_num2)
        for j in range(loop_num1):
            r = 0
            D_Loop_fra = np.array([])
            C_Loop_fra = np.array([])
            for i in range(loop_num2):
                print("loop_num1: " + str(j) + " loop_num2: " + str(i))
                r = r + 0.1
                D_Y, C_Y = self.run(r, type="line2")
                print(D_Y, C_Y)
                D_Loop_fra = np.append(D_Loop_fra, D_Y)
                C_Loop_fra = np.append(C_Loop_fra, C_Y)
            D_Final_fra = D_Final_fra + D_Loop_fra
            C_Final_fra = C_Final_fra + C_Loop_fra
        D_Final_fra = D_Final_fra / loop_num1
        C_Final_fra = C_Final_fra / loop_num1
        self.draw_line_pic(np.arange(loop_num2) / 10, D_Final_fra, C_Final_fra, np.arange(loop_num2 / 10), fra_yticks,
                           r='0-10', epoches=self.epoches, type="line2", xlabel="r", ylabel="fraction")

    def line1_pic(self, r):
        D_Y_ave, C_Y_ave, D_Value_ave, C_Value_ave,all_value_ave, type_t_matrix, count_0_ave, count_1_ave = np.zeros(epoches + 1), np.zeros(epoches + 1), np.zeros(epoches),np.zeros(epoches), np.zeros(epoches), np.zeros(epoches), 0, 0
        loop_num = 1
        for i in range(loop_num):
            D_Y, C_Y, D_Value, C_Value, type_t_matrix, count_0, count_1, all_value = self.run(r, type="line1")
            # D_Y_ave = D_Y_ave + D_Y
            # C_Y_ave = C_Y_ave + C_Y
            # D_Value_ave = D_Value_ave + D_Value
            # C_Value_ave = C_Value_ave + C_Value
            # count_0_ave = count_0_ave + count_0
            # count_1_ave = count_1_ave + count_1
            # all_value_ave=all_value_ave+all_value
            # self.save_data('C_fra', 'C_fra', r, i, C_Y)
            # self.save_data('D_fra', 'D_fra', r, i, D_Y)
            # self.save_data('C_value', 'C_value', r, i, C_Value)
            # self.save_data('D_value', 'D_value', r, i, D_Value)
            # self.save_data('all_value', 'all_value', r, i, all_value)
        # D_Y_ave = D_Y_ave / loop_num
        # C_Y_ave = C_Y_ave / loop_num
        # D_Value_ave = D_Value_ave / loop_num
        # C_Value_ave = C_Value_ave / loop_num
        # count_0_ave = count_0_ave / loop_num
        # count_1_ave = count_1_ave / loop_num
        # all_value_ave=all_value_ave/loop_num
        #
        # self.draw_line_pic( D_Y_ave, C_Y_ave, xticks, fra_yticks, r=r, epoches=self.epoches)
        # self.draw_line_pic(D_Value_ave, C_Value_ave, xticks, profite_yticks, r=r, ylim=(0, 15),
        #                    epoches=self.epoches,
        #                    xlabel="T", ylabel="value")
        # self.draw_line_pic( all_value_ave, all_value_ave, xticks, profite_yticks, r=r, ylim=(0, 20),epoches=self.epoches,xlabel="T", ylabel="all_value")
        # #self.shot_pic(type_t_matrix,i)
        #
        # print(count_0_ave / (L_num * L_num))
        # print(count_1_ave / (L_num * L_num))
        # print(D_Value_ave[-1])
        # print(C_Value_ave[-1])

def draw_shot():
    r_list=[2.5,2.9,3.3]
    for r in r_list:
        SPGG = SPGG_Fermi(epoches,L_num,device,r)
        SPGG.line1_pic(r)

if __name__ == '__main__':
    r = 5.0
    try:
        spgg=SPGG_Fermi(epoches,L_num,device,r)
        #spgg.line1_pic(r)
        #spgg.line2_pic(loop_num1=10,loop_num2=51)
        #spgg.run_line2_pic(loop_num1=51,loop_num2=10)
        # spgg=SPGG_Fermi(epoches,L_num,device,r,cal_transfer=True)
        #spgg.cal_transfer_pic()
        draw_shot()

    except Exception as e:
        print(e)
        current_time = datetime.now()
        milliseconds = current_time.microsecond // 1000
        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")
        raise e

    current_time = datetime.now()
    milliseconds = current_time.microsecond // 1000
    print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}.{milliseconds}")


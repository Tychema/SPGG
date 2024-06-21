import numpy as np
import matplotlib.pyplot as plt
from Function.draw_line_pic import draw_line_pic
from Function.mkdir import mkdir
from Origin.Function.read_data import read_data
from Origin.Function.draw_line1 import draw_line1
from Origin.Function.draw_line2 import draw_line2
from Origin.Function.draw_value_line import draw_value_line,draw_all_value_line
from Origin.Function.draw_four_line import draw_transfer_pic,cal_transfer_pic

L_num=200
#colors=['red','green','blue','black']
colors=[(217/255,82/255,82/255),(31/255,119/255,180/255),(120/255,122/255,192/255),(161/255,48/255,63/255),'gold','green']
labels=['D','C']
type_labels=['DD','CC','CDC','StickStrategy']
xticks=[-1,0, 10, 100, 1000, 10000, 100000]
r_xticks=[0,1,2,3,4,5]
fra_yticks=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95, 1.00]
# profite_yticks=[ 8,10,12,14,16,18,20,22]
profite_yticks=[ 0,2,4,6,8,10,12,14,16,18,20,22]
all_value_sum_yticks=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]






def draw_line_qtable(loop_num,name,r,updateMethod,epoches=10000,L_num=200,ylim=(0,1)):
    data=[]
    for i in range(4):
        final_data = np.zeros(epoches)
        for count in range(loop_num):
            loop_data = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), name[i], name[i], r, epoches, L_num,
                                                                         str(count)))
            final_data= final_data + loop_data
        data.append(final_data)
    data=np.array(data)
    data = data / loop_num
    draw_transfer_pic( data[0], data[1],data[2],data[3], xticks, fra_yticks, r=r,updateMethod=updateMethod, epoches=epoches, ylim=ylim)

def draw_line_four_type(loop_num,name,r,updateMethod,labels,epoches=10000,L_num=200,ylim=(0,1),yticks=fra_yticks,ylabel='Fractions'):
    data=[]
    for i in range(len(name)):
        final_data = np.zeros(epoches)
        for count in range(loop_num):
            loop_data = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), name[i], name[i], r, epoches, L_num,
                                                                         str(count)))
            final_data= final_data + loop_data
        data.append(final_data)
    data=np.array(data)
    data = data / loop_num
    draw_transfer_pic( data[0], data[1],data[2],data[3], xticks, yticks,labels,r=r,updateMethod=updateMethod, epoches=epoches, ylim=ylim, ylable=ylabel)

def draw_line_four_type_value(loop_num,name,r,updateMethod,labels,epoches=10000,L_num=200,ylim=(0,1),yticks=fra_yticks,ylabel='Fractions'):
    data=[]
    for i in range(len(name)):
        final_data = np.zeros(epoches)
        for count in range(loop_num):
            loop_data = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), name[i], name[i], r, epoches, L_num,
                                                                         str(count)))
            final_data= final_data + loop_data
        data.append(final_data)
    data=np.array(data)
    data = data / loop_num
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    DD_data = np.insert(data[0], 0, data[0])
    CC_data = np.insert(data[1], 0, data[1])
    CDC_data = np.insert(data[2], 0, data[2])
    SS_data = np.insert(data[3], 0, data[3])
    CDC_D_data = np.insert(data[4], 0, data[4])
    CDC_C_data = np.insert(data[5], 0, data[5])
    plt.plot(np.arange(DD_data.shape[0]), DD_data, color=colors[0],marker='o', label=labels[0], linestyle='-', linewidth=1, markeredgecolor=colors[0], markersize=1,
             markeredgewidth=1)
    plt.plot(np.arange(CC_data.shape[0]), CC_data, color=colors[1],marker='o', label=labels[1], linestyle='-', linewidth=1, markeredgecolor=colors[1],
             markersize=1, markeredgewidth=1)
    plt.plot(np.arange(CDC_data.shape[0]), CDC_data, color=colors[2],marker='o', label=labels[2], linestyle='-', linewidth=1, markeredgecolor=colors[2], markersize=1,markeredgewidth=1)
    plt.plot(np.arange(SS_data.shape[0]), SS_data,color=colors[3],marker='o', label=labels[3], linestyle='-', linewidth=1, markeredgecolor=colors[3],
             markersize=1, markeredgewidth=1)
    plt.plot(np.arange(CDC_D_data.shape[0]), CDC_D_data, color=colors[4],marker='s', label=labels[4], linestyle='-', linewidth=1, markeredgecolor=colors[4], markersize=1,
                markeredgewidth=1)
    plt.plot(np.arange(CDC_C_data.shape[0]), CDC_C_data, color=colors[5],marker='s', label=labels[5], linestyle='-', linewidth=1, markeredgecolor=colors[5], markersize=1,
                markeredgewidth=1)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylim(ylim)
    plt.xscale('log')
    plt.ylabel(ylabel)
    plt.xlabel('t')
    plt.title(str(updateMethod[7:])+': ' + 'L' + str(L_num) + ' r=' + str(r) + ' T=' + str(epoches))
    plt.legend()
    plt.pause(0.001)
    plt.clf()
    plt.close("all")


if __name__ == '__main__':
    #r=3.8
    loop_num=10
    Fermi="Origin_Fermi"
    Qlearning="Origin_Qlearning"
    Origin_Qlearning_NeiborLearning = "Origin_Qlearning_NeiborLearning"
    name=["D_fra","C_fra"]

    #折线图随时间
    # for r in [3.6,3.8,5.0]:
    #     draw_line1(loop_num,name,r, Qlearning,generated='')
    # for r in [3.6,3.7, 3.8, 5.0]:
    #     draw_line1(loop_num,name,r, Fermi,generated='')
    #for r in [3.8]:
    #    draw_line1(loop_num,name,r, Qlearning,generated='generated2')
    #    draw_line1(loop_num,name,r, Fermi,generated='generated2')
    #draw_line1(loop_num,name,4.7, Qlearning,generated='generated3')
    #draw_line1(loop_num,name,r, Fermi)
    #draw_line1(loop_num, name, r, Origin_Qlearning_NeiborLearning)

    #折线图随r
    #draw_line2(51,10,Qlearning)
    #draw_line2(51,10,Fermi)
    #draw_line2(51, 10, Origin_Qlearning_NeiborLearning)

    #all_value折线图
    #draw_all_value_line(loop_num,'all_value',r,ylim=(0,22))

    #value折线图
    #draw_value_line(loop_num,['D_value','C_value'],r,"Origin_Qlearning",ylim=(-1,8))
    #draw_value_line(loop_num,['D_value','C_value'],r,"Origin_Fermi",ylim=(-1,8))

    #transfer折线图
    #cal_transfer_pic(loop_num, ["CC_fra", "DD_fra", "CD_fra", "DC_fra"], r=r, updateMethod="Origin_Qlearning")


    #qtable转换
    #draw_line_qtable(loop_num, ["CC_fra", "DD_fra", "CD_fra", "DC_fra"], r=4.9, updateMethod="Origin_Qlearning")

    #type_four一套
    #draw_line1(loop_num,name,r, Qlearning)
    #draw_value_line(loop_num,['D_value','C_value'],r,"Origin_Qlearning",ylim=(0,12))
    #draw_line_four_type(loop_num, ["DD_Y", "CC_Y", "CDC_Y", "StickStrategy_Y"], r=r, updateMethod="Origin_Qlearning",labels=type_labels,ylim=(0,1),yticks=fra_yticks,ylabel='Fractions')
    #draw_line_four_type(loop_num, ["DD_value_np", "CC_value_np", "CDC_value_np", "StickStrategy_value_np"], r=r, updateMethod="Origin_Qlearning",labels=type_labels,ylim=(0,12),yticks=profite_yticks,ylabel='Average Payoffs')

    #type_five一套
    #draw_line1(loop_num,name,r, Qlearning)
    #draw_value_line(loop_num,['D_value','C_value'],r,"Origin_Qlearning",ylim=(0,12))
    #draw_line_four_type(loop_num, ["DD_Y", "CC_Y", "CDC_Y", "StickStrategy_Y"], r=r, updateMethod="Origin_Qlearning",labels=type_labels,ylim=(0,1),yticks=fra_yticks,ylabel='Fractions')
    #draw_line_four_type_value(loop_num, ["DD_value_np", "CC_value_np", "CDC_value_np", "StickStrategy_value_np","CDC_D_value_np","CDC_C_value_np"], r=r, updateMethod="Origin_Qlearning",labels=['DD','CC','CDC','StickStrategy','CDC_D','CDC_C'],ylim=(8,12),yticks=profite_yticks,ylabel='Average Payoffs')

    #CDC_neibor_num
    #draw_line1(loop_num, ["CDC_neibor_num_np"], r, Qlearning,ylim=(0,5),yticks=[0,1,2,3,4,5])
    #draw_value_line(loop_num, ["CDC_neibor_DD_value_np","CDC_neibor_CC_value_np"], r, Qlearning,ylim=(0,14))

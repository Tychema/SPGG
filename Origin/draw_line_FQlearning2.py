import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Function.draw_line_pic import draw_line_pic
from Function.mkdir import mkdir
from Origin.Function.read_data import read_data
from Origin.Function.draw_line1 import draw_line1
from Origin.Function.draw_line2 import draw_line2
from Origin.Function.draw_value_line import draw_value_line,draw_all_value_line
from Origin.Function.draw_four_line import draw_transfer_pic,cal_transfer_pic
from Function.shot_pic import draw_all_shot_pic_torch
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









def draw_all_value_line(loop_num, name, r, epoches=10000, L_num=200, ylim=(0, 1)):
    updateMethod=["Origin_Qlearning","Origin_Fermi"]
    all_value_ave1=np.zeros(epoches)
    for i in range(loop_num):
        all_value=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod[0]), name, name, r, epoches, L_num,str(i)))
        all_value_ave1=all_value_ave1+all_value/(L_num*L_num)
    all_value_ave1=all_value_ave1/loop_num
    all_value_ave2=np.zeros(epoches)
    for i in range(loop_num):
        all_value=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod[1]), name, name, r, epoches, L_num,str(i)))
        all_value_ave2=all_value_ave2+all_value/(L_num*L_num)
    all_value_ave2=all_value_ave2/loop_num
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(np.arange(all_value_ave1.shape[0]), all_value_ave1, color=colors[0], marker='s',label='Qlearning', linestyle='-', linewidth=1, markeredgecolor=colors[0], markersize=1,markeredgewidth=1)
    plt.plot(np.arange(all_value_ave2.shape[0]), all_value_ave2, color=colors[1], marker='s',label='Fermi', linestyle='-', linewidth=1, markeredgecolor=colors[1], markersize=1,markeredgewidth=1)
    plt.xticks(xticks)
    plt.yticks(profite_yticks)
    #plt.yticks([ 0,1,2,3,4,5,6,7,8])
    plt.ylim(ylim)
    #plt.ylim((0,8))
    plt.xscale('log')
    plt.ylabel('Payoff Average')
    plt.xlabel('t')
    plt.title( 'L' + str(L_num) + ' r=' + str(r) + ' T=' + str(epoches))
    plt.legend()
    plt.pause(0.001)
    plt.clf()
    plt.close("all")
    # draw_line_pic(all_value_ave1, all_value_ave2, xticks, profite_yticks, r=r,updateMethod=updateMethod,epoches=epoches, type="line1", ylabel='Average Payoffs', xlable='t',ylim=(8,20))

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

def com_zhuzhuang(names1, names2, r, updateMethod, epoches=10000, L_num=200):
    colors = [(217 / 255, 82 / 255, 82 / 255), (31 / 255, 119 / 255, 180 / 255)]

    # 准备数据和绘图
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))  # 初始化一个包含两个子图的figure

    # 简化后的数据6
    D_data = {
        'Condition': ['Cond1', 'Cond2'],
        'D': [20, 22],
        'C': [15, 17],
    }
    C_data = {
        'Condition': ['Cond1', 'Cond2'],
        'D': [20, 22],
        'C': [15, 17],
    }
    D_df = pd.DataFrame(D_data)
    C_df = pd.DataFrame(C_data)
    # 假设的读取数据过程，实际应替换为正确的数据读取逻辑
    D_DD = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names1[0], names1[0], r, epoches,L_num, str(loop_num)))[-1]
    D_DC = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names1[1], names1[1], r, epoches,L_num, str(loop_num)))[-1]
    D_CD = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names1[2], names1[2], r, epoches,L_num, str(loop_num)))[-1]
    D_CC = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names1[3], names1[3], r, epoches,L_num, str(loop_num)))[-1]
    C_DD = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names2[0], names2[0], r, epoches,L_num, str(loop_num)))[-1]
    C_DC = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names2[1], names2[1], r, epoches,L_num, str(loop_num)))[-1]
    C_CD = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names2[2], names2[2], r, epoches,L_num, str(loop_num)))[-1]
    C_CC = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names2[3], names2[3], r, epoches,L_num, str(loop_num)))[-1]
    D_df['D'] = [D_DD, D_DC]
    D_df['C'] = [D_CD, D_CC]
    C_df['D'] = [C_DD, C_DC]
    C_df['C'] = [C_CD, C_CC]

    # 绘制柱状图5
    bar_width = 0.25
    gap = 0.23
    r1 = np.arange(len(D_df['D']))
    r2 = [x + bar_width + gap for x in r1]
    bars1 = axs[0].bar(r1, D_df['D'], color='black', width=bar_width, edgecolor='grey')
    bars2 = axs[0].bar(r2, D_df['C'], color='grey', width=bar_width, edgecolor='grey')
    bars3 = axs[1].bar(r1, C_df['D'], color='black', width=bar_width, edgecolor='grey')
    bars4 = axs[1].bar(r2, C_df['C'], color='grey', width=bar_width, edgecolor='grey')
    axs[0].set_ylabel('Q')
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[1].set_ylabel('Q')
    axs[0].bar_label(bars1,padding=3)
    axs[0].bar_label(bars2,padding=3)
    axs[1].bar_label(bars3,padding=3)
    axs[1].bar_label(bars4,padding=3)

    # 在图表下方创建第一个表格
    table_data_1 = [['D', 'C', 'D', 'C']]  # 用空字符串填充以匹配列宽
    the_table1 = axs[0].table(cellText=table_data_1,
                           colWidths=[0.12] * 4,
                           cellLoc='center',
                           bbox=[0, -0.08, 1, 0.08])  # 调整bbox以适应图表大小
    the_table3 = axs[1].table(cellText=table_data_1,
                           colWidths=[0.12] * 4,
                           cellLoc='center',
                           bbox=[0, -0.08, 1, 0.08])  # 调整bbox以适应图表大小

    the_table1.auto_set_font_size(False)
    the_table1.set_fontsize(15)
    the_table3.auto_set_font_size(False)
    the_table3.set_fontsize(15)

    # 在第一个表格下方创建第二个表格
    table_data_2 = [['D', 'C']]
    the_table2 = axs[0].table(cellText=table_data_2,
                           colWidths=[0.12] * 2,
                           cellLoc='center',
                           bbox=[0, -0.16, 1, 0.08])  # 调整bbox以适应图表大小
    the_table4 = axs[1].table(cellText=table_data_2,
                           colWidths=[0.12] * 2,
                           cellLoc='center',
                           bbox=[0, -0.16, 1, 0.08])  # 调整bbox以适应图表大小

    the_table2.auto_set_font_size(False)
    the_table2.set_fontsize(15)
    the_table4.auto_set_font_size(False)
    the_table4.set_fontsize(15)

    # 调整子图间距
    plt.tight_layout()

    # 显示图表
    plt.show()





if __name__ == '__main__':
    r=3.3
    loop_num=10

    Origin_Fermi_Qlearning2="Origin_Fermi_Qlearning2"
    name=["D_fra","C_fra"]

    # for r in [2.5,25/9,3.3]:
    #     draw_line1(loop_num,name,r, Origin_Fermi_Qlearning2,generated='')


    for r in [2.5,25/9,3.3]:
        draw_all_shot_pic_torch(r,epoches=10000,L_num=200,count=0,updateMethod=Origin_Fermi_Qlearning2,generated="generated1")

    #折线图随时间
    #draw_line1(loop_num, name, r, Origin_Fermi_Qlearning2,epoches=10000,L_num=200)
    #draw_line1(loop_num, name, r, Origin_selfQlearning,epoches=10000)

    #折线图随r
    #draw_line2(51, 10, Origin_Fermi_Qlearning1,epoches=20000)
    #draw_line2(51, 10, Origin_selfQlearning,epoches=10000)

    #all_value折线图
    #draw_all_value_line(loop_num,'all_value',r,ylim=(0,22))

    #value折线图
    #draw_value_line(loop_num,['D_value','C_value'],r,Origin_Fermi_Qlearning2,ylim=(0,14),epoches=50000)
    #draw_value_line(loop_num,['D_value','C_value'],r,Origin_selfQlearning,ylim=(0,14),epoches=10000)

    #Qtable
    #draw_line_four_type(5, ["Q_D_DD", "Q_D_DC", "Q_D_CD", "Q_D_CC"], na='QtableD',epoches=50000, r=r,updateMethod=Origin_Fermi_Qlearning2, labels=['D_DD', 'D_CD', 'D_DC', 'D_CC'], ylim=(0, 40),yticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], ylabel='Fractions')
    #draw_line_four_type(5, ["Q_C_DD", "Q_C_DC", "Q_C_CD", "Q_C_CC"],na='QtableD',epoches=50000, r=r, updateMethod=Origin_Fermi_Qlearning2,labels=['C_DD','C_CD','C_DC','C_CC'], ylim=(0, 40), yticks=[ 0,5,10,15,20,25,30,35,40,45,50,55,60], ylabel='Fractions')
    #draw_line_four_type(loop_num, ["Q_D_DD", "Q_D_DC", "Q_D_CD", "Q_D_CC"], na='QtableD',epoches=10000, r=r,updateMethod=Origin_selfQlearning, labels=['D_DD', 'D_CD', 'D_DC', 'D_CC'], ylim=(0, 200),yticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,200], ylabel='Fractions')
    #draw_line_four_type(loop_num, ["Q_C_DD", "Q_C_DC", "Q_C_CD", "Q_C_CC"],na='QtableC',epoches=10000, r=r, updateMethod=Origin_selfQlearning,labels=['C_DD','C_CD','C_DC','C_CC'], ylim=(0, 200),yticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,200], ylabel='Fractions')
    #zhuzhuangtu(5,["Q_D_DD", "Q_D_DC", "Q_D_CD", "Q_D_CC"],r,Origin_Fermi_Qlearning2,epoches=50000,L_num=200)
    #zhuzhuangtu(5, ["Q_C_DD", "Q_C_DC", "Q_C_CD", "Q_C_CC"], r, Origin_Fermi_Qlearning2, epoches=50000, L_num=200)
    #com_zhuzhuang(["Q_D_DD", "Q_D_DC", "Q_D_CD", "Q_D_CC"], ["Q_C_DD", "Q_C_DC", "Q_C_CD", "Q_C_CC"], r, Origin_Fermi_Qlearning2, epoches=50000, L_num=200)

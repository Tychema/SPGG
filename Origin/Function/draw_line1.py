import numpy as np
import matplotlib.pyplot as plt
colors=[(217/255,82/255,82/255),(31/255,119/255,180/255),(120/255,122/255,192/255),(161/255,48/255,63/255),'gold','green']
labels=['D','C']
linestyle=['-','--']
marker=['s','+']
fra_yticks= [0,0.2,0.4,0.6,0.8,1]
xticks=[1,10,100,1000,10000]
from Origin.Function.read_data import read_data

# def draw_line1(loop_num,name,r,updateMethod,epoches=10000,L_num=200,ylim=(0,1),yticks=fra_yticks):
#     plt.clf()
#     plt.close("all")
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     i=0
#     for na in name:
#         final_data = np.zeros(epoches+1)
#         for count in range(loop_num):
#             data=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),na, na, r, epoches,L_num, str(count)))
#             final_data=final_data+data
#         final_data=final_data/loop_num
#         final_data = np.insert(final_data, 0, final_data[0])
#         print(final_data[-1])
#         plt.plot(np.arange(final_data.shape[0]), final_data, color=colors[i], marker=marker[i],label=labels[i], linestyle=linestyle[i], linewidth=1, markeredgecolor=colors[i], markersize=5,
#                  markeredgewidth=1)
#         i=i+1
#     plt.xticks(xticks,fontsize=13)
#     plt.yticks(yticks,fontsize=13)
#     plt.ylim(ylim)
#     plt.xscale('log')
#     plt.ylabel('Fractions',fontsize=14)
#     plt.xlabel('t',fontsize=15)
#     #plt.title(str(updateMethod[7:])+': ' + 'L' + str(L_num) + ' r=' + str(r) + ' T=' + str(epoches))
#     plt.legend()
#
#     plt.pause(0.001)
#     plt.clf()
#     plt.close("all")

log_start_points = [2,4,6,8,10,20,40,60,80,100,200,400,600,800,1000,2000,4000,6000,8000,10000]
def draw_line1(loop_num,name,r,updateMethod,generated,epoches=10000,L_num=200,ylim=(0,1),yticks=fra_yticks):
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i=0
    for na in name:
        final_data = np.zeros(epoches+1)
        for count in range(loop_num):
            data=read_data('data/{}{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),generated if generated=='' else '/'+generated,na, na, r, epoches if r!=25/9 else 50000,L_num, str(count)))
            if r==25/9: data=data[0:epoches+1]
            final_data=final_data+data
        final_data=final_data/loop_num
        final_data = np.insert(final_data, 0, final_data[0])
        print(final_data[-1])
        if na=="C_fra":
            plt.plot(np.arange(final_data.shape[0]), final_data, color=colors[i], marker='+',label=labels[i], linestyle='-', linewidth=1, markeredgecolor=colors[i], markersize=5,
                 markeredgewidth=1,markevery=log_start_points )
        else:
            plt.plot(np.arange(final_data.shape[0]), final_data, color=colors[i], marker='>',label=labels[i], linestyle='-', linewidth=1, markeredgecolor=colors[i], markersize=5,
                 markeredgewidth=1,markevery=log_start_points )
        i=i+1
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylim(ylim)
    plt.xscale('log')
    plt.ylabel('Fractions',fontsize=14)
    plt.xlabel('t',fontsize=15)
    # 获取当前的y轴刻度标签
    y_labels = ax.get_yticklabels()
    x_labels = ax.get_xticklabels()
    # 设置y轴刻度标签的字体大小
    for label in y_labels:
        label.set_fontsize(15)  # 这里14是字体大小，你可以替换为任何你想要的大小
        label.set_fontname('Times New Roman')
    for label in x_labels:
        label.set_fontsize(15)
        label.set_fontname('Times New Roman')
    #plt.title(str(updateMethod[7:])+': ' + 'L' + str(L_num) + ' r=' + str(r) + ' T=' + str(epoches))
    plt.legend()
    import os
    if not os.path.exists('data/{}/Line_pic/r={}'.format(str(updateMethod),str(r))):
        os.makedirs('data/{}/Line_pic/r={}'.format(str(updateMethod),str(r)))
    plt.savefig(
        'data/{}/Line_pic/r={}/{}{}_L={}_r={}_T={}.jpeg'.format(str(updateMethod), str(r), str(updateMethod),generated if generated=='' else '_'+generated, str(L_num),str(r), str(epoches)), format='jpeg', dpi=1000,pad_inches=0, quality=95)
    plt.pause(0.001)
    plt.clf()
    plt.close("all")
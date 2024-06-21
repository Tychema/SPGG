import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

L_num=200
#colors=['red','green','blue','black']
colors=[(120/255,122/255,192/255),(31/255,119/255,180/255),(217/255,82/255,82/255),(161/255,48/255,63/255),'gold','green','blue','black']
labels=['D','C']
type_labels=['DD','CC','CDC','StickStrategy']
xticks=[-1,0, 10, 100, 1000, 10000, 100000]
r_xticks=[0,1,2,3,4,5]
fra_yticks= [0,0.2,0.4,0.6,0.8,1]
# profite_yticks=[ 8,10,12,14,16,18,20,22]
profite_yticks=[ 0,2,4,6,8,10,12,14,16,18,20,22]
all_value_sum_yticks=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
log_start_points = [2,4,6,8,10,20,40,60,80,100,200,400,600,800,1000,2000,4000,6000,8000,10000]
def read_data(path):
    data = np.loadtxt(path)
    return data

def mkdir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def draw_transfer_pic( data, xticks, yticks,labels,r,updateMethod, na,ylim=(0, 1), epoches=10000, ylable='Fractions'):
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom=0.13,top=0.95)
    i=0
    for data1 in data:
        data1 = np.insert(data1, 0, data1[0])
        if i==2:
            plt.plot(np.arange(data1.shape[0]), data1, marker='+', linestyle='-', linewidth=3,color=colors[i],markeredgecolor=colors[i], label=labels[i],markersize=7,markeredgewidth=1,markevery=log_start_points)
        else:
            plt.plot(np.arange(data1.shape[0]), data1, marker='o', linestyle='-', linewidth=1,color=colors[i],markeredgecolor=colors[i], label=labels[i],markersize=1,markeredgewidth=1)
        i=i+1
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylim(ylim)
    plt.xscale('log')
    plt.ylabel(ylable,fontsize=15)
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
    #plt.title(str(updateMethod[7:])+': ' + 'L=' + str(L_num) + '_r=' + str(r) + '_T=' + str(epoches))
    plt.legend()
    mkdir('data/Line_pic/r={}'.format(r))
    #plt.savefig('data/Line_pic/r={}/{}_{}_L={}_r={}_T={}.png'.format(r,updateMethod,na,L_num, r, epoches))
    plt.savefig('data/Line_pic/r={}/{}_{}_L={}_r={}_T={}.jpeg'.format(r,updateMethod,na,L_num, r, epoches),dpi=1000,format='jpeg',bbox_inches='tight',pad_inches=0.03)
    plt.show()
    plt.pause(0.001)
    plt.clf()
    plt.close("all")

def draw_EightType(loop_num,name,r,updateMethod,epoches=10000,L_num=200,ylim=(0,1)):
    data=[]
    for i in range(len(name)):
        final_data = np.zeros(epoches)
        for count in range(loop_num):
            loop_data = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), name[i], name[i], r, epoches, L_num,
                                                                         str(count)))
            final_data= final_data + loop_data
        data.append(final_data)
    data=np.array(data)
    data = data / loop_num /40000

    draw_transfer_pic( data, xticks, fra_yticks, labels=["CCC","CCD","CDC","CDD","DDC","DDD","DCD","DCC"],na="EightType",r=r,updateMethod=updateMethod,
                           epoches=epoches, ylim=ylim)


if __name__ == '__main__':
    r=25/9
    loop_num=10
    Origin_Qlearning="Origin_Qlearning"
    Origin_Fermi_Qlearning2="Origin_Fermi_Qlearning2"
    name=["CCC_np","CCD_np","CDC_np","CDD_np","DDC_np","DDD_np","DCD_np","DCC_np"]

    #折线图随时间
    draw_EightType(loop_num, name, r, Origin_Fermi_Qlearning2,epoches=10000,L_num=200)
    #draw_EightType(loop_num, name, 4.7, Origin_Qlearning,epoches=10000,L_num=200)
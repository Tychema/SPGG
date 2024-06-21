import matplotlib.pyplot as plt
import numpy as np
from Origin.Function.read_data import read_data
from Origin.Function.mkdir import mkdir
colors=[(217/255,82/255,82/255),(31/255,119/255,180/255),(120/255,122/255,192/255),(161/255,48/255,63/255),'gold','green']
xticks=[0, 10, 100, 1000, 10000, 100000]
profite_yticks=[ 8,10,12,14,16,18,20,22]
labels=['D','C']
L_num=200
fra_yticks=[0,0.2,0.4,0.6,0.8,1]

def draw_transfer_pic( DD_data,CC_data, CD_data, DC_data, xticks, yticks,labels,r,updateMethod, na,ylim=(0, 1), epoches=10000, ylable='Fractions'):
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    DD_data = np.insert(DD_data, 0, DD_data[0])
    CC_data = np.insert(CC_data, 0, CC_data[0])
    CD_data = np.insert(CD_data, 0, CD_data[0])
    DC_data = np.insert(DC_data, 0, DC_data[0])
    plt.plot(np.arange(DD_data.shape[0]), DD_data, color=colors[0],marker='o', label=labels[0], linestyle='-', linewidth=1, markeredgecolor=colors[0], markersize=1,
             markeredgewidth=1)
    plt.plot(np.arange(CC_data.shape[0]), CC_data, color=colors[1],marker='o', label=labels[1], linestyle='-', linewidth=1, markeredgecolor=colors[1],
             markersize=1, markeredgewidth=1)
    plt.plot(np.arange(CD_data.shape[0]), CD_data, color=colors[2],marker='o', label=labels[2], linestyle='-', linewidth=1, markeredgecolor=colors[2], markersize=1,markeredgewidth=1)
    plt.plot(np.arange(DC_data.shape[0]), DC_data,color=colors[3],marker='o', label=labels[3], linestyle='-', linewidth=1, markeredgecolor=colors[3],
             markersize=1, markeredgewidth=1)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylim(ylim)
    plt.xscale('log')
    plt.ylabel(ylable)
    plt.xlabel('t')
    plt.title(str(updateMethod[7:])+': ' + 'L' + str(L_num) + ' r=' + str(r) + ' T=' + str(epoches))
    plt.legend()
    mkdir('data/Line_pic/r={}'.format(r))
    plt.savefig('data/Line_pic/r={}/{}_{}_L={}_r={}_T={}.png'.format(r,updateMethod,na,L_num, r, epoches))
    plt.pause(0.001)
    plt.clf()
    plt.close("all")

def cal_transfer_pic(loop_num,name,r,updateMethod,epoches=10000,L_num=200,ylim=(0,1)):
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

    draw_transfer_pic( data[0], data[1],data[2],data[3], xticks, fra_yticks, r=r,updateMethod=updateMethod,
                           epoches=epoches, ylim=ylim)
import matplotlib.pyplot as plt
import numpy as np
from Origin.Function.read_data import read_data
from Origin.Function.mkdir import mkdir
colors=[(217/255,82/255,82/255),(31/255,119/255,180/255),(120/255,122/255,192/255),(161/255,48/255,63/255),'gold','green']
xticks=[0, 10, 100, 1000, 10000, 100000]
profite_yticks=[ 8,10,12,14,16,18,20,22]
labels=['D','C']

def draw_value_line(loop_num,name,r,updateMethod,epoches=10000,L_num=200,ylim=(0,1)):
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i=0
    for na in name:
        final_data = np.zeros(epoches)
        for count in range(loop_num):
            data=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),na, na, r, epoches,L_num, str(count)))
            final_data=final_data+data
        final_data=final_data/loop_num
        final_data = np.insert(final_data, 0, final_data[0])
        plt.plot(np.arange(final_data.shape[0]), final_data, color=colors[i], marker='o',label=labels[i], linestyle='-', linewidth=1, markeredgecolor=colors[i], markersize=1,
                 markeredgewidth=1)
        i=i+1
    plt.xticks(xticks)
    plt.yticks(profite_yticks)
    plt.ylim(ylim)
    plt.xscale('log')
    plt.ylabel('Average Payoffs')
    plt.xlabel('t')
    plt.title(str(updateMethod[7:])+':' + 'L' + str(L_num) + ' r=' + str(r) + ' T=' + str(epoches))
    plt.legend()
    mkdir('data/Line_pic/r={}'.format(r))
    plt.savefig('data/Line_pic/r={}/{}_value_L={}_r={}_T={}.png'.format(r,updateMethod,L_num, r, epoches))
    plt.pause(0.001)
    plt.clf()
    plt.close("all")


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
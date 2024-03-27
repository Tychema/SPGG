import numpy as np
import matplotlib.pyplot as plt

L_num=200
colors=['red','green','blue','black']
labels=['D','C']
xticks=[0, 10, 100, 1000, 10000, 100000]
fra_yticks=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95, 1.00]
profite_yticks=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
def read_data(path):
    data = np.loadtxt(path)
    return data

def draw_line_pic(obsX,D_Y,C_Y,xticks,yticks,r,ylim=(0,1),epoches=10000,type="line1",xlable='step',ylabel='fraction'):
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
    plt.title('Q_learning:'+'L='+str(L_num)+' r='+str(r)+' n_iter='+str(epoches))
    plt.pause(0.001)
    plt.clf()
    plt.close("all")

def draw_line2(loop_num1,loop_num2,updateMethod,epoches=10000,L_num=200):
    D_Final_fra = np.array([])
    C_Final_fra = np.array([])
    r = 0
    for i in range(loop_num1):
        D_Loop_fra=0
        C_Loop_fra=0
        for count in range(loop_num2):
            C_Y=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),'C_fra', 'C_fra', r, epoches,L_num, str(count)))
            D_Y=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),'D_fra', 'D_fra', r, epoches,L_num, str(count)))
            D_Loop_fra = D_Loop_fra+D_Y[-1]
            C_Loop_fra = C_Loop_fra+C_Y[-1]
        D_Final_fra = np.append(D_Final_fra, D_Loop_fra)
        C_Final_fra = np.append(C_Final_fra, C_Loop_fra)
        r = r + 0.1
    D_Final_fra = D_Final_fra / loop_num2
    C_Final_fra = C_Final_fra / loop_num2
    draw_line_pic(np.arange(loop_num1) / 10, D_Final_fra, C_Final_fra, np.arange(loop_num1 / 10), fra_yticks,
                       r='0-5', epoches=epoches, type="line2", ylabel='fraction', xlable='r')

def draw_line1(loop_num,name,r,updateMethod,epoches=10000,L_num=200,ylim=(0,1)):
    r1=r+0.01
    r2=0
    while r2+0.1<=r1:
        r2=r2+0.1
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i=0
    for na in name:
        final_data = np.zeros(epoches+1)
        for count in range(loop_num):
            data=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),na, na, r2, epoches,L_num, str(count)))
            final_data=final_data+data
        final_data=final_data/loop_num
        plt.plot(np.arange(final_data.shape[0]), final_data, color=colors[i], label=labels[i], linestyle='-', linewidth=1, markeredgecolor=colors[i], markersize=1,
                 markeredgewidth=1)
        i=i+1
    plt.xticks(xticks)
    plt.yticks(fra_yticks)
    plt.ylim(ylim)
    plt.xscale('log')
    plt.ylabel('fraction')
    plt.xlabel('step')
    plt.title('Q_learning:' + 'L' + str(L_num) + ' r=' + str(r) + ' n_iter=' + str(epoches))
    plt.legend()
    plt.pause(0.001)
    plt.clf()
    plt.close("all")

def draw_transfer_pic( obsX, CC_data, DD_data, CD_data, DC_data, xticks, yticks, r, ylim=(0, 1), epoches=10000):
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
    plt.ylabel('fraction')
    plt.xlabel('step')
    plt.title('Q_learning:'+'L'+str(L_num)+' r='+str(r)+' n_iter='+str(epoches))
    plt.legend()
    plt.pause(0.001)
    plt.clf()
    plt.close("all")

def cal_transfer_pic(loop_num,name,r,updateMethod,epoches=10000,L_num=200,ylim=(0,1)):
    r1 = r + 0.01
    r2 = 0
    while r2 + 0.1 <= r1:
        r2 = r2 + 0.1
    data=[]
    for i in range(4):
        final_data = np.zeros(epoches)
        for count in range(loop_num):
            loop_data = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), name[i], name[i], r2, epoches, L_num,
                                                                         str(count)))
            final_data= final_data + loop_data
        data.append(final_data)
    data=np.array(data)
    data = data / loop_num

    draw_transfer_pic(np.arange(epoches), data[0], data[1],data[2],data[3], xticks, fra_yticks, r=r,
                           epoches=epoches, ylim=ylim)

if __name__ == '__main__':
    loop_num=10
    name=["D_fra","C_fra"]
    draw_line1(loop_num,name,r=4.9,updateMethod="Origin_Qlearning")
    #draw_line2(50,10,"Origin_Qlearning")
    cal_transfer_pic(loop_num, ["CC_fra", "DD_fra", "CD_fra", "DC_fra"], r=4.9, updateMethod="Origin_Qlearning")

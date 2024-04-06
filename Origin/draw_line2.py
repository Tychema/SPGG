import numpy as np
import matplotlib.pyplot as plt

L_num=200
#colors=['red','green','blue','black']
colors=[(217/255,82/255,82/255),(31/255,119/255,180/255),(120/255,122/255,192/255),(161/255,48/255,63/255)]
labels=['D','C']
type_labels=['DD','CC','CDC','StickStrategy']
xticks=[-1,0, 10, 100, 1000, 10000, 100000]
r_xticks=[0,1,2,3,4,5]
fra_yticks=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95, 1.00]
# profite_yticks=[ 8,10,12,14,16,18,20,22]
profite_yticks=[ 0,2,4,6,8,10,12,14,16,18,20,22]
all_value_sum_yticks=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
def read_data(path):
    data = np.loadtxt(path)
    return data

def draw_line_pic(D_Y,C_Y,xticks,yticks,r,updateMethod,ylim=(0,1),epoches=10000,type="line1",xlable='t',ylabel='Fractions'):
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if(type=="line1"):
        D_Y = np.insert(D_Y, 0, D_Y[0])
        C_Y = np.insert(C_Y, 0, C_Y[0])
        D_X=np.arange(D_Y.shape[0])
        C_X=np.arange(C_Y.shape[0])
    else:
        D_X = np.arange(D_Y.shape[0]) / 10
        C_X = np.arange(C_Y.shape[0]) / 10
    plt.plot(D_X, D_Y, color=colors[0],marker='s',markersize=5,markerfacecolor='none',label='D', linestyle='-', linewidth=1, markeredgecolor='r', markeredgewidth=1)
    plt.plot(C_X, C_Y, color=colors[1],marker='s',markersize=5,markerfacecolor='none',label='C', linestyle='-', linewidth=1, markeredgecolor='b', markeredgewidth=1)
    if(updateMethod=="Origin_Qlearning" and type=="line2"):
        plt.plot([3.6,4.7,5.0],[D_Y[36],D_Y[47],D_Y[50]],color='black',marker='o',markersize=5,markeredgecolor='black',markeredgewidth=1,linestyle='')
        plt.plot([3.6,4.7,5.0],[C_Y[36],C_Y[47],C_Y[50]],color='black',marker='o',markersize=5,markeredgecolor='black',markeredgewidth=1,linestyle='')
        plt.axvline(x=3.6, ymin=0, ymax=1, color='lightgray', linestyle='--')  # 画一条垂直的虚线
        plt.axvline(x=4.7, ymin=0, ymax=1, color='lightgray', linestyle='--')  # 画一条垂直的虚线
        plt.axvline(x=5.0, ymin=0, ymax=1, color='lightgray', linestyle='--')  # 画一条垂直的虚线
    elif(updateMethod=="Origin_Fermi" and type=="line2"):
        plt.plot([3.7,3.9,5.0],[D_Y[37],D_Y[39],D_Y[50]],color='black',marker='o',markersize=5,markeredgecolor='black',markeredgewidth=1,linestyle='')
        plt.plot([3.7,3.9,5.0],[C_Y[37],C_Y[39],C_Y[50]],color='black',marker='o',markersize=5,markeredgecolor='black',markeredgewidth=1,linestyle='')
        plt.axvline(x=3.7, ymin=0, ymax=1, color='lightgray', linestyle='--')  # 画一条垂直的虚线
        plt.axvline(x=3.9, ymin=0, ymax=1, color='lightgray', linestyle='--')  # 画一条垂直的虚线
        plt.axvline(x=5.0, ymin=0, ymax=1, color='lightgray', linestyle='--')  # 画一条垂直的虚线
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylim(ylim)
    if(type=="line1"):
        plt.xscale('log')
    plt.ylabel(ylabel)
    plt.xlabel(xlable)
    plt.title(str(updateMethod[7:])+': '+'L='+str(L_num)+' r='+str(r)+' T='+str(epoches))
    plt.legend()
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
            C_Y=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),'C_fra', 'C_fra', r/10, epoches,L_num, str(count)))
            D_Y=read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),'D_fra', 'D_fra', r/10, epoches,L_num, str(count)))
            D_Loop_fra = D_Loop_fra+D_Y[-1]
            C_Loop_fra = C_Loop_fra+C_Y[-1]
        D_Final_fra = np.append(D_Final_fra, D_Loop_fra)
        C_Final_fra = np.append(C_Final_fra, C_Loop_fra)
        r = r + 1
    D_Final_fra = D_Final_fra / loop_num2
    C_Final_fra = C_Final_fra / loop_num2

    draw_line_pic( D_Final_fra, C_Final_fra, r_xticks, fra_yticks,r='0-5',updateMethod=updateMethod, epoches=epoches, type="line2", ylabel='Fractions', xlable='r')

def draw_line1(loop_num,name,r,updateMethod,epoches=10000,L_num=200,ylim=(0,1)):
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i=0
    for na in name:
        final_data = np.zeros(epoches+1)
        for count in range(loop_num):
            data=read_data('data/{}/generated3/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),na, na, r, epoches,L_num, str(count)))
            final_data=final_data+data
        final_data=final_data/loop_num
        final_data = np.insert(final_data, 0, final_data[0])
        print(final_data[-1])
        plt.plot(np.arange(final_data.shape[0]), final_data, color=colors[i], marker='o',label=labels[i], linestyle='-', linewidth=1, markeredgecolor=colors[i], markersize=1,
                 markeredgewidth=1)
        i=i+1
    plt.xticks(xticks)
    plt.yticks(fra_yticks)
    plt.ylim(ylim)
    plt.xscale('log')
    plt.ylabel('Fractions')
    plt.xlabel('t')
    plt.title(str(updateMethod[7:])+': ' + 'L' + str(L_num) + ' r=' + str(r) + ' T=' + str(epoches))
    plt.legend()
    plt.pause(0.001)
    plt.clf()
    plt.close("all")


def draw_value_line(loop_num,name,r,updateMethod,epoches=10000,L_num=200,ylim=(0,1)):
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i=0
    for na in name:
        final_data = np.zeros(epoches)
        for count in range(loop_num):
            data=read_data('data/{}/generated3/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod),na, na, r, epoches,L_num, str(count)))
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
    plt.pause(0.001)
    plt.clf()
    plt.close("all")

def draw_transfer_pic(CC_data, DD_data, CD_data, DC_data, xticks, yticks,labels,r,updateMethod, ylim=(0, 1), epoches=10000, ylable='Fractions'):
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
    plt.yticks([ 4,6,8,10,12,14,16,18,20,22])
    #plt.yticks([ 0,1,2,3,4,5,6,7,8])
    plt.ylim((4,22))
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

def draw_line_four_type(loop_num,name,r,updateMethod,labels,epoches=10000,L_num=200,ylim=(0,1),yticks=fra_yticks,ylabel='Fractions'):
    data=[]
    for i in range(4):
        final_data = np.zeros(epoches)
        for count in range(loop_num):
            loop_data = read_data('data/{}/generated3/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), name[i], name[i], r, epoches, L_num,
                                                                         str(count)))
            final_data= final_data + loop_data
        data.append(final_data)
    data=np.array(data)
    data = data / loop_num
    draw_transfer_pic( data[0], data[1],data[2],data[3], xticks, yticks,labels,r=r,updateMethod=updateMethod, epoches=epoches, ylim=ylim, ylable=ylabel)


if __name__ == '__main__':
    r=3.6
    loop_num=10
    Fermi="Origin_Fermi"
    Qlearning="Origin_Qlearning"
    name=["D_fra","C_fra"]

    #折线图随时间
    #draw_line1(loop_num,name,r, Qlearning)
    #draw_line1(loop_num,name,r, Fermi)

    #折线图随r
    #draw_line2(51,10,Qlearning)
    #draw_line2(51,10,Fermi)

    #all_value折线图
    #draw_all_value_line(loop_num,'all_value',r)

    #value折线图
    #draw_value_line(loop_num,['D_value','C_value'],r,"Origin_Qlearning",ylim=(8,14))

    #transfer折线图
    #cal_transfer_pic(loop_num, ["CC_fra", "DD_fra", "CD_fra", "DC_fra"], r=r, updateMethod="Origin_Qlearning")


    #qtable转换
    #draw_line_qtable(loop_num, ["CC_fra", "DD_fra", "CD_fra", "DC_fra"], r=4.9, updateMethod="Origin_Qlearning")

    #type_four一套
    draw_line1(loop_num,name,r, Qlearning)
    draw_value_line(loop_num,['D_value','C_value'],r,"Origin_Qlearning",ylim=(0,12))
    draw_line_four_type(loop_num, ["DD_Y", "CC_Y", "CDC_Y", "StickStrategy_Y"], r=r, updateMethod="Origin_Qlearning",labels=type_labels,ylim=(0,1),yticks=fra_yticks,ylabel='Fractions')
    draw_line_four_type(loop_num, ["DD_value_np", "CC_value_np", "CDC_value_np", "StickStrategy_value_np"], r=r, updateMethod="Origin_Qlearning",labels=type_labels,ylim=(0,12),yticks=profite_yticks,ylabel='Average Payoffs')

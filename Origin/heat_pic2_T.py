import numpy as np
import matplotlib.pyplot as plt

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
def read_data(path):
    data = np.loadtxt(path)
    return data

def mkdir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

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
    mkdir('data/Line_pic')
    plt.savefig('data/Line_pic/{}_L={}_r={}_T={}.png'.format(updateMethod,L_num, '0-5', epoches))
    plt.pause(0.001)
    plt.clf()
    plt.close("all")

def draw_line1(loop_num,name,r,updateMethod,epoches=10000,L_num=200,ylim=(0,1),yticks=fra_yticks,eta=0.1,gamma=0.1):
    iterations=read_data('data/Origin_Fermi_Qlearning2/heat_pic2_T/eta={}/gammm={}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(eta),str(gamma),str('iterations'),str('iterations'),str(r),str(epoches),str(L_num),str(0)))
    print('iterations:',iterations)
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i=0
    for na in name:
        final_data = np.zeros(int(iterations)+1)
        for count in range(loop_num):
            data=read_data('data/Origin_Fermi_Qlearning2/heat_pic2_T/eta={}/gammm={}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(eta),str(gamma),str(na),str(na),str(r),str(epoches),str(L_num),str(0)))
            final_data=final_data+data
        final_data=final_data/loop_num
        final_data = np.insert(final_data, 0, final_data[0])
        print(final_data[-1])
        plt.plot(np.arange(final_data.shape[0]), final_data, color=colors[i], marker='o',label=labels[i], linestyle='-', linewidth=1, markeredgecolor=colors[i], markersize=1,
                 markeredgewidth=1)
        i=i+1
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylim(ylim)
    plt.xscale('log')
    plt.ylabel('Fractions')
    plt.xlabel('t')
    plt.title(str(updateMethod[7:]) + '_L=' + str(L_num) + '_r=' + str(r) + '_T=' + str(epoches))
    plt.legend()
    mkdir('data/Line_pic/r={}'.format(r))
    #plt.savefig('data/Line_pic/r={}/{}_L={}_r={}_T={}.png'.format(r,updateMethod,L_num, r, epoches))
    plt.show()
    plt.clf()
    plt.close("all")


def draw_eta_iterations(loop_num, name, r, updateMethod, epoches=10000, L_num=50, gamma=0.1):
    iterations_list = np.array([])
    for eta1 in range(1, 51):
        eta = eta1 / 100
        iterations = read_data('data/Origin_Fermi_Qlearning2/heat_pic2_T/eta={}/gammm={}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(eta), str(gamma), str('iterations'), str('iterations'), str(r), str(epoches), str(L_num), str(0)))
        iterations_list = np.append(iterations_list, iterations)

    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot((np.arange(iterations_list.shape[0]) + 1) / 100, iterations_list, color=colors[0], marker='o',
             linestyle='-', linewidth=1, markeredgecolor=colors[0], markersize=1,
             markeredgewidth=1)

    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.yticks([10000, 100000, 200000, 300000, 400000, 500000])
    # 确保y轴的下限与最小刻度值对齐
    # plt.ylim([0, 500000])
    plt.ylabel('t')  # 这里可能会被遮挡，接下来我们将调整边距

    # 调整图表边距，增大左侧空间
    plt.subplots_adjust(left=0.15)  # 增大左侧边距，0.15是一个示例值，可以根据需要调整

    # plt.yscale('log')
    plt.xlabel('η')
    # plt.title(str(updateMethod[7:]) + f'r={r} L={L_num} T at convergence')
    #plt.legend()

    plt.savefig('data/paper pic/eta_iteration.jpeg',dpi=1000,format='jpeg',bbox_inches='tight',pad_inches=0.03)
    plt.show()
    plt.clf()
    plt.close("all")

def draw_gamma_iterations(loop_num,name,r,updateMethod,epoches=10000,L_num=50,eta=0.1):
    iterations_list=np.array([])
    for gamma1 in range(50,100):
        gamma=gamma1/100
        data=read_data('data/Origin_Fermi_Qlearning2/heat_pic2_T/eta={}/gammm={}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(eta),str(gamma),str('C_fra'),str('C_fra'),str(r),str(epoches),str(L_num),str(0)))
        iterations_list=np.append(iterations_list,data.shape)
        mkdir('data/Origin_Fermi_Qlearning2/heat_pic2_T/eta={}/gammm={}/{}/'.format(str(eta),str(gamma),str('iterations'),str('iterations')))
        # np.savetxt(
        #     'data/Origin_Fermi_Qlearning2/heat_pic2_T/eta={}/gammm={}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(
        #         str(eta), str(gamma), str('iterations'), str('iterations'), str(r), str(epoches), str(L_num), str(0)),np.array([data.shape]))

    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot((np.arange(50,50+iterations_list.shape[0]))/100, iterations_list, color=colors[0], marker='o', linestyle='-', linewidth=1, markeredgecolor=colors[0], markersize=1,
                 markeredgewidth=1)
    plt.xticks([0.5,0.6,0.7,0.8,0.9,1.0])
    plt.yticks([10000,100000,200000,300000,400000,500000])
    # 确保y轴的下限与最小刻度值对齐
    #plt.ylim([0,500000])
    plt.ylabel('t')

    # 调整图表边距，增大左侧空间
    plt.subplots_adjust(left=0.15)  # 增大左侧边距，0.15是一个示例值，可以根据需要调整
    #plt.yscale('log')
    plt.xlabel('γ')
    #plt.title(str(updateMethod[7:]) +f'r={r} L={L_num} T at convergence')
    #plt.legend()
    plt.savefig('data/paper pic/gamma_iteration.jpeg',dpi=1000,format='jpeg',bbox_inches='tight',pad_inches=0.03)
    plt.show()
    plt.clf()
    plt.close("all")

def save_heat_matrix(r=2.9,epoches=10000,L_num=50,count=0,type="C_fra"):
    eta_values = 101
    gamma_values = 101
    # 创建一个二维数组来保存每个 fine 和 T 组合下的结果
    results = np.zeros((101, 101))

    # 进行仿真实验并记录结果
    for j in range(gamma_values):
        gamma = j / 100
        for i in range(eta_values):
            eta = i / 100
            data = read_data(
                'data/Origin_Fermi_Qlearning2/heat_pic2/eta={}/gammm={}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(
                    str(eta), str(gamma), str(type), "C_fra", str(r), str(epoches), str(L_num), str(count)))
            results[j, i] = np.mean(data[-100:])

    name = "data/Origin_Fermi_Qlearning2/r=2.9_eta=0-1_gamma=0-1的热图数据_三位小数版本2.csv"
    #np.savetxt(name, results, delimiter=",")
    np.savetxt(name, results, delimiter=",", fmt='%.3f')

def draw_heat_pic(name):
    # 绘制热图
    fig,axs=plt.subplots()
    results = np.loadtxt(name, delimiter=",")
    plt.imshow(results, cmap='hot', origin='lower', extent=[0.0, 1.0, 0.0, 1.0], aspect='auto')
    plt.colorbar()
    plt.xlabel('η')
    plt.ylabel('γ')
    #plt.title('Heatmap of SPGG')
    # 保存热图为图像文件
    fig.text(0.12,0.89,"(b)",fontsize=15,fontweight=20,fontname='Times New Roman')
    plt.savefig("data/paper pic/r=2.9_eta=0-1_gamma=0-1的热图数据2.jpeg",dpi=1000, format='jpeg', bbox_inches='tight', pad_inches=0.03)
    plt.show()

if __name__ == '__main__':
    r=2.9
    loop_num=1
    Fermi="Origin_Fermi"
    Qlearning="Origin_Qlearning"
    Origin_Qlearning_NeiborLearning = "Origin_Qlearning_NeiborLearning"
    Origin_Qlearning_Fermi= "Origin_Qlearning_Fermi"
    Origin_Fermi_Qlearning1="Origin_Fermi_Qlearning1"
    Origin_Fermi_Qlearning2="Origin_Fermi_Qlearning2"
    Origin_selfQlearning="Origin_selfQlearning"
    name=["D_fra","C_fra"]

    #折线图随时间
    #draw_line1(1, name, r, Origin_Fermi_Qlearning2,epoches=1000000,eta=0.01,gamma=0.99,L_num=50)
    #draw_eta_iterations(1, name, r, Origin_Fermi_Qlearning2,epoches=1000000,gamma=0.99,L_num=50)
    #draw_gamma_iterations(1, name, r, Origin_Fermi_Qlearning2,epoches=1000000,eta=0.01,L_num=50)
    #draw_iretations(1, name, r, Origin_Fermi_Qlearning2,epoches=1000000,eta=0.01,gamma=0.99,L_num=50)
    #折线图随r
    #draw_line2(51,10,Qlearning)
    #draw_line2(51,10,Fermi)
    #draw_line2(51, 10, Origin_Fermi_Qlearning1,epoches=20000)
    #draw_line2(51, 10, Origin_selfQlearning,epoches=10000)

    #热图
    #save_heat_matrix(r=r,epoches=10000,L_num=50,count=1,type="C_fra")
    #draw_heat_pic("/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/heat_pic/r=2.9_eta=0-1_gamma=0-1的热图数据_三位小数版本.csv")
    draw_heat_pic("/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/heat_pic3_T/r=2.9_eta=0-1_gamma=0-1的热图数据_三位小数版本.csv")

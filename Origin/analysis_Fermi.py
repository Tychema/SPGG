import torch


# 画快照
from matplotlib import pyplot as plt
from torch import tensor
import numpy as np

text=["(a)","(b)","(c)","(d)","(e)"]
def shot_pic1(type_t_matrix: tensor, i,profit_matrix,t=-1):
    plt.clf()
    plt.close("all")
    # 初始化图表和数据
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    cmap = plt.get_cmap('Set1', 2)
    # 指定图的大小
    #             plt.figure(figsize=(500, 500))  # 10x10的图
    #             plt.matshow(type_t_matrix.cpu().numpy(), cmap=cmap)
    #             plt.colorbar(ticks=[0, 1, 2], label='Color')
    # 显示图片
    # 定义颜色映射
    color_map = {
        # 0设置为黑色
        0: (0, 0, 0),  # 黑色
        # 1设置为白色
        1: (255, 255, 255),  # 白色
    }
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.text(0.25, 0.92, text[t-4985], ha='center', va='center',fontsize=20)
    image = np.zeros((type_t_matrix.shape[0], type_t_matrix.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        image[type_t_matrix.cpu() == label] = color
    ## 在每个像素点上添加profit_matrix的值作为标签
    #for i in range(type_t_matrix.shape[0]):
    #    for j in range(type_t_matrix.shape[1]):
    #        if type_t_matrix[i, j] == 0:
    #            ax.text(j, i,f"{profit_matrix[i, j]:.2f}", ha="center", va="center", color="white",fontsize=40)
    #        else:
    #            ax.text(j, i,f"{profit_matrix[i, j]:.1f}", ha="center", va="center", color="black",fontsize=40)

    #fig.text(0.5, 0.92, 'Case Ⅱ', ha='center', va='center',fontsize=120)
    plt.imshow(image, interpolation='None')
    if t==-1:
        plt.savefig('data/paper pic/pic_tuancu.jpeg',format='jpeg', dpi=1000,bbox_inches='tight',pad_inches=0.3)
        print("no_i_j")
    else:
       plt.savefig('data/paper pic/r=2.77_epoches=10000_L=200_T={}.jpeg'.format(t),format='jpeg', dpi=1000,bbox_inches='tight',pad_inches=0.3)
       print("i_j")
    plt.show()
    plt.clf()
    plt.close("all")



def shot_pic2(type_t_matrix: tensor):
    #type_t_matrix[1, 6] = 2
    plt.clf()
    plt.close("all")

    # 初始化图表和数据
    fig, axes = plt.subplots(1, 3, figsize=(30, 10),gridspec_kw={'width_ratios': [1, 1, 1]})
    plt.subplots_adjust(left=0.1, right=0.9, top=0.875, bottom=0.1, wspace=0.05, hspace=0.2)
    fig.text(0.5, 0.95, 'Case Ⅲ', ha='center', va='center',fontsize=60,fontweight=10)
    fig.text(0.1,0.90,"(a)",fontsize=40,fontweight=20,fontname='Times New Roman')
    fig.text(0.37,0.90,"(b)",fontsize=40,fontweight=20,fontname='Times New Roman')
    fig.text(0.64,0.90,"(c)",fontsize=40,fontweight=20,fontname='Times New Roman')
    cmap = plt.get_cmap('Set1', 2)
    # 定义颜色映射
    color_map = {
        0: (0, 0, 0),  # 黑色
        1: (255, 255, 255),  # 白色
    }
    image = np.zeros((type_t_matrix.shape[0], type_t_matrix.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        image[type_t_matrix.cpu() == label] = color
    # 绘制原始图像
    #axes[0].set_title('Original Matrix')
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].spines['bottom'].set_linewidth(3)  # 设置x轴底部线条宽度
    axes[0].spines['left'].set_linewidth(3)  # 设置y轴左侧线条宽度
    axes[0].spines['top'].set_linewidth(3)
    axes[0].spines['right'].set_linewidth(3)
    axes[0].imshow(image, interpolation='None')

    # 绘制黑白相间的方格
    for i in range(5):
        for j in range(5):
            if (i + j) % 2 == 0:
                # 绘制黑色方格
                axes[1].add_patch(plt.Rectangle((i, j), 1, 1, color='black'))
                # 写入文字 4.64，并确保字体颜色清晰显示在黑色方格上
                axes[1].text(i + 0.5, j + 0.5, '4.44', color='white', ha='center', va='center',fontsize=30)
            else:
                # 绘制白色方格
                axes[1].add_patch(plt.Rectangle((i, j), 1, 1, color='white'))
                # 写入文字 4.86，并确保字体颜色清晰显示在白色方格上
                axes[1].text(i + 0.5, j + 0.5, '4.44', color='black', ha='center', va='center',fontsize=30)

    # 设置坐标轴刻度
    axes[1].set_xticks(range(6))
    axes[1].set_yticks(range(6))
    # 隐藏坐标轴刻度标签
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])

    # 显示网格线
    axes[1].grid(True)
    # 绘制截取并放大的部分
    #axes[1].set_title('D Center')



    # 绘制黑白相间的方格
    for i in range(5):
        for j in range(5):
            if (i + j) % 2 == 0:
                # 绘制黑色方格
                axes[2].add_patch(plt.Rectangle((i, j), 1, 1, color='white'))
                # 写入文字 4.64，并确保字体颜色清晰显示在黑色方格上
                axes[2].text(i + 0.5, j + 0.5, '4.44', color='black', ha='center', va='center',fontsize=30)
            else:
                # 绘制白色方格
                axes[2].add_patch(plt.Rectangle((i, j), 1, 1, color='black'))
                # 写入文字 4.86，并确保字体颜色清晰显示在白色方格上
                axes[2].text(i + 0.5, j + 0.5, '4.44', color='white', ha='center', va='center',fontsize=30)

    # 设置坐标轴刻度
    axes[2].set_xticks(range(6))
    axes[2].set_yticks(range(6))
    # 隐藏坐标轴刻度标签
    axes[2].set_xticklabels([])
    axes[2].set_yticklabels([])

    # 显示网格线
    axes[2].grid(True)
    # 绘制截取并放大的部分
    #axes[2].set_title('C Center')
    # 显示图像
    plt.savefig('data/paper pic/pic9.jpeg', format='jpeg', dpi=1000, bbox_inches='tight', pad_inches=0.3)
    plt.show()

def pic10():
    # type_t_matrix[1, 6] = 2
    plt.clf()
    plt.close("all")

    # 初始化图表和数据
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]})
    # 绘制黑白相间的方格
    for i in range(5):
        for j in range(5):
            if (i + j) % 2 == 0:
                # 绘制黑色方格
                axes[0].add_patch(plt.Rectangle((i, j), 1, 1, color='black'))
                # 写入文字 4.44，并确保字体颜色清晰显示在黑色方格上
                axes[0].text(i + 0.5, j + 0.5, '4.44', color='white', ha='center', va='center')
            else:
                # 绘制白色方格
                axes[0].add_patch(plt.Rectangle((i, j), 1, 1, color='white'))
                # 写入文字 4.44，并确保字体颜色清晰显示在白色方格上
                axes[0].text(i + 0.5, j + 0.5, '4.44', color='black', ha='center', va='center')

    # 设置坐标轴刻度
    axes[0].set_xticks(range(6))
    axes[0].set_yticks(range(6))
    # 隐藏坐标轴刻度标签
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])

    # 显示网格线
    axes[0].grid(True)
    # 绘制截取并放大的部分
    #axes[1].set_title('D Center')



    # 绘制黑白相间的方格
    for i in range(5):
        for j in range(5):
            if (i==0 and j==0)  or (i==0 and j==4)   or (i==4 and j==0)  or (i==4 and j==4):
                # 绘制黑色方格
                axes[1].add_patch(plt.Rectangle((i, j), 1, 1, color='black'))
                # 写入文字 4.44，并确保字体颜色清晰显示在黑色方格上
                axes[1].text(i + 0.5, j + 0.5, '4.44', color='white', ha='center', va='center')
            elif (i==0 and j==1) or (i==0 and j==3) or (i==1 and j==0) or (i==1 and j==4) or (i==3 and j==0) or (i==3 and j==4) or (i==4 and j==1) or (i==4 and j==3):
                # 绘制白色方格
                axes[1].add_patch(plt.Rectangle((i, j), 1, 1, color='white'))
                # 写入文字 4.86，并确保字体颜色清晰显示在白色方格上
                axes[1].text(i + 0.5, j + 0.5, '4.44', color='black', ha='center', va='center')
            elif(i + j) % 2 == 1:
                # 绘制白色方格
                axes[1].add_patch(plt.Rectangle((i, j), 1, 1, color='white'))
                # 写入文字 4.86，并确保字体颜色清晰显示在白色方格上
                axes[1].text(i + 0.5, j + 0.5, '5.55', color='black', ha='center', va='center')
            elif (i==2 and j==2):
                # 绘制白色方格
                axes[1].add_patch(plt.Rectangle((i, j), 1, 1, color='white'))
                # 写入文字 4.86，并确保字体颜色清晰显示在白色方格上
                axes[1].text(i + 0.5, j + 0.5, '2.22', color='black', ha='center', va='center')
            else:
                # 绘制黑色方格
                axes[1].add_patch(plt.Rectangle((i, j), 1, 1, color='black'))
                # 写入文字 4.64，并确保字体颜色清晰显示在黑色方格上
                axes[1].text(i + 0.5, j + 0.5, '5.55', color='white', ha='center', va='center')

    # 设置坐标轴刻度
    axes[1].set_xticks(range(6))
    axes[1].set_yticks(range(6))
    # 隐藏坐标轴刻度标签
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])

    # 显示网格线
    axes[1].grid(True)
    # 绘制截取并放大的部分
    #axes[2].set_title('C Center')
    # 显示图像
    #plt.savefig('data/paper pic/pic8.png')
    plt.savefig('data/paper pic/pic8.jpeg',format='jpeg', dpi=1000,bbox_inches='tight',pad_inches=0.3)
    plt.show()

def i_j_pic(i,j,t):
    # FQ2
    #type_t_minux_matrix=torch.tensor(np.loadtxt('/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1_3/type_t_matrix/type_t_matrix_r=2.7777777777777777_epoches=10000_L=200_T={}_第0次实验数据.txt'.format(t-1))).to('cuda:4')
    type_t_matrix = torch.tensor(np.loadtxt(
        '/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1_3/type_t_matrix/type_t_matrix_r=2.7777777777777777_epoches=10000_L=200_T={}_第0次实验数据.txt'.format(t))).to('cuda:4')
    # Qtable=torch.load('data/Origin_Fermi_Qlearning_extract/Qtable/Qtable_r=2.9_epoches=20000_L=200_第0次实验数据.txt')
    #Q_matrix = torch.load('/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1_3/Q_matrix/Q_matrix_r=2.7777777777777777_epoches=10000_L=200_T={}_第0次实验数据.txt'.format(t)).to('cuda:4')
    profit_matrix = torch.tensor(np.loadtxt(
        '/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1_3/profit_matrix/profit_matrix_r=2.7777777777777777_epoches=10000_L=200_T={}_第0次实验数据.txt'.format(t))).to(
        'cuda:4')
    Q_sa_matrix=torch.tensor(np.loadtxt('/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1_3/Q_sa_matrix/Q_sa_matrix_r=2.7777777777777777_epoches=10000_L=200_T={}_第0次实验数据.txt'.format(t))).to('cuda:4')
    #C_indices = torch.arange(type_t_matrix.numel()).to('cuda:4')
    sub_type_t_matrix = type_t_matrix[i:i + 7, j:j + 7]
    shot_pic1(sub_type_t_matrix, 10000, profit_matrix[i:i + 7, j:j+ 7].cpu().numpy(),t)

def pic_12_shot_pic():
    #print("hello")
    #sub_type_t_matrix=type_t_matrix[i2:i2 +20, j2:j2 +20]
    #shot_pic1(sub_type_t_matrix,10000,profit_matrix[i2:i2 + 20, j2:j2 + 20].cpu().numpy())
    # i1_2=122
    # j1_2=30
    i1_3=185
    j1_3=17
    # type_t_matrix=torch.tensor(np.loadtxt('/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1_3/type_t_matrix/type_t_matrix_r=2.7777777777777777_epoches=10000_L=200_T=4990_第0次实验数据.txt')).to('cuda:4')
    # shot_pic1(type_t_matrix,10000,torch.zeros(200,200).cpu().numpy(),showNum=False)
    for t in range(4985,4990):
        i_j_pic(i1_3,j1_3,t)

if __name__ == '__main__':
    print("analysis_Fermi")
    #type_t_matrix=torch.zeros(200,200).to('cuda:4')
    #Q_matrix=torch.load('/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1_3/Q_matrix/Q_matrix_r=2.7777777777777777_epoches=10000_L=200_T=4991_第0次实验数据.txt').to('cuda:4')
    #indices = torch.arange(type_t_matrix.numel()).reshape(200, 200)[i1_3:i1_3 + 7, j1_3:j1_3 + 7].reshape(7 * 7).to('cpu')
    #sub_Q_matrix=Q_matrix[indices]
    #print(indices)
    #print(sub_Q_matrix)

    # type_t_equal=(type_t_matrix4980==type_t_matrix4981)
    # shot_pic1(type_t_matrix4981,10000,profit_matrix4980.cpu().numpy())
    # shot_pic1(type_t_equal,10000,profit_matrix4980.cpu().numpy())




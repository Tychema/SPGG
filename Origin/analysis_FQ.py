import torch


# 画快照
from matplotlib import pyplot as plt
from torch import tensor
import numpy as np

def shot_pic1(type_t_matrix: tensor, i):
    plt.clf()
    plt.close("all")
    # 初始化图表和数据
    fig = plt.figure(figsize=(20, 20))
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
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    image = np.zeros((type_t_matrix.shape[0], type_t_matrix.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        image[type_t_matrix.cpu() == label] = color
    plt.title('Qlearning: ' + f"T:{i}")
    plt.imshow(image, interpolation='None')
    plt.show()
    plt.clf()
    plt.close("all")


def shot_pic2(type_t_matrix: tensor):
    #type_t_matrix[1, 6] = 2
    plt.clf()
    plt.close("all")

    # 初始化图表和数据
    fig, axes = plt.subplots(1, 3, figsize=(90, 30),gridspec_kw={'width_ratios': [1, 1, 1]})
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
                axes[1].text(i + 0.5, j + 0.5, '4.44', color='white', ha='center', va='center',fontsize=60)
            else:
                # 绘制白色方格
                axes[1].add_patch(plt.Rectangle((i, j), 1, 1, color='white'))
                # 写入文字 4.86，并确保字体颜色清晰显示在白色方格上
                axes[1].text(i + 0.5, j + 0.5, '4.44', color='black', ha='center', va='center',fontsize=60)

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
                axes[2].text(i + 0.5, j + 0.5, '4.44', color='black', ha='center', va='center',fontsize=60)
            else:
                # 绘制白色方格
                axes[2].add_patch(plt.Rectangle((i, j), 1, 1, color='black'))
                # 写入文字 4.86，并确保字体颜色清晰显示在白色方格上
                axes[2].text(i + 0.5, j + 0.5, '4.44', color='white', ha='center', va='center',fontsize=60)

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
    plt.savefig('data/paper pic/pic8_new.png')
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
    plt.show()

if __name__ == '__main__':
    type_t_matrix=torch.load('data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1/type_t_matrix/type_t_matrix_r=2.7777777777777777_epoches=50000_L=200_T=50000_第0次实验数据.txt',map_location={'cuda:0': 'cuda:4'})
    #Qtable=torch.load('data/Origin_Fermi_Qlearning_extract/Qtable/Qtable_r=2.9_epoches=20000_L=200_第0次实验数据.txt')
    #profit_matrix=torch.tensor(np.loadtxt('data/Origin_Fermi_Qlearning2/profit_matrix/r=2.777/e=100000_L=200_T=100000_第0次.txt')).to('cuda:4')
    core1=torch.tensor([
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0]
                       ],dtype=torch.float64).to('cuda:4')
    core2=torch.tensor([
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
                       ],dtype=torch.float64).to('cuda:4')
    # 寻找匹配的位置
    matching_indices1 = []
    matching_indices2 = []
    #shot_pic(type_t_matrix=type_t_matrix,i=10000)
    #使用滑动窗口方法逐个子区域进行比较
    for i in range(96):  # 200 - 10 + 1
        for j in range(96):  # 200 - 10 + 1
            sub_matrix = type_t_matrix[i:i + 5, j:j + 5]
            if torch.all(sub_matrix == core1):
                matching_indices1.append((i, j))
    for i in range(96):  # 200 - 10 + 1
        for j in range(96):  # 200 - 10 + 1
            sub_matrix = type_t_matrix[i:i + 5, j:j + 5]
            if torch.all(sub_matrix == core2):
                matching_indices2.append((i, j))
    #print(matching_indices)
    i1=35
    j1=21
    i2=39
    j2=58
    #print(type_t_matrix[0:0 + 5, 8:8 + 5])
    type_t_matrix.view(-1)
    C_indices = torch.arange(type_t_matrix.numel()).to('cuda:3').reshape(type_t_matrix.shape[0],type_t_matrix.shape[1])[0:0 + 5, 8:8 + 5].reshape(5*5)
    #print(C_indices)
    #print(Qtable[C_indices])
    # print(profit_matrix[i1:i1 + 5, j1:j1 + 5])
    # print(type_t_matrix[i1:i1 + 5, j1:j1 + 5])
    # print(profit_matrix[i2:i2 + 5, j2:j2 + 5])
    # print(type_t_matrix[i2:i2 + 5, j2:j2 + 5])
    shot_pic2(type_t_matrix=type_t_matrix)
    #pic10()
    #shot_pic2(type_t_matrix=type_t_matrix,i=i2,j=j2)
    #
    #print(type_t_matrix.shape)
    #print(Qtable.shape)
    #print(profit_matrix.shape)
    #print(profit_matrix[0:0 + 10, 0:0 + 10])
    #print(type_t_matrix[0:0 + 10, 0:0 + 10])

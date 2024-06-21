import numpy as np
import matplotlib.pyplot as plt
from Origin.Function.mkdir import mkdir
import torch
def draw_all_shot_pic(r,epoches,L_num,count,updateMethod,generated):
    try:
        for t in [0,9,99,999,9999,19999]:
            shot_pic(r,epoches,L_num,t,count,updateMethod,generated)
    except:
        for t in [0,1,10,50,100,300,500,800,1000,5000,10000]:
            shot_pic(r,epoches,L_num,t,count,updateMethod,generated)

def draw_all_shot_pic_torch(r,epoches,L_num,count,updateMethod,generated):
    try:
        for t in [0,9,99,999,9999,19999]:
            shot_pic_torch(r,epoches,L_num,t,count,updateMethod,generated)
    except:
        for t in [0,1,10,100,300,500,800,1000,5000,10000]:
            shot_pic_torch(r,epoches,L_num,t,count,updateMethod,generated)

def shot_pic(r,epoches,L_num,t,count,updateMethod,generated):
    type_t_matrix=np.loadtxt('data/{}/shot_pic/r={}/two_type/{}/type_t_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(updateMethod),str(r), str(generated),"type_t_matrix", str(r), str(epoches), str(L_num), str(t), str(count)))
    plt.clf()
    plt.close("all")
    # 初始化图表和数据
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.get_cmap('Set1', 2)
    # 指定图的大小
    #             plt.figure(figsize=(500, 500))  # 10x10的图
    #             plt.matshow(type_t_matrix.cpu().numpy(), cmap=cmap)
    #             plt.colorbar(ticks=[0, 1, 2], label='Color')
    # 显示图片
    # 定义颜色映射
    color_map = {
        #0设置为黑色
        0: (0, 0, 0),  # 黑色
        #1设置为白色
        1: (255, 255, 255),  # 白色
    }
    image = np.zeros((L_num, L_num, 3), dtype=np.uint8)
    for label, color in color_map.items():
        image[type_t_matrix == label] = color
    #plt.title('Qlearning: '+f"T:{i}")
    # 隐藏坐标轴刻度标签
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['bottom'].set_linewidth(1)  # 设置x轴底部线条宽度
    ax.spines['left'].set_linewidth(1)  # 设置y轴左侧线条宽度
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    plt.imshow(image,interpolation='None')
    mkdir('data/{}/shot_pic/r={}/two_type/{}'.format(str(updateMethod),r,str(generated)))
    #plt.savefig('data/{}/shot_pic/r={}/two_type/{}/t={}.png'.format(str(updateMethod),r,str(generated),t))
    plt.savefig('data/{}/shot_pic/r={}/two_type/{}/t={}.jpeg'.format(str(updateMethod), r, str(generated), t),format='jpeg', dpi=1000, pad_inches=0, quality=95)

    #plt.show()
    plt.clf()
    plt.close("all")

def shot_pic_torch(r,epoches,L_num,t,count,updateMethod,generated):
    type_t_matrix=torch.load('data/{}/shot_pic/r={}/two_type/{}/type_t_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(updateMethod),str(r), str(generated),"type_t_matrix", str(r), str(epoches), str(L_num), str(t), str(count)))
    plt.clf()
    plt.close("all")
    # 初始化图表和数据
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.get_cmap('Set1', 2)
    # 指定图的大小
    #             plt.figure(figsize=(500, 500))  # 10x10的图
    #             plt.matshow(type_t_matrix.cpu().numpy(), cmap=cmap)
    #             plt.colorbar(ticks=[0, 1, 2], label='Color')
    # 显示图片
    # 定义颜色映射
    color_map = {
        #0设置为黑色
        0: (0, 0, 0),  # 黑色
        #1设置为白色
        1: (255, 255, 255),  # 白色
    }
    image = np.zeros((L_num, L_num, 3), dtype=np.uint8)
    for label, color in color_map.items():
        image[type_t_matrix.cpu() == label] = color
    #plt.title('Qlearning: '+f"T:{i}")
    # 隐藏坐标轴刻度标签
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['bottom'].set_linewidth(1)  # 设置x轴底部线条宽度
    ax.spines['left'].set_linewidth(1)  # 设置y轴左侧线条宽度
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    plt.imshow(image,interpolation='None')
    mkdir('data/{}/shot_pic/r={}/two_type/{}'.format(str(updateMethod),r,str(generated)))
    #plt.savefig('data/{}/shot_pic/r={}/two_type/{}/t={}.png'.format(str(updateMethod),r,str(generated),t))
    plt.savefig('data/{}/shot_pic/r={}/two_type/{}/t={}.jpeg'.format(str(updateMethod), r, str(generated), t),format='jpeg', dpi=1000, pad_inches=0, quality=95)

    #plt.show()
    plt.clf()
    plt.close("all")
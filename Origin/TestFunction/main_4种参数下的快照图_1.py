import matplotlib.pyplot as plt
import numpy as np
import pickle

from plot_graph import plot_lattice  # 导入自定义模块的相关函数

# 从文件中加载对象,反序列化在特定的时间点保存的图对象
with open('serialize_Snapshots_a_1.pkl', 'rb') as f:
    serialize_Graph_a_1 = pickle.load(f)
with open('serialize_Snapshots_a_2.pkl', 'rb') as f:
    serialize_Graph_a_2 = pickle.load(f)
with open('serialize_Snapshots_b_1.pkl', 'rb') as f:
    serialize_Graph_b_1 = pickle.load(f)
with open('serialize_Snapshots_b_2.pkl', 'rb') as f:
    serialize_Graph_b_2 = pickle.load(f)
all_Graph = [serialize_Graph_a_1, serialize_Graph_a_2, serialize_Graph_b_1, serialize_Graph_b_2]

# 初始化一个4行5列的子图网格
fig, axs = plt.subplots(4, 5, figsize=(15, 12))

# 设置左侧标签
y_labels = ['(a-1)', '(a-2)', '(b-1)', '(b-2)']
# 设置上方标签
time_labels = ['T=1', 'T=10', 'T=30', 'T=50', 'T=1000']
time_labels2 = ['T=1', 'T=5', 'T=14', 'T=27', 'T=1000']
# 遍历每一个子图
for i in range(4):
    for j in range(5):
        plot_lattice(all_Graph[i][j], axs[i, j])  # 可视化方格网络
        axs[i, j].set_aspect('equal', adjustable='box')  # 强制子图为正方形尺寸
        axs[i, j].set_xlim(20, 80)  # 设置横坐标轴的范围
        axs[i, j].set_ylim(20, 80)  # 设置纵坐标轴的范围

        # 对第一列的图设置左侧标签
        if j == 0:
            axs[i, j].set_ylabel(y_labels[i], fontsize=18, fontweight='bold', rotation=0,labelpad=30)
        # 对第一行的图设置上方标签
        if i == 0:
            axs[i, j].set_title(time_labels[j], fontsize=18, fontweight='bold')
        # 对第三行的图设置上方标签
        if i == 2:
            axs[i, j].set_title(time_labels2[j], fontsize=18, fontweight='bold')
        # 隐藏刻度线和刻度标签
        axs[i, j].tick_params(axis='x',  # 对x轴操作
                              which='both',  # 对所有刻度（主刻度和次刻度）操作
                              bottom=False,  # 不显示底部刻度线
                              top=False,  # 不显示顶部刻度线
                              labelbottom=False)  # 不显示底部刻度标签
        axs[i, j].tick_params(axis='y',  # 对y轴操作
                              which='both',  # 对所有刻度操作
                              left=False,  # 不显示左侧刻度线
                              right=False,  # 不显示右侧刻度线
                              labelleft=False)  # 不显示左侧刻度标签
# 调整子图之间的间距
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.15)

# 显示图表
plt.savefig('./Figures/4种参数下的快照图_1.png')

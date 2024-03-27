import matplotlib.pyplot as plt
import numpy as np

# 时间序列
gen_list = np.loadtxt('gen_list.txt')

# 第一张图的三组数据
a1_C = np.loadtxt('fraction_C_10次实验数据的平均值.txt')
a1_D = np.loadtxt('fraction_D_10次实验数据的平均值.txt')

# 第二张图的三组数据
# a2_C = np.loadtxt('a2_C_10次实验数据的平均值.txt')
# a2_D = np.loadtxt('a2_D_10次实验数据的平均值.txt')
# a2_PC = np.loadtxt('a2_PC_10次实验数据的平均值.txt')

# 第三张图的三组数据
# b1_C = np.loadtxt('b1_C_10次实验数据的平均值.txt')
# b1_D = np.loadtxt('b1_D_10次实验数据的平均值.txt')
# b1_PC = np.loadtxt('b1_PC_10次实验数据的平均值.txt')

# 第四张图的三组数据
# b2_C = np.loadtxt('b2_C_10次实验数据的平均值.txt')
# b2_D = np.loadtxt('b2_D_10次实验数据的平均值.txt')
# b2_PC = np.loadtxt('b2_PC_10次实验数据的平均值.txt')

# 创建一个图形和4个子图，布局为2行2列
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# 绘制第一张图的三条折线
axs[0, 0].plot(gen_list, a1_C, '-', color='blue')
axs[0, 0].plot(gen_list, a1_D, '-', color='red')
axs[0, 0].legend(['C', 'D'],fontsize=14, frameon=True, loc='upper left')
axs[0, 0].text(0.95, 0.95, '(a-1)',fontsize=14, ha='right', va='top', transform=axs[0, 0].transAxes)
axs[0, 0].set_ylabel('Fractions', fontsize=16)
axs[0, 0].set_xticklabels([])
# 隐藏x轴刻度标签
# axs[0, 0].tick_params(axis='x', bottom=False, labelbottom=False)  # 第一行第一列
axs[0, 0].set_ylim([-0.05,1.05]) # 在y轴的两端各留出一些空间
axs[0, 0].set_xscale('log')

# 绘制第二张图的三条折线
# axs[0, 1].plot(gen_list, a2_C, '-', color='blue')
# axs[0, 1].plot(gen_list, a2_D, '-', color='red')
# axs[0, 1].plot(gen_list, a2_PC, '-', color='yellow')
# axs[0, 1].text(0.95, 0.95, '(a-2)',fontsize=14, ha='right', va='top', transform=axs[0, 1].transAxes)
# axs[0, 1].set_xticklabels([])
# axs[0, 1].set_yticks([])  # 移除y轴的刻度线和标签
# 隐藏x轴刻度标签
# axs[0, 1].tick_params(axis='x', bottom=False, labelbottom=False)  # 第一行第一列
# axs[0, 1].set_ylim([-0.05,1.05]) # 在y轴的两端各留出一些空间
# axs[0, 1].set_xscale('log')

# 绘制第三张图的三条折线
# axs[1, 0].plot(gen_list, b1_C, '-', color='blue')
# axs[1, 0].plot(gen_list, b1_D, '-', color='red')
# axs[1, 0].plot(gen_list, b1_PC, '-', color='yellow')
# axs[1, 0].text(0.95, 0.95, '(b-1)',fontsize=14, ha='right', va='top', transform=axs[1, 0].transAxes)
# axs[1, 0].set_ylabel('Fractions', fontsize=16)
# axs[1, 0].set_xlabel('iterations', fontsize=16)
# axs[1, 0].set_ylim([-0.05,1.05]) # 在y轴的两端各留出一些空间
# axs[1, 0].set_xscale('log')

# 绘制第四张图的三条折线
# axs[1, 1].plot(gen_list, b2_C, '-', color='blue')
# axs[1, 1].plot(gen_list, b2_D, '-', color='red')
# axs[1, 1].plot(gen_list, b2_PC, '-', color='yellow')
# axs[1, 1].text(0.95, 0.95, '(b-2)',fontsize=14, ha='right', va='top', transform=axs[1, 1].transAxes)
# axs[1, 1].set_xlabel('iterations', fontsize=16)
# axs[1, 1].set_yticks([])  # 移除y轴的刻度线和标签
# axs[1, 1].set_ylim([-0.05,1.05]) # 在y轴的两端各留出一些空间
# axs[1, 1].set_xscale('log')
plt.tight_layout()
plt.savefig('./Figures/两策略演化占比折线图.png')

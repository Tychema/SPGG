import torch
import numpy as np
# 假设L_num已经给定，这里我们使用一个示例
from matplotlib import pyplot as plt


colors=['red','green','blue','black']
labels=['D','C']
labels2=['DD','CC','CD','DC']
xticks=[0, 10, 100, 1000, 10000, 100000]
fra_yticks=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95, 1.00]
profite_yticks=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
def read_data(path):
    data = np.loadtxt(path)
    return data

def extract_Qtable(Q_tensor, type_t_matrix):
    C_indices = torch.where(type_t_matrix.squeeze() == 1)[0]
    D_indices = torch.where(type_t_matrix.squeeze() == 0)[0]
    C_Q_table = Q_tensor[C_indices]
    D_indices = Q_tensor[D_indices]
    C_q_mean_matrix = torch.mean(C_Q_table, dim=0)
    D_q_mean_matrix = torch.mean(D_indices, dim=0)
    return C_q_mean_matrix, D_q_mean_matrix

def draw_line1(Loop_num,name,r,epoches=10000,L_num=100,ylim=(0,1)):
    plt.clf()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i=0
    for na in name:
        final_data = np.zeros(epoches+1)
        for count in range(Loop_num):
            data=read_data('data/Origin_Qlearning/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(na, na, r, epoches,
                                                                                      L_num, str(count)))
            final_data=final_data+data
        final_data=final_data/Loop_num
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

if __name__ == '__main__':
    draw_line1(10, ['C_fra', 'D_fra'], 4.000000000000002, 10000, 200, (0, 1))

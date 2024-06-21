import matplotlib.pyplot as plt
import numpy as np
colors=[(217/255,82/255,82/255),(31/255,119/255,180/255),(120/255,122/255,192/255),(161/255,48/255,63/255),'gold','green']

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
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylim(ylim)
    if(type=="line1"):
        plt.xscale('log')
    plt.ylabel(ylabel)
    plt.xlabel(xlable)
    #plt.title(str(updateMethod[7:])+': '+'L='+str(L_num)+' r='+str(r)+' T='+str(epoches))
    # 设置图表的边界留有额外空间，以便容纳外部的文字
    plt.subplots_adjust(left=0.15)  # 调整左边的边界，避免文本被裁剪
    plt.subplots_adjust(left=0.15)  # 调整左边的边界，避免文本被裁剪

    # 在左上角外部添加文字标记 (a)，通过减小x的值使其更靠外
    #offset = -0.05  # 这个值可以根据需要调整，以改变文本离图表边缘的距离
    #plt.text(offset, 1.05, '(a)', transform=ax.transAxes, fontsize=10, va='bottom', ha='left')
    plt.savefig('data/{}/Line_pic/r={}/{}_L={}_r={}_T={}.jpeg'.format(str(updateMethod),str(r), str(updateMethod),str(L_num),str(r), str(epoches)),format='jpeg', dpi=1000, pad_inches=0, quality=95)
    plt.legend()
    plt.pause(0.001)
    plt.clf()
    plt.close("all")


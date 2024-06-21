import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import torch
from draw_pic import crop_image, crop_image_jpeg,add_white_border,add_right_white_pixels
from matplotlib import rcParams
from Origin.analysis_Fermi import shot_pic1,shot_pic2
L_num=200
#colors=['red','green','blue','black']
colors=[(217/255,82/255,82/255),(31/255,119/255,180/255),(120/255,122/255,192/255),(161/255,48/255,63/255),'gold','green']
labels=['D','C']
type_labels=['DD','CC','CDC','StickStrategy']
xticks=[-1,0, 10, 100, 1000, 10000, 100000]
r_xticks=[0,1,2,3,4,5]
fra_yticks= [0,0.2,0.4,0.6,0.8,1]
# profite_yticks=[ 8,10,12,14,16,18,20,22]
profite_yticks=[ 0,2,4,6,8,10,12,14,16,18,20,22]
all_value_sum_yticks=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

def read_data(path):
    data = np.loadtxt(path)
    return data



def pic1():
    loop_num1 = 51
    loop_num2 = 10
    Origin_Fermi = "Origin_Fermi"
    Origin_Qlearning = "Origin_Qlearning"
    epoches = 10000
    F_D_Final_fra = np.array([])
    F_C_Final_fra = np.array([])
    Q_D_Final_fra = np.array([])
    Q_C_Final_fra = np.array([])
    r = 0
    for i in range(loop_num1):
        F_D_Loop_fra = 0
        F_C_Loop_fra = 0
        Q_D_Loop_fra = 0
        Q_C_Loop_fra = 0
        for count in range(loop_num2):
            F_D_Y = read_data(
                'data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(Origin_Fermi, 'D_fra', 'D_fra', r / 10,epoches, L_num, str(count)))
            F_C_Y = read_data(
                'data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(Origin_Fermi, 'C_fra', 'C_fra', r / 10,epoches, L_num, str(count)))
            Q_D_Y = read_data(
                'data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(Origin_Qlearning, 'D_fra', 'D_fra', r / 10,epoches, L_num, str(count)))
            Q_C_Y = read_data(
                'data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(Origin_Qlearning, 'C_fra', 'C_fra', r / 10,epoches, L_num, str(count)))
            F_D_Loop_fra = F_D_Loop_fra + F_D_Y[-1]
            F_C_Loop_fra = F_C_Loop_fra + F_C_Y[-1]
            Q_D_Loop_fra = Q_D_Loop_fra + Q_D_Y[-1]
            Q_C_Loop_fra = Q_C_Loop_fra + Q_C_Y[-1]
        F_D_Final_fra = np.append(F_D_Final_fra, F_D_Loop_fra)
        F_C_Final_fra = np.append(F_C_Final_fra, F_C_Loop_fra)
        Q_D_Final_fra = np.append(Q_D_Final_fra, Q_D_Loop_fra)
        Q_C_Final_fra = np.append(Q_C_Final_fra, Q_C_Loop_fra)
        r = r + 1
    F_D_Final_fra = F_D_Final_fra / loop_num2
    F_C_Final_fra = F_C_Final_fra / loop_num2
    Q_D_Final_fra = Q_D_Final_fra / loop_num2
    Q_C_Final_fra = Q_C_Final_fra / loop_num2
    D_X = np.arange(F_D_Final_fra.shape[0]) / 10
    C_X = np.arange(F_C_Final_fra.shape[0]) / 10

    # 设置子图的布局，调整宽度和高度比例以适应需求
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 增加宽度，减小高度以使子图看起来更宽

    # 子图 (b): 绘制 Q_D_Final_fra 和 Q_C_Final_fra
    axs[0].plot(D_X, Q_D_Final_fra, color=colors[0], marker='o', markersize=7, markerfacecolor='none', linestyle='--',
                linewidth=1.2, markeredgecolor=colors[0], markeredgewidth=1, label='D')
    axs[0].plot(C_X, Q_C_Final_fra, color=colors[1], marker='s', markersize=7, markerfacecolor='none', linestyle='-',
                linewidth=1.2, markeredgecolor=colors[1], markeredgewidth=1, label='C')
    axs[0].set_xticks(r_xticks)
    axs[0].set_yticks(fra_yticks)
    axs[0].set_ylim((0, 1))
    axs[0].set_ylabel('Fractions',fontsize=16)
    axs[0].set_xlabel('r',fontsize=16)

    axs[0].legend()
    # 在子图 (b) 左上角添加 (b) 标记
    #axs[0].text(0, 1.01, '(b)', transform=axs[1].transAxes, fontsize=10, va='bottom', ha='left')  # 修改位置为左上角

    # 子图 (a): 绘制 F_D_Final_fra 和 F_C_Final_fra
    axs[1].plot(D_X, F_D_Final_fra, color=colors[0], marker='o', markersize=5, markerfacecolor='none', linestyle='--',
                linewidth=1.2, markeredgecolor=colors[0], markeredgewidth=1, label='D')
    axs[1].plot(C_X, F_C_Final_fra, color=colors[1], marker='s', markersize=5, markerfacecolor='none', linestyle='-',
                linewidth=1.2, markeredgecolor=colors[1], markeredgewidth=1, label='C')
    axs[1].set_xticks(r_xticks)
    axs[1].set_yticks(fra_yticks)
    axs[1].set_ylim((0, 1))
    if (type == "line1"):
        axs[1].set_xscale('log')
    axs[1].set_ylabel('Fractions',fontsize=16)
    axs[1].set_xlabel('r',fontsize=16)
    axs[1].legend()
    rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    axs[0].set_title('Case Ⅰ',fontsize=30,fontweight=20)
    axs[1].set_title('Case Ⅱ',fontsize=30,fontweight=20)
    fig.text(0.05,0.92,"(a)",fontsize=20,fontweight=20,fontname='Times New Roman')
    fig.text(0.55,0.92,"(b)",fontsize=20,fontweight=20,fontname='Times New Roman')

    # 获取当前的y轴刻度标签
    y_labels0 = axs[0].get_yticklabels()
    x_labels0 = axs[0].get_xticklabels()
    y_labels1 = axs[1].get_yticklabels()
    x_labels1 = axs[1].get_xticklabels()
    i=0
    # 设置y轴刻度标签的字体大小
    for label in y_labels0:
        label.set_fontsize(15)  # 这里14是字体大小，你可以替换为任何你想要的大小
        label.set_fontname('Times New Roman')
        y_labels1[i].set_fontsize(15)
        y_labels1[i].set_fontname('Times New Roman')
        i=i+1
    i=0
    for label in x_labels0:
        label.set_fontsize(15)
        label.set_fontname('Times New Roman')
        x_labels1[i].set_fontsize(15)
        x_labels1[i].set_fontname('Times New Roman')

    # 在子图 (a) 左上角添加 (a) 标记
    #axs[0].text(0, 1.01, '(a)', transform=axs[0].transAxes, fontsize=10, va='bottom', ha='left')  # 修改位置为左上角

    # 调整子图间距，避免标签重叠
    plt.tight_layout()

    # 保存图片
    plt.savefig('data/paper pic/pic1.jpeg',format='jpeg', dpi=1000, pad_inches=0, quality=95)

    # 显示图表
    plt.show()

    # 清理资源（可选）
    plt.clf()
    plt.close("all")

def pic2():
    Qlearning_Line_plus_shotsnap = [
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.6/Origin_Qlearning_L=200_r=3.6_T=10000.jpeg",
        "data/Origin_Qlearning/shot_pic/r=3.6/two_type/generated1_1/t=1.jpeg",
        "data/Origin_Qlearning/shot_pic/r=3.6/two_type/generated1_1/t=10.jpeg",
        "data/Origin_Qlearning/shot_pic/r=3.6/two_type/generated1_1/t=100.jpeg",
        "data/Origin_Qlearning/shot_pic/r=3.6/two_type/generated1_1/t=1000.jpeg",
        "data/Origin_Qlearning/shot_pic/r=3.6/two_type/generated1_1/t=10000.jpeg",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.8/Origin_Qlearning_L=200_r=3.8_T=10000.jpeg",
        "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated1_1/t=1.jpeg",
        "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated1_1/t=10.jpeg",
        "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated1_1/t=100.jpeg",
        "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated1_1/t=1000.jpeg",
        "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated1_1/t=10000.jpeg",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=5.0/Origin_Qlearning_L=200_r=5.0_T=10000.jpeg",
        "data/Origin_Qlearning/shot_pic/r=5.0/two_type/generated1_1/t=1.jpeg",
        "data/Origin_Qlearning/shot_pic/r=5.0/two_type/generated1_1/t=10.jpeg",
        "data/Origin_Qlearning/shot_pic/r=5.0/two_type/generated1_1/t=100.jpeg",
        "data/Origin_Qlearning/shot_pic/r=5.0/two_type/generated1_1/t=1000.jpeg",
        "data/Origin_Qlearning/shot_pic/r=5.0/two_type/generated1_1/t=10000.jpeg",
        ]
    images = [mpimg.imread(path) for path in Qlearning_Line_plus_shotsnap]

    num_rows = 3
    num_cols = 6
    size36=(35, 15)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=size36)
    plt.suptitle('Case Ⅰ', fontsize=80, y=0.98,fontweight=10)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.90, wspace=-0.62, hspace=0.05)
    fig.text(0.091,0.88,"(a)",fontsize=20,fontweight=20,fontname='Times New Roman')
    fig.text(0.091,0.59,"(b)",fontsize=20,fontweight=20,fontname='Times New Roman')
    fig.text(0.091,0.29,"(c)",fontsize=20,fontweight=20,fontname='Times New Roman')
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        if i == num_cols * 0 or i == num_cols * 1 or i == num_cols * 2 or i == num_cols * 3 or i == num_cols * 4:
            cropped_image = crop_image_jpeg(images[i], left_padding=0, right_padding=0, top_padding=0, bottom_padding=0)
            cropped_image = add_right_white_pixels(cropped_image,1200)
            #cropped_image = add_white_border(cropped_image,top=33,right=600)
            ax.imshow(cropped_image)
        else:
            cropped_image = crop_image_jpeg(images[i], left_padding=30, right_padding=0, top_padding=0,bottom_padding=30)
            if i == num_cols * 0+1 or i == num_cols * 1+1 or i == num_cols * 2+1 or i == num_cols * 3+1 or i == num_cols * 4+1:
                cropped_image = add_white_border(cropped_image, left=15,bottom=15)
            ax.imshow(cropped_image, interpolation='None')
        # 绘制图像

        ax.axis('off')

    print('finished!')
    plt.savefig('data/paper pic/pic2.jpeg',format='jpeg', dpi=1000,bbox_inches='tight',pad_inches=0.3)



def pic3():
    Fermi_Line_plus_shotsnap = [
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi/Line_pic/r=3.7/Origin_Fermi__L=200_r=3.7_T=10000.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=1.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=10.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=100.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=1000.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=10000.jpeg",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi/Line_pic/r=3.8/Origin_Fermi_L=200_r=3.8_T=10000.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=1.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=10.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=100.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=1000.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=10000.jpeg",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi/Line_pic/r=5.0/Origin_Fermi_L=200_r=5.0_T=10000.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=1.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=10.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=100.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=1000.jpeg",
        "data/Origin_Fermi/shot_pic/r=3.7/two_type/generated1/t=10000.jpeg",
        ]
    images = [mpimg.imread(path) for path in  Fermi_Line_plus_shotsnap]

    num_rows = 3
    num_cols = 6
    size36=(35, 15)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=size36)
    plt.suptitle('Case Ⅱ', fontsize=80, y=0.98,fontweight=10)
    fig.text(0.091,0.88,"(a)",fontsize=20,fontweight=20,fontname='Times New Roman')
    fig.text(0.091,0.59,"(b)",fontsize=20,fontweight=20,fontname='Times New Roman')
    fig.text(0.091,0.29,"(c)",fontsize=20,fontweight=20,fontname='Times New Roman')
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.90, wspace=-0.62, hspace=0.05)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        if i == num_cols * 0 or i == num_cols * 1 or i == num_cols * 2 or i == num_cols * 3 or i == num_cols * 4:
            cropped_image = crop_image_jpeg(images[i], left_padding=0, right_padding=0, top_padding=0, bottom_padding=0)
            cropped_image = add_right_white_pixels(cropped_image,1200)
            #cropped_image = add_white_border(cropped_image,top=33,right=600)
            ax.imshow(cropped_image)
        else:
            cropped_image = crop_image_jpeg(images[i], left_padding=30, right_padding=0, top_padding=0,bottom_padding=30)
            if i == num_cols * 0+1 or i == num_cols * 1+1 or i == num_cols * 2+1 or i == num_cols * 3+1 or i == num_cols * 4+1:
                cropped_image = add_white_border(cropped_image, left=15,bottom=15)
            ax.imshow(cropped_image, interpolation='None')
        # 绘制图像

        ax.axis('off')

    print('finished!')
    #plt.show(bbox_inches='tight')
    plt.savefig('data/paper pic/pic3.jpeg',format='jpeg', dpi=1000,bbox_inches='tight',pad_inches=0.3)


def pic4():
    generated2_Line_plus_shotsnap = [
                    "data/Origin_Qlearning/Line_pic/r=3.8/Origin_Qlearning_generated2_L=200_r=3.8_T=10000.jpeg",
                    "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated2/t=0.jpeg",
                    "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated2/t=100.jpeg",
                    "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated2/t=500.jpeg",
                    "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated2/t=800.jpeg",
                    "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated2/t=10000.jpeg",
                    "data/Origin_Fermi/Line_pic/r=3.8/Origin_Fermi_generated2_L=200_r=3.8_T=10000.jpeg",
                    "data/Origin_Fermi/shot_pic/r=3.8/two_type/generated2/t=1.jpeg",
                    "data/Origin_Fermi/shot_pic/r=3.8/two_type/generated2/t=100.jpeg",
                    "data/Origin_Fermi/shot_pic/r=3.8/two_type/generated2/t=500.jpeg",
                    "data/Origin_Fermi/shot_pic/r=3.8/two_type/generated2/t=800.jpeg",
                    "data/Origin_Fermi/shot_pic/r=3.8/two_type/generated2/t=10000.jpeg",
                   ]
    images = [mpimg.imread(path) for path in generated2_Line_plus_shotsnap]

    num_rows = 2
    num_cols = 6
    size36=(35, 10)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=size36)
    fig.text(0.5, 0.98, 'Case Ⅰ', fontsize=50, ha='center', va='center', fontweight=10)
    fig.text(0.5, 0.48, 'Case Ⅱ', fontsize=50, ha='center', va='center', fontweight=10)
    fig.text(0.115,0.90,"(a)",fontsize=20,fontweight=20,fontname='Times New Roman')
    fig.text(0.115,0.39,"(b)",fontsize=20,fontweight=20,fontname='Times New Roman')
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.93, wspace=-0.69, hspace=0.30)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        if i == num_cols * 0 or i == num_cols * 1 or i == num_cols * 2 or i == num_cols * 3 or i == num_cols * 4:
            cropped_image = crop_image_jpeg(images[i], left_padding=0, right_padding=0, top_padding=0, bottom_padding=0)
            cropped_image = add_right_white_pixels(cropped_image, 1200)
            # cropped_image = add_white_border(cropped_image,top=33,right=600)
            ax.imshow(cropped_image)
        else:
            cropped_image = crop_image_jpeg(images[i], left_padding=30, right_padding=0, top_padding=0,
                                            bottom_padding=30)
            if i == num_cols * 0 + 1 or i == num_cols * 1 + 1 or i == num_cols * 2 + 1 or i == num_cols * 3 + 1 or i == num_cols * 4 + 1:
                cropped_image = add_white_border(cropped_image, left=15, bottom=15)
            ax.imshow(cropped_image, interpolation='None')
        # 绘制图像

        ax.axis('off')

    print('finished!')
    # plt.show(bbox_inches='tight')
    plt.savefig('data/paper pic/pic4.jpeg', format='jpeg', dpi=1000, bbox_inches='tight', pad_inches=0.3)

def pic5():
    generated3_Line_plus_shotsnap = [
                    "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=4.7/Origin_Qlearning_generated3_L=200_r=4.7_T=10000.jpeg",
                    "data/Origin_Qlearning/shot_pic/r=4.7/two_type/generated3/t=0.jpeg",
                    "data/Origin_Qlearning/shot_pic/r=4.7/two_type/generated3/t=100.jpeg",
                    "data/Origin_Qlearning/shot_pic/r=4.7/two_type/generated3/t=500.jpeg",
                    "data/Origin_Qlearning/shot_pic/r=4.7/two_type/generated3/t=800.jpeg",
                    "data/Origin_Qlearning/shot_pic/r=4.7/two_type/generated3/t=10000.jpeg",

                   ]
    images = [mpimg.imread(path) for path in generated3_Line_plus_shotsnap]

    num_rows = 1
    num_cols = 6
    size36=(40, 6)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=size36)
    fig.text(0.5, 0.98, 'Case Ⅰ', fontsize=50, ha='center', va='center', fontweight=10)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.90, wspace=-0.53, hspace=0.05)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        if i == num_cols * 0 or i == num_cols * 1 or i == num_cols * 2 or i == num_cols * 3 or i == num_cols * 4:
            cropped_image = crop_image_jpeg(images[i], left_padding=0, right_padding=0, top_padding=0, bottom_padding=0)
            cropped_image = add_right_white_pixels(cropped_image, 1300)
            # cropped_image = add_white_border(cropped_image,top=33,right=600)
            ax.imshow(cropped_image)
        else:
            cropped_image = crop_image_jpeg(images[i], left_padding=30, right_padding=0, top_padding=0,
                                            bottom_padding=30)
            if i == num_cols * 0 + 1 or i == num_cols * 1 + 1 or i == num_cols * 2 + 1 or i == num_cols * 3 + 1 or i == num_cols * 4 + 1:
                cropped_image = add_white_border(cropped_image, left=15, bottom=15)
            ax.imshow(cropped_image, interpolation='None')
        # 绘制图像

        ax.axis('off')

    print('finished!')
    # plt.show(bbox_inches='tight')
    plt.savefig('data/paper pic/pic5.jpeg', format='jpeg', dpi=1000, bbox_inches='tight', pad_inches=0.3)

def pic6():
    loop_num1 = 51
    loop_num2 = 10
    Origin_Fermi_Qlearning2 = "Origin_Fermi_Qlearning2"
    epoches = 10000
    L_num = 200
    F_D_Final_fra = np.array([])
    F_C_Final_fra = np.array([])
    r = 0
    for i in range(loop_num1):
        F_D_Loop_fra = 0
        F_C_Loop_fra = 0
        for count in range(loop_num2):
            F_D_Y = read_data(
                'data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(Origin_Fermi_Qlearning2, 'D_fra', 'D_fra', r / 10,epoches, L_num, str(count)))
            F_C_Y = read_data(
                'data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(Origin_Fermi_Qlearning2, 'C_fra', 'C_fra', r / 10,epoches, L_num, str(count)))
            F_D_Loop_fra = F_D_Loop_fra + F_D_Y[-1]
            F_C_Loop_fra = F_C_Loop_fra + F_C_Y[-1]
        F_D_Final_fra = np.append(F_D_Final_fra, F_D_Loop_fra)
        F_C_Final_fra = np.append(F_C_Final_fra, F_C_Loop_fra)
        r = r + 1
    F_D_Final_fra = F_D_Final_fra / loop_num2
    F_C_Final_fra = F_C_Final_fra / loop_num2
    D_X = np.arange(F_D_Final_fra.shape[0]) / 10
    C_X = np.arange(F_C_Final_fra.shape[0]) / 10

    # 设置子图的布局，调整宽度和高度比例以适应需求
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))  # 增加宽度，减小高度以使子图看起来更宽
    plt.subplots_adjust(top=0.8)
    # 子图 (a): 绘制 F_D_Final_fra 和 F_C_Final_fra
    plt.plot(D_X, F_D_Final_fra, color=colors[0], marker='o', markersize=5, markerfacecolor='none', linestyle='--',linewidth=1.2, markeredgecolor=colors[0], markeredgewidth=1, label='D')
    plt.plot(C_X, F_C_Final_fra, color=colors[1], marker='s', markersize=5, markerfacecolor='none', linestyle='-',linewidth=1.2, markeredgecolor=colors[1], markeredgewidth=1, label='C')
    plt.xticks(r_xticks)
    plt.yticks(fra_yticks)
    plt.ylim((0, 1))
    plt.ylabel('Fractions',fontsize=16)
    plt.xlabel('r',fontsize=16)
    plt.legend()
    fig.text(0.55, 0.94, 'Case Ⅲ', fontsize=30, ha='center', va='center', fontweight=20)
    fig.suptitle(' ', fontsize=20, fontweight=10, va='top',ha='center')

    # 获取当前的y轴刻度标签
    y_labels0 = axs.get_yticklabels()
    x_labels0 = axs.get_xticklabels()


    # 设置y轴刻度标签的字体大小
    for label in y_labels0:
        label.set_fontsize(15)  # 这里14是字体大小，你可以替换为任何你想要的大小
        label.set_fontname('Times New Roman')
    for label in x_labels0:
        label.set_fontsize(15)
        label.set_fontname('Times New Roman')

    # 在子图 (a) 左上角添加 (a) 标记
    #axs[0].text(0, 1.01, '(a)', transform=axs[0].transAxes, fontsize=10, va='bottom', ha='left')  # 修改位置为左上角

    # 调整子图间距，避免标签重叠
    plt.tight_layout()

    # 保存图片
    plt.savefig('data/paper pic/pic6.jpeg',format='jpeg', dpi=1000, pad_inches=0)
    # 显示图表
    plt.show()

    # 清理资源（可选）
    plt.clf()
    plt.close("all")

def pic7():
    FQ_Line_plus_shotsnap = [
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/Line_pic/r=2.5/Origin_Fermi_Qlearning2_L=200_r=2.5_T=10000.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.5/two_type/generated1/t=1.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.5/two_type/generated1/t=10.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.5/two_type/generated1/t=100.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.5/two_type/generated1/t=1000.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.5/two_type/generated1/t=10000.jpeg",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/Line_pic/r=2.7777777777777777/Origin_Fermi_Qlearning2_L=200_r=2.7777777777777777_T=10000.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1/t=1.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1/t=10.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1/t=100.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1/t=1000.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1/t=10000.jpeg",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/Line_pic/r=3.3/Origin_Fermi_Qlearning2_L=200_r=3.3_T=10000.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=1.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=10.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=100.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=1000.jpeg",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=10000.jpeg",
        ]
    images = [mpimg.imread(path) for path in FQ_Line_plus_shotsnap]

    num_rows = 3
    num_cols = 6
    size36=(35, 15)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=size36)
    plt.suptitle('Case Ⅲ', fontsize=80, y=0.98,fontweight=10)
    fig.text(0.091,0.88,"(a)",fontsize=20,fontweight=20,fontname='Times New Roman')
    fig.text(0.091,0.59,"(b)",fontsize=20,fontweight=20,fontname='Times New Roman')
    fig.text(0.091,0.29,"(c)",fontsize=20,fontweight=20,fontname='Times New Roman')
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.90, wspace=-0.62, hspace=0.05)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        if i == num_cols * 0 or i == num_cols * 1 or i == num_cols * 2 or i == num_cols * 3 or i == num_cols * 4:
            cropped_image = crop_image_jpeg(images[i], left_padding=0, right_padding=0, top_padding=0, bottom_padding=0)
            cropped_image = add_right_white_pixels(cropped_image,1200)
            #cropped_image = add_white_border(cropped_image,top=33,right=600)
            ax.imshow(cropped_image)
        else:
            cropped_image = crop_image_jpeg(images[i], left_padding=30, right_padding=0, top_padding=0,bottom_padding=30)
            if i == num_cols * 0+1 or i == num_cols * 1+1 or i == num_cols * 2+1 or i == num_cols * 3+1 or i == num_cols * 4+1:
                cropped_image = add_white_border(cropped_image, left=15,bottom=15)
            ax.imshow(cropped_image, interpolation='None')
        # 绘制图像

        ax.axis('off')

    print('finished!')
    #plt.show(bbox_inches='tight')
    plt.savefig('data/paper pic/pic7.jpeg',format='jpeg', dpi=1000,bbox_inches='tight',pad_inches=0.3)

def pic_tuancu():

    #Fermi
    #type_t_matrix=torch.load('data/Origin_Fermi/shot_pic/r=3.8/generated1_2/type_t_matrix/type_t_matrix_r=3.8_epoches=10000_L=200_第0次实验数据.txt',map_location={'cuda:0': 'cuda:4'})
    type_t_matrix=torch.tensor(np.loadtxt('data/Origin_Fermi/shot_pic/r=3.8/two_type/generated1_2/type_t_matrix/type_t_matrix_r=3.8_epoches=10000_L=200_第0次实验数据.txt')).to('cuda:4')
    #Qtable=torch.load('data/Origin_Fermi_Qlearning_extract/Qtable/Qtable_r=2.9_epoches=20000_L=200_第0次实验数据.txt')
    profit_matrix=torch.tensor(np.loadtxt('data/Origin_Fermi/shot_pic/r=3.8/two_type/generated1_2/profit_matrix/profit_matrix_r=3.8_epoches=10000_L=200_第0次实验数据.txt')).to('cuda:4')
    i2=105
    j2=97
    sub_type_t_matrix=type_t_matrix[i2:i2 +20, j2:j2 +20]
    shot_pic1(sub_type_t_matrix,10000,profit_matrix[i2:i2 + 20, j2:j2 + 20].cpu().numpy())

def pic9():
    type_t_matrix = torch.load('data/Origin_Fermi_Qlearning2/shot_pic/r=2.7777777777777777/two_type/generated1/type_t_matrix/type_t_matrix_r=2.7777777777777777_epoches=10000_L=200_T=10000_第0次实验数据.txt').to('cuda:4')
    shot_pic2(type_t_matrix)

def pic_eighType():
    #from draw_line_FQlearning_EightType import draw_EightType
    eightType=[ "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Line_pic/r=4.7/Origin_Qlearning_EightType_L=200_r=4.7_T=10000.jpeg",
                "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Line_pic/r=2.7777777777777777/Origin_Fermi_Qlearning2_EightType_L=200_r=2.7777777777777777_T=10000.jpeg"]
    images = [mpimg.imread(path) for path in eightType]

    num_cols = 2
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 增加宽度，减小高度以使子图看起来更宽

    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.90, wspace=-0.15, hspace=0.05)
    fig.text(0.125,0.905,"(a)",fontsize=20,fontweight=20,fontname='Times New Roman')
    fig.text(0.56,0.905,"(b)",fontsize=20,fontweight=20,fontname='Times New Roman')
    axs[0].set_title('Case Ⅰ',fontsize=30,fontweight=20)
    axs[1].set_title('Case Ⅲ',fontsize=30,fontweight=20)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        cropped_image = crop_image_jpeg(images[i], left_padding=0, right_padding=0, top_padding=0, bottom_padding=0)
        # cropped_image = add_right_white_pixels(cropped_image,1200)
        #cropped_image = add_white_border(cropped_image,top=33,right=600)
        ax.imshow(cropped_image)
        # 绘制图像
        ax.axis('off')

    print('finished!')
    #plt.show(bbox_inches='tight')
    plt.savefig('data/paper pic/pic_eighType.jpeg',format='jpeg', dpi=1000,bbox_inches='tight',pad_inches=0.3)
pic_eighType()
def pic_com_zhuzhuang(names1, names2, r, updateMethod, epoches=10000, L_num=200):
    loop_num=0
    colors = [(217 / 255, 82 / 255, 82 / 255), (31 / 255, 119 / 255, 180 / 255)]

    # 准备数据和绘图
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))  # 初始化一个包含两个子图的figure
    plt.subplots_adjust(top=0.90)  # 调整子图之间的间距
    fig.text(0.5, 0.95, 'Case Ⅲ', ha='center', va='center', fontsize=40,fontweight=10)  # 添加x轴标签
    # 简化后的数据6
    D_data = {
        'Condition': ['Cond1', 'Cond2'],
        'D': [20, 22],
        'C': [15, 17],
    }
    C_data = {
        'Condition': ['Cond1', 'Cond2'],
        'D': [20, 22],
        'C': [15, 17],
    }
    D_df = pd.DataFrame(D_data)
    C_df = pd.DataFrame(C_data)
    # 假设的读取数据过程，实际应替换为正确的数据读取逻辑
    D_DD = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names1[0], names1[0], r, epoches,L_num, str(loop_num)))[-1]
    D_DC = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names1[1], names1[1], r, epoches,L_num, str(loop_num)))[-1]
    D_CD = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names1[2], names1[2], r, epoches,L_num, str(loop_num)))[-1]
    D_CC = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names1[3], names1[3], r, epoches,L_num, str(loop_num)))[-1]
    C_DD = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names2[0], names2[0], r, epoches,L_num, str(loop_num)))[-1]
    C_DC = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names2[1], names2[1], r, epoches,L_num, str(loop_num)))[-1]
    C_CD = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names2[2], names2[2], r, epoches,L_num, str(loop_num)))[-1]
    C_CC = read_data('data/{}/{}/{}_r={}_epoches={}_L={}_第{}次实验数据.txt'.format(str(updateMethod), names2[3], names2[3], r, epoches,L_num, str(loop_num)))[-1]
    D_df['D'] = [D_DD, D_DC]
    D_df['C'] = [D_CD, D_CC]
    C_df['D'] = [C_DD, C_DC]
    C_df['C'] = [C_CD, C_CC]

    # 绘制柱状图5
    bar_width = 0.25
    gap = 0.23
    r1 = np.arange(len(D_df['D']))
    r2 = [x + bar_width + gap for x in r1]
    bars1 = axs[0].bar(r1, D_df['D'], color=colors[0], width=bar_width, edgecolor='grey')
    bars2 = axs[0].bar(r2, D_df['C'], color=colors[1], width=bar_width, edgecolor='grey')
    bars3 = axs[1].bar(r1, C_df['D'], color=colors[0], width=bar_width, edgecolor='grey')
    bars4 = axs[1].bar(r2, C_df['C'], color=colors[1], width=bar_width, edgecolor='grey')
    axs[0].set_ylabel('Q')
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[1].set_ylabel('Q')
    axs[0].bar_label(bars1,padding=3)
    axs[0].bar_label(bars2,padding=3)
    axs[1].bar_label(bars3,padding=3)
    axs[1].bar_label(bars4,padding=3)

    # 在图表下方创建第一个表格
    table_data_1 = [['D', 'C', 'D', 'C']]  # 用空字符串填充以匹配列宽
    the_table1 = axs[0].table(cellText=table_data_1,
                           colWidths=[0.12] * 4,
                           cellLoc='center',
                           bbox=[0, -0.08, 1, 0.08])  # 调整bbox以适应图表大小
    the_table3 = axs[1].table(cellText=table_data_1,
                           colWidths=[0.12] * 4,
                           cellLoc='center',
                           bbox=[0, -0.08, 1, 0.08])  # 调整bbox以适应图表大小

    the_table1.auto_set_font_size(False)
    the_table1.set_fontsize(15)
    the_table3.auto_set_font_size(False)
    the_table3.set_fontsize(15)

    # 在第一个表格下方创建第二个表格
    table_data_2 = [['D', 'C']]
    the_table2 = axs[0].table(cellText=table_data_2,
                           colWidths=[0.12] * 2,
                           cellLoc='center',
                           bbox=[0, -0.16, 1, 0.08])  # 调整bbox以适应图表大小
    the_table4 = axs[1].table(cellText=table_data_2,
                           colWidths=[0.12] * 2,
                           cellLoc='center',
                           bbox=[0, -0.16, 1, 0.08])  # 调整bbox以适应图表大小

    the_table2.auto_set_font_size(False)
    the_table2.set_fontsize(15)
    the_table4.auto_set_font_size(False)
    the_table4.set_fontsize(15)

    # 调整子图间距
    #plt.tight_layout()
    plt.savefig('data/paper pic/pic_com_zhuzhuang.jpeg',format='jpeg', dpi=1000, pad_inches=0.3,bbox_inches='tight')
    # 显示图表
    plt.show()

def pic_heat_pic():
    heat_pic=[ "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/paper pic/r=2.9_eta=0-1_gamma=0-1的热图数据1.jpeg",
                "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/paper pic/r=2.9_eta=0-1_gamma=0-1的热图数据2.jpeg"]
    images = [mpimg.imread(path) for path in heat_pic]

    num_cols = 2
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 增加宽度，减小高度以使子图看起来更宽

    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.98, wspace=-0.18, hspace=0.05)
    fig.text(0.5, 1.03, 'Case Ⅲ', fontsize=40, ha='center', va='center', fontweight=10)
    #axs[0].set_title('Case Ⅰ',fontsize=30,fontweight=20)
    #axs[1].set_title('Case Ⅲ',fontsize=30,fontweight=20)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        cropped_image = crop_image_jpeg(images[i], left_padding=0, right_padding=0, top_padding=0, bottom_padding=0)
        # cropped_image = add_right_white_pixels(cropped_image,1200)
        #cropped_image = add_white_border(cropped_image,top=33,right=600)
        ax.imshow(cropped_image)
        # 绘制图像
        ax.axis('off')

    print('finished!')
    #plt.show(bbox_inches='tight')
    plt.savefig('data/paper pic/heat_pic.jpeg',format='jpeg', dpi=1000,bbox_inches='tight',pad_inches=0.3)

def iteration_pic():
    #from heat_pic2_T import draw_eta_iterations
    #from heat_pic2_T import draw_gamma_iterations
    iteration_pic=[ "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/paper pic/eta_iteration.jpeg",
                "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/paper pic/gamma_iteration.jpeg"]
    images = [mpimg.imread(path) for path in iteration_pic]

    num_cols = 2
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 增加宽度，减小高度以使子图看起来更宽

    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.94, wspace=0, hspace=0.05)
    fig.text(0.5, 1.03, 'Case Ⅲ', fontsize=40, ha='center', va='center', fontweight=10)
    fig.text(0.105,0.94,"(a)",fontsize=20,fontweight=20,fontname='Times New Roman')
    fig.text(0.57,0.94,"(b)",fontsize=20,fontweight=20,fontname='Times New Roman')
    #axs[0].set_title('Case Ⅰ',fontsize=30,fontweight=20)
    #axs[1].set_title('Case Ⅲ',fontsize=30,fontweight=20)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        cropped_image = crop_image_jpeg(images[i], left_padding=0, right_padding=0, top_padding=0, bottom_padding=0)
        # cropped_image = add_right_white_pixels(cropped_image,1200)
        #cropped_image = add_white_border(cropped_image,top=33,right=600)
        ax.imshow(cropped_image)
        # 绘制图像
        ax.axis('off')

    print('finished!')
    plt.savefig('data/paper pic/iteration_pic.jpeg',format='jpeg', dpi=1000,bbox_inches='tight',pad_inches=0.3)
    plt.show(bbox_inches='tight')

def pic12():
    #import analysis_FQ
    pic12=[ "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/paper pic/r=2.77_epoches=10000_L=200_T=4985.jpeg",
            "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/paper pic/r=2.77_epoches=10000_L=200_T=4986.jpeg",
            "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/paper pic/r=2.77_epoches=10000_L=200_T=4987.jpeg",
            "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/paper pic/r=2.77_epoches=10000_L=200_T=4988.jpeg",
            "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/paper pic/r=2.77_epoches=10000_L=200_T=4989.jpeg",

    ]
    images = [mpimg.imread(path) for path in pic12]

    num_rows = 1
    num_cols = 5
    size36 = (40,7)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=size36)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.94, wspace=-0.5, hspace=0.05)
    fig.text(0.5, 0.98, 'Case Ⅲ', fontsize=50, ha='center', va='center', fontweight=10)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        cropped_image = crop_image_jpeg(images[i], left_padding=30, right_padding=0, top_padding=0,
                                            bottom_padding=30)
        ax.imshow(cropped_image, interpolation='None')
        # 绘制图像

        ax.axis('off')

    print('finished!')
    plt.savefig('data/paper pic/pic12.jpeg', format='jpeg', dpi=1000, bbox_inches='tight', pad_inches=0.3)
    #plt.show(bbox_inches='tight')


# if __name__ == '__main__':
#    #pic_eighType()
#
#    pic_com_zhuzhuang(["Q_D_DD", "Q_D_DC", "Q_D_CD", "Q_D_CC"], ["Q_C_DD", "Q_C_DC", "Q_C_CD", "Q_C_CC"], 25/9, "Origin_Fermi_Qlearning2", epoches=50000, L_num=200)
#    #pic12()
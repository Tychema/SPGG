import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def read_data(path):
    data = np.loadtxt(path)
    return data
def crop_image(image, left_padding=400, right_padding=400, top_padding=400, bottom_padding=400):
    # 计算非零像素的最小外接矩形
    nonzero_rows = np.any(image != [255, 255, 255, 255], axis=(1, 2))
    nonzero_cols = np.any(image != [255, 255, 255, 255], axis=(0, 2))
    min_row, max_row = np.where(nonzero_rows)[0][[0, -1]]
    min_col, max_col = np.where(nonzero_cols)[0][[0, -1]]
    # 裁剪图像
    cropped_image = image[min_row+top_padding:max_row+1-bottom_padding,
                          min_col+left_padding:max_col+1-right_padding]
    return cropped_image


def add_right_white_pixels(image, pixels=10):
    # # 检查图像的通道数，以正确设置白色值
    # if image.shape[2] == 4:  # RGBA图像
    #     white_value = [255, 255, 255, 255]
    # else:  # RGB图像
    #     white_value = [255, 255, 255]

    # 创建一个新的图像数组，宽度增加了指定的像素数，高度和通道数保持不变
    new_width = image.shape[1] + pixels
    new_image = np.ones((image.shape[0], new_width, image.shape[2]), dtype=image.dtype) * 255  # 先设为全白

    # 将原始图像复制到新图像的左侧
    new_image[:, :image.shape[1]] = image

    # 右侧的像素已经是白色，如果需要特定处理可以在这里添加代码

    return new_image

if __name__ == '__main__':
    value_line=[
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.6/all_value_ave_L200_r=3.6_T=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.7/all_value_ave_L200_r=3.7_T=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=4.7/all_value_ave_L200_r=4.7_T=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=5.0/all_value_ave_L200_r=5.0_T=10000.png"
    ]

    FQ_generated2_Line_plus_shotsnap=[
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/Line_pic/r=3.3/Fermi_Qlearning2 L=200_r=3.3_T=10000.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=1.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=10.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=100.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=1000.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=5000.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=3.3/two_type/generated1/t=10000.png",
    ]
    Fermi_Qlearning_FQ_generated1_Line_plus_shotsnap=[
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi/Line_pic/r=2.9/Fermi L=200 r=2.9 T=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi/Line_pic/r=2.9/Fermi_value L=200 r=2.9 T=10000.png",
        "data/Origin_Fermi/shot_pic/r=2.9/generated1/t=1.png",
        "data/Origin_Fermi/shot_pic/r=2.9/generated1/t=10.png",
        "data/Origin_Fermi/shot_pic/r=2.9/generated1/t=100.png",
        "data/Origin_Fermi/shot_pic/r=2.9/generated1/t=1000.png",
        "data/Origin_Fermi/shot_pic/r=2.9/generated1/t=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=2.9/Qlearning L=200 r=2.9 T=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=2.9/Qlearning_value L=200 r=2.9 T=10000.png",
        "data/Origin_Qlearning/shot_pic/r=2.9/two_type/generated1/t=1.png",
        "data/Origin_Qlearning/shot_pic/r=2.9/two_type/generated1/t=10.png",
        "data/Origin_Qlearning/shot_pic/r=2.9/two_type/generated1/t=100.png",
        "data/Origin_Qlearning/shot_pic/r=2.9/two_type/generated1/t=1000.png",
        "data/Origin_Qlearning/shot_pic/r=2.9/two_type/generated1/t=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/Line_pic/r=2.9/Fermi_Qlearning2 L=200_r=2.9_T=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/Line_pic/r=2.9/Fermi_Qlearning2_value L=200_r=2.9_T=10000.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.9/two_type/generated1/t=1.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.9/two_type/generated1/t=10.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.9/two_type/generated1/t=100.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.9/two_type/generated1/t=1000.png",
        "data/Origin_Fermi_Qlearning2/shot_pic/r=2.9/two_type/generated1/t=10000.png",
    ]
    Fermi_Qlearning2_r29_Qtable=[
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/Line_pic/r=2.9/Fermi_Qlearning2_value L=200_r=2.9_T=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/Line_pic/r=2.9/Fermi_Qlearning2_QtableD L=200_r=2.9_T=20000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Fermi_Qlearning2/Line_pic/r=2.9/Fermi_Qlearning2_QtableC L=200_r=2.9_T=20000.png"
    ]

    #images = [mpimg.imread(path) for path in Qlearning_Line_plus_shotsnap]
    #images = [mpimg.imread(path) for path in Fermi_Line_plus_shotsnap]
    images = [mpimg.imread(path) for path in Fermi_Qlearning2_r29_Qtable]
    #images = [mpimg.imread(path) for path in generated3_Line_plus_shotsnap]
    #images = [mpimg.imread(path) for path in value_line]


    # 设置子图的排列方式和间距
    num_rows = 1
    num_cols = 3
    size37=(160,60)
    size13=(60,20)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=size13)
    #plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97, wspace=-0.4, hspace=0.03)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97, wspace=-0.2, hspace=0.03)
    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        if i==num_cols*0 or i==num_cols*1 or i==num_cols*2 or i==num_cols*3 or i==num_cols*4 or i==1 or i==2:
            cropped_image = crop_image(images[i], left_padding=0, right_padding=0, top_padding=53, bottom_padding=20)
            cropped_image=add_right_white_pixels(cropped_image, pixels=80)
            #cropped_image = add_white_border(cropped_image,top=33)
        # elif i==num_cols*0+1 or i==num_cols*1+1 or i==num_cols*2+1 or i==num_cols*3+1 or i==num_cols*4+1:
        #     cropped_image = crop_image(images[i], left_padding=20, right_padding=0, top_padding=51, bottom_padding=20)
        #     cropped_image=add_right_white_pixels(cropped_image, pixels=80)
        # else:
        #     cropped_image = crop_image(images[i], left_padding=143, right_padding=125, top_padding=55, bottom_padding=51)
        #     #cropped_image = add_white_border(cropped_image, left=36, top=35,bottom=34)
        # if num_rows==2 and num_cols==2:
        #     cropped_image = crop_image(images[i], left_padding=20, right_padding=0, top_padding=51, bottom_padding=20)
        #     cropped_image=add_right_white_pixels(cropped_image, pixels=80)

        # 绘制图像
        ax.imshow(cropped_image)
        ax.axis('off')
    plt.savefig('data/Origin_Fermi_Qlearning2/ Fermi_Qlearning2_r29_Qtable.png')
    plt.show()

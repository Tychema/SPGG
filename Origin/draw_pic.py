import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

def read_data(path):
    data = np.loadtxt(path)
    return data


def crop_image_jpeg(image, left_padding=400, right_padding=400, top_padding=400, bottom_padding=400):
    # 使用PIL加载JPEG图像
    # 对于JPEG图像，我们只需要处理RGB三个通道
    nonzero_rows = np.any(image != [255, 255, 255], axis=(1, 2))
    nonzero_cols = np.any(image != [255, 255, 255], axis=(0, 2))
    min_row, max_row = np.where(nonzero_rows)[0][[0, -1]]
    min_col, max_col = np.where(nonzero_cols)[0][[0, -1]]
    # 调整裁剪范围以适应JPEG图像
    cropped_image = image[min_row + top_padding:max_row + 1 - bottom_padding,
                    min_col + left_padding:max_col + 1 - right_padding, :]
    # 如果需要，可以将裁剪后的numpy数组转回PIL图像或直接保存为文件
    # cropped_img = Image.fromarray(cropped_image)
    # cropped_img.save('cropped_image.jpg')
    return cropped_image


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
    # 检查图像的通道数，以正确设置白色值
    if image.shape[2] == 4:  # RGBA图像
        white_value = [255, 255, 255, 255]
    else:  # RGB图像
        white_value = [255, 255, 255]

    # 创建一个新的图像数组，宽度增加了指定的像素数，高度和通道数保持不变
    new_width = image.shape[1] + pixels
    new_image = np.ones((image.shape[0], new_width, image.shape[2]), dtype=image.dtype) * 255  # 先设为全白

    # 将原始图像复制到新图像的左侧
    new_image[:, :image.shape[1]] = image

    # 右侧的像素已经是白色，如果需要特定处理可以在这里添加代码

    return new_image

def add_white_border(image, top=0, bottom=0, left=0, right=0):
    image=image.copy()
    # 检查图像是否为RGBA（4通道）或RGB（3通道），并设置相应的白色值
    if image.shape[2] == 4:  # 对于RGBA图像
        white_value = [255, 255, 255, 255]
    else:  # 对于RGB图像
        white_value = [255, 255, 255]

    # 设置顶部和底部的白边
    if top > 0:
        image[:top, :] = white_value
    if bottom > 0:
        image[-bottom:, :] = white_value

    # 设置左侧和右侧的白边
    if left > 0:
        image[:, :left] = white_value
    if right > 0:
        image[:, -right:] = white_value

    return image

if __name__ == '__main__':
    # 读入图片
    Qlearning_Line_plus_shotsnap = ["/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.6/Qlearning_L200r=3.6 T=10000.png",
                                    "data/Origin_Qlearning/shot_pic/r=3.6/two_type/t=1.png",
                                    "data/Origin_Qlearning/shot_pic/r=3.6/two_type/t=10.png",
                                    "data/Origin_Qlearning/shot_pic/r=3.6/two_type/t=100.png",
                                    "data/Origin_Qlearning/shot_pic/r=3.6/two_type/t=1000.png",
                                    "data/Origin_Qlearning/shot_pic/r=3.6/two_type/t=10000.png",
                                    "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=4.7/Qlearning_L200 r=4.7 T=10000.png",
                                    "data/Origin_Qlearning/shot_pic/r=4.7/two_type/t=1.png",
                                    "data/Origin_Qlearning/shot_pic/r=4.7/two_type/t=10.png",
                                    "data/Origin_Qlearning/shot_pic/r=4.7/two_type/t=100.png",
                                    "data/Origin_Qlearning/shot_pic/r=4.7/two_type/t=1000.png",
                                    "data/Origin_Qlearning/shot_pic/r=4.7/two_type/t=10000.png",
                                    "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=5.0/Qlearning_L200r=5.0 T=10000.png",
                                    "data/Origin_Qlearning/shot_pic/r=5.0/two_type/t=1.png",
                                    "data/Origin_Qlearning/shot_pic/r=5.0/two_type/t=10.png",
                                    "data/Origin_Qlearning/shot_pic/r=5.0/two_type/t=100.png",
                                    "data/Origin_Qlearning/shot_pic/r=5.0/two_type/t=1000.png",
                                    "data/Origin_Qlearning/shot_pic/r=5.0/two_type/t=10000.png",
                   ]

    Fermi_Line_plus_shotsnap = ["/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.6/Fermi_L200 r=3.6 T=10000.png", "data/Origin_Fermi/shot_pic/r=3.6/two_type/t=1.png",
                   "data/Origin_Fermi/shot_pic/r=3.6/two_type/t=10.png", "data/Origin_Fermi/shot_pic/r=3.6/two_type/t=100.png", "data/Origin_Fermi/shot_pic/r=3.6/two_type/t=1000.png",
                   "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.8/Fermi_L200 r=3.8 T=10000.png", "data/Origin_Fermi/shot_pic/r=3.8/two_type/t=1.png",
                   "data/Origin_Fermi/shot_pic/r=3.8/two_type/t=10.png", "data/Origin_Fermi/shot_pic/r=3.8/two_type/t=100.png", "data/Origin_Fermi/shot_pic/r=3.8/two_type/t=1000.png",
                   "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=5.0/Fermi_L200 r=5.0 T=10000.png", "data/Origin_Fermi/shot_pic/r=5.0/two_type/t=1.png",
                   "data/Origin_Fermi/shot_pic/r=5.0/two_type/t=10.png", "data/Origin_Fermi/shot_pic/r=5.0/two_type/t=100.png", "data/Origin_Fermi/shot_pic/r=5.0/two_type/t=1000.png",
                   ]


    generated2_Line_plus_shotsnap = [
                    "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.8/Qlearning_generated2_L200 r=3.8 T=10000.png","data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated2/t=0.png",
                    "data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated2/t=100.png","data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated2/t=500.png","data/Origin_Qlearning/shot_pic/r=3.8/two_type/generated2/t=800.png",
                    "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.8/Fermi_generated2_L200 r=3.8 T=10000.png", "data/Origin_Fermi/shot_pic/r=3.8/generated2/t=1.png",
                   "data/Origin_Fermi/shot_pic/r=3.8/generated2/t=100.png", "data/Origin_Fermi/shot_pic/r=3.8/generated2/t=500.png", "data/Origin_Fermi/shot_pic/r=3.8/generated2/t=800.png",
                   ]

    generated3_Line_plus_shotsnap = [
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=4.7/Qlearning_generated3_L200 r=4.7 T=10000.png","data/Origin_Qlearning/shot_pic/r=4.7/two_type/generated3/t=0.png",
        "data/Origin_Qlearning/shot_pic/r=4.7/two_type/generated3/t=100.png","data/Origin_Qlearning/shot_pic/r=4.7/two_type/generated3/t=500.png","data/Origin_Qlearning/shot_pic/r=4.7/two_type/generated3/t=800.png",
                                    ]

    value_line=[
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.6/all_value_ave_L200_r=3.6_T=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=3.7/all_value_ave_L200_r=3.7_T=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=4.7/all_value_ave_L200_r=4.7_T=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/r=5.0/all_value_ave_L200_r=5.0_T=10000.png"
    ]
    Qlearning_neiborlearning_Line_plus_shotsnap = [
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/Neibor/r=2.4/Qlearning NeiborLearning L=200 r=2.4 T=20000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.4/two_type/generated1_T20000/t=1.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.4/two_type/generated1_T20000/t=10.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.4/two_type/generated1_T20000/t=100.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.4/two_type/generated1_T20000/t=1000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.4/two_type/generated1_T20000/t=10000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.4/two_type/generated1_T20000/t=20000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/Neibor/r=3.3/Qlearning NeiborLearning L=200 r=3.3 T=20000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=1.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=10.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=100.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=1000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=10000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=20000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/Neibor/r=4.2/Qlearning NeiborLearning L=200 r=4.2 T=20000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.2/two_type/generated1_T20000/t=1.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.2/two_type/generated1_T20000/t=10.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.2/two_type/generated1_T20000/t=100.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.2/two_type/generated1_T20000/t=1000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.2/two_type/generated1_T20000/t=10000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.2/two_type/generated1_T20000/t=20000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/Neibor/r=4.3/Qlearning NeiborLearning L=200 r=4.3 T=20000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.3/two_type/generated1_T20000/t=1.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.3/two_type/generated1_T20000/t=10.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.3/two_type/generated1_T20000/t=100.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.3/two_type/generated1_T20000/t=1000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.3/two_type/generated1_T20000/t=10000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=4.3/two_type/generated1_T20000/t=20000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/Neibor/r=5.0/Qlearning NeiborLearning L=200 r=5.0 T=20000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=5.0/two_type/generated1_T20000/t=1.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=5.0/two_type/generated1_T20000/t=10.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=5.0/two_type/generated1_T20000/t=100.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=5.0/two_type/generated1_T20000/t=1000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=5.0/two_type/generated1_T20000/t=10000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=5.0/two_type/generated1_T20000/t=20000.png",
        ]
    Qlearning_neiborlearning_Line_plus_shotsnap2 = [
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/Neibor/r=2.6/Qlearning NeiborLearning L=200 r=2.6 T=20000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.6/two_type/generated1_T20000/t=1.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.6/two_type/generated1_T20000/t=10.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.6/two_type/generated1_T20000/t=100.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.6/two_type/generated1_T20000/t=1000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.6/two_type/generated1_T20000/t=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/Neibor/r=2.8/Qlearning NeiborLearning L=200 r=2.8 T=20000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.8/two_type/generated1_T20000/t=1.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.8/two_type/generated1_T20000/t=10.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.8/two_type/generated1_T20000/t=100.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.8/two_type/generated1_T20000/t=1000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=2.8/two_type/generated1_T20000/t=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/Neibor/r=3.0/Qlearning NeiborLearning L=200 r=3.0 T=20000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.0/two_type/generated1_T20000/t=1.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.0/two_type/generated1_T20000/t=10.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.0/two_type/generated1_T20000/t=100.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.0/two_type/generated1_T20000/t=1000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.0/two_type/generated1_T20000/t=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/Neibor/r=3.2/Qlearning NeiborLearning L=200 r=3.2 T=20000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.2/two_type/generated1_T20000/t=1.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.2/two_type/generated1_T20000/t=10.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.2/two_type/generated1_T20000/t=100.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.2/two_type/generated1_T20000/t=1000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.2/two_type/generated1_T20000/t=10000.png",
        "/rjxy/t0a/teacher01/myj/project/SPGG/Origin/data/Origin_Qlearning/Line_pic/Neibor/r=3.3/Qlearning NeiborLearning L=200 r=3.3 T=20000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=1.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=10.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=100.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=1000.png",
        "data/Origin_Qlearning_NeiborLearning/shot_pic/r=3.3/two_type/generated1_T20000/t=10000.png",
        ]


    #images = [mpimg.imread(path) for path in Qlearning_Line_plus_shotsnap]
    #images = [mpimg.imread(path) for path in Fermi_Line_plus_shotsnap]
    #images = [mpimg.imread(path) for path in generated2_Line_plus_shotsnap]
    #images = [mpimg.imread(path) for path in generated3_Line_plus_shotsnap]
    #images = [mpimg.imread(path) for path in value_line]
    #images=[mpimg.imread(path) for path in Qlearning_neiborlearning_Line_plus_shotsnap]
    images=[mpimg.imread(path) for path in Qlearning_neiborlearning_Line_plus_shotsnap2]


    # 设置子图的排列方式和间距
    num_rows = 5
    num_cols = 6
    Qlearning_Line_plus_shotsnap_size=(40, 20)
    generated2_Line_plus_shotsnap_size=(60, 20)
    generated3_Line_plus_shotsnap_size=(60,10)
    value_line_size=(40,30)
    Qlearning_neiborlearning_Line_plus_shotsnap_size=(100,60)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=Qlearning_neiborlearning_Line_plus_shotsnap_size)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97, wspace=-0.4, hspace=0.03)

    # 在每个子图中绘制图像
    for i, ax in enumerate(axs.flat):
        # 对图像进行裁剪
        if i==num_cols*0 or i==num_cols*1 or i==num_cols*2 or i==num_cols*3 or i==num_cols*4:
            cropped_image = crop_image(images[i], left_padding=20, right_padding=0, top_padding=51, bottom_padding=20)
            cropped_image=add_right_white_pixels(cropped_image, pixels=80)
            #cropped_image = add_white_border(cropped_image,top=33)
        else:
            cropped_image = crop_image(images[i], left_padding=143, right_padding=125, top_padding=55, bottom_padding=51)
            #cropped_image = add_white_border(cropped_image, left=36, top=35,bottom=34)
        if num_rows==2 and num_cols==2:
            cropped_image = crop_image(images[i], left_padding=20, right_padding=0, top_padding=51, bottom_padding=20)
            cropped_image=add_right_white_pixels(cropped_image, pixels=80)

        # 绘制图像
        ax.imshow(cropped_image)
        ax.axis('off')
    plt.savefig('data/Origin_Qlearning_NeiborLearning/Qlearning_NeiborLearning.png')
    plt.show()

from multiprocessing import Process

import numpy as np
import SPGG
def compute_on_one_process(count):
        # 给实例spgg设置参数、创建方格网络、设置人口属性
        spgg = SPGG(r=4, c=1,K=0.1, L=200, iterations=10000, num_of_strategies=2, population_type=0)
        result = []
        # 进行指定迭代次数的博弈
        spgg.run(it_records=result)
        # 分别取出两列数据
        first_column = [record[0] for record in result]
        second_column = [record[1] for record in result]
        np.savetxt('fraction_C_' + '第' + str(count) + '次实验数据.txt',
                   first_column)
        np.savetxt('fraction_D_' + '第' + str(count) + '次实验数据.txt',
                   second_column)

if __name__ == '__main__':
    # 参数设置
    # 用到的CPU核心数
    total_cpu_core = 10
    # 进行10次重复试验取平均值
    process_list = []
    for i in range(total_cpu_core):
        p = Process(target=compute_on_one_process, args=(i+1,))
        process_list.append(p)
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()
    # 对10次实验数据求平均值，保存到一个文件中
    # 文件名列表
    filenames1 = [f'fraction_C_第{i}次实验数据.txt' for i in range(1, 11)]
    # 初始化一个列表，用于存储每个文件同一行的数据
    line_data1 = []
    # 从第一个文件中获取行数
    with open(filenames1[0], 'r') as f:
        for line in f:
            # 对于每一行，初始化一个为0的累加值
            line_data1.append([])
    # 遍历每个文件
    for filename in filenames1:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                # 将当前行的数据转换为浮点数并累加
                line_data1[i].append(float(line.strip()))
    # 计算平均值
    averages1 = [np.mean(line) for line in line_data1]
    # 保存平均值到新文件
    with open('fraction_C_10次实验数据的平均值.txt', 'w') as f:
        for avg in averages1:
            f.write(f"{avg}\n")

    # 对10次实验数据求平均值，保存到一个文件中
    # 文件名列表
    filenames2 = [f'fraction_D_第{i}次实验数据.txt' for i in range(1, 11)]
    # 初始化一个列表，用于存储每个文件同一行的数据
    line_data2 = []
    # 从第一个文件中获取行数
    with open(filenames2[0], 'r') as f:
        for line in f:
            # 对于每一行，初始化一个为0的累加值
            line_data2.append([])
    # 遍历每个文件
    for filename in filenames2:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                # 将当前行的数据转换为浮点数并累加
                line_data2[i].append(float(line.strip()))
    # 计算平均值
    averages2 = [np.mean(line) for line in line_data2]
    # 保存平均值到新文件
    with open('fraction_D_10次实验数据的平均值.txt', 'w') as f:
        for avg in averages2:
            f.write(f"{avg}\n")
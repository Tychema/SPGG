# 开发人员:毕研政
# 开发时间:2024/3/9 15:46
import numpy as np
from multiprocessing import Process

overlap5 = lambda A: A + np.roll(A, -1, 0) + np.roll(A, 1, 0) + np.roll(A, -1, 1) + np.roll(A, 1, 1)

class SPGG:
    def __init__(self, r=4.0, c=1,K=0.1, L=200, iterations=10000, num_of_strategies=2,
                 population_type=0, S_in_one=None):
        np.random.seed() # 生成随机数种子，以便在后续需要生成随机数时，用的是同一个随机数序列
        all_params = dict(locals()) # 创建一个包含当前作用域局部变量和params字典中键值对的新字典
        del all_params['self'] # 去除重复参数
        self.params = all_params # 将参数作为实例的字典
        for key in self.params: # 动态地将字典中的键值对转换为对象的属性和属性值
            setattr(self, key, self.params[key])
        self.cache = {} # 绑定一个字典作为实例的属性
        self._Sn = S_in_one # 不想让外部访问这个实例属性,这个属性保存的是整个网络的策略分布
        self.create_population() # 创建方格网络以及设置人口属性

    def create_population(self):
        L = self.L # 从实例中获取属性L的值
        S_in_one = self._Sn # 从实例中获取属性_Sn的值
        if S_in_one == None: # 没有创建方格网络
            if self.population_type == 0: # 2策略
                S_in_one = np.random.randint(0, 2, size=L * L).reshape([L, L]) # 创建方格网络，设置策略分布
                self._Sn = S_in_one
        self._S = [] # 不想让外部访问这个属性，这个属性保存了2个二维矩阵，每一个二维矩阵都保存了一个策略在方格网络上的分布情况，在自己的二维矩阵上数值为1，代表是自己的策略，数值为0是其他的策略
        for j in range(self.num_of_strategies):
            S = (S_in_one == j) * 1
            self._S.append(S)
        return self._S

    def fun_args_id(self, *args):
        return hash(args)

    def S(self, group_offset=(0, 0), member_offset=(0, 0)):
        key = self.fun_args_id("S", group_offset, member_offset)
        if key in self.cache:
            return self.cache[key]
        result = self._S
        if group_offset != (0, 0):
            result = [np.roll(s, *group_offset) for s in result]
        if member_offset != (0, 0):
            result = [np.roll(s, *member_offset) for s in result]
        self.cache[key] = result
        return result

    def N(self, group_offset=(0, 0)):
        key = self.fun_args_id("N", group_offset)
        if key in self.cache:
            return self.cache[key]
        # N只和在哪个组有关，而和组中位置无关
        S = self.S(group_offset=group_offset)
        result = [overlap5(s) for s in S]
        self.cache[key] = result
        return result

    def P_g_m(self, group_offset=(0, 0), member_offset=(0, 0)):
        key = self.fun_args_id("P_g_m", group_offset, member_offset)
        if key in self.cache:
            return self.cache[key]
        N = self.N(group_offset)
        S = self.S(group_offset, member_offset)
        r, c = self.r, self.c
        n = 5
        N1, N2 = N[0], N[1]
        S1, S2 = S[0], S[1]
        P = (r * c * (N1) / n - c) * S1 + \
            (r * c * (N1) / n) * S2
        self.cache[key] = P
        return P

    def P_AT_g_m(self, group_offset=(0, 0), member_offset=(0, 0)):
        P = self.P_g_m(group_offset, member_offset)
        return P

    def run(self, update=True, it_records=None):
        L, K = self.L, self.K # 取出实例的L属性和K属性
        S = self._S # 取出实例的_S属性，即2个策略各自的二维矩阵，表明各自在网络上的分布情况
        S1, S2 = S[0], S[1] # 分别取出2个策略对应的二维矩阵
        # 进行指定迭代次数的同步更新
        for i in range(1, self.iterations + 1):
            self.cache = {}
            S_in_one = self._Sn # 取出2种策略在方格网络的完整策略分布
            n = 5 # 一个博弈小组的成员数量
            P = self.P_AT_g_m() + self.P_AT_g_m((1, 0), (-1, 0)) + self.P_AT_g_m((-1, 0), (1, 0)) + self.P_AT_g_m(
                (1, 1), (-1, 1)) + self.P_AT_g_m((-1, 1), (1, 1))
            self.P = P
            if it_records != None:
                S = self.S()
                S1, S2 = S[0], S[1]
                # record = (np.sum(S1) / (L * L), np.sum(S2) / (L * L), np.sum(S3) / (L * L), \
                #           P.sum(), \
                #           np.average(P), \
                #           np.average(P[S1 == 1]) if np.sum(S1) > 0 else None, \
                #           np.average(P[S2 == 1]) if np.sum(S2) > 0 else None, \
                #           np.average(P[S3 == 1]) if np.sum(S3) > 0 else None)
                record = (np.sum(S1) / (L * L) ,np.sum(S2) / (L * L))
                it_records.append(record)
            # 只在画热图时开启，因为在画其他图时，多次实验最终记录的数据量可能不同，造成无法求平均值
            # if np.sum(S1 + S3) == 0:
            #     break
            # if i == self.iterations:
            #     break
            if update:
                W_w = 1 / (1 + np.exp((P - np.roll(P, 1, 1)) / K))
                W_e = 1 / (1 + np.exp((P - np.roll(P, -1, 1)) / K))
                W_n = 1 / (1 + np.exp((P - np.roll(P, 1, 0)) / K))
                W_s = 1 / (1 + np.exp((P - np.roll(P, -1, 0)) / K))
                RandomNeighbour = np.random.randint(0, n - 1, size=L * L).reshape([L, L])
                Random01 = np.random.uniform(0, 1, size=L * L).reshape([L, L])
                S_in_one = (RandomNeighbour == 0) * ((Random01 <= W_w) * np.roll(S_in_one, 1, 1) + (Random01 > W_w) * S_in_one) + \
                           (RandomNeighbour == 1) * ((Random01 <= W_e) * np.roll(S_in_one, -1, 1) + (Random01 > W_e) * S_in_one) + \
                           (RandomNeighbour == 2) * ((Random01 <= W_n) * np.roll(S_in_one, 1, 0) + (Random01 > W_n) * S_in_one) + \
                           (RandomNeighbour == 3) * ((Random01 <= W_s) * np.roll(S_in_one, -1, 0) + (Random01 > W_s) * S_in_one)
                self._S = []
                for j in range(self.num_of_strategies):
                    S = (S_in_one == j) * 1
                    self._S.append(S)
                self._Sn = S_in_one

def compute_on_one_process(count):
    # 给实例spgg设置参数、创建方格网络、设置人口属性
    spgg = SPGG(r=4, c=1, K=0.1, L=5, iterations=10000, num_of_strategies=2, population_type=0)
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
        p = Process(target=compute_on_one_process, args=(i + 1,))
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
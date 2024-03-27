import numpy as np


def save_data(path,data):
    np.savetxt(path, data)

def mkdir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    data = np.random.rand(1000)
    mkdir("data")
    path="data/data.txt"
    try:
        #save_data(path,data)
        print("Save successfully")
    except:
        print("Save failed")
    data=np.loadtxt('../data/Origin_Qlearning/C_Qtable/C_Qtable_r=4.7_epoches=10000_L=200_第0次实验数据.txt')
    print(data)
    print(data.shape)
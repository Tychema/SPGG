import numpy as np

def read_data(path):
    data = np.loadtxt(path)
    return data
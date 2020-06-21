import os
import numpy as np

def exists(path):
    return os.path.exists(path)

def join(path_1, path_2):
    return os.path.join(path_1, path_2)

def load_data_txt(path_to_file, num_true, num_false, false_first=True):
    if not exists(path_to_file):
        raise FileNotFoundError

    data = np.loadtxt(path_to_file)
    y = []
    if false_first:
        for i in range(num_false):
            y.append(0)
        for i in range(num_true):
            y.append(1)
    else:
        for i in range(num_true):
            y.append(1)
        for i in range(num_false):
            y.append(0)

    assert len(data)==len(y)
    return data, y

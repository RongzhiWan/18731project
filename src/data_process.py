import os
import numpy as np
import csv

def random_forest_data_gather_save(dir_name, data_save_name):
    X = None
    Y = np.array([])
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith(".csv"):
                continue
            with open(file) as f:
                for row in f.readlines():
                    data = row.split('\t')
                    
                    class_num = int(data[0])
                    Y = np.append(Y, class_num)

                    row_num = int(data[1])
                    col_num = int(data[2])
                    mat_str = data[3].split(' ')
                    mat = np.array(mat_str).reshape((row_num, col_num))
                    mat = mat.astype(np.float)
                    mat_part = mat[:23, :]
                    mat_part = mat_part.reshape(1, mat_part.size)
                    if (X is None):
                        X = mat_part
                    else:
                        X = np.append(X, mat_part, axis=0)
    Y = Y.reshape(Y.size, 1)
    np.savetxt('{}_Y.csv'.format(data_save_name), Y, delimiter=',', fmt='%i')
    np.savetxt('{}_X.csv'.format(data_save_name), X, delimiter=',')

def random_forest_data_gather_load(data_save_name):
    X = np.loadtxt('{}_X.csv'.format(data_save_name), delimiter=',')
    Y = np.loadtxt('{}_Y.csv'.format(data_save_name), delimiter=',').astype(int)
    Y = Y.reshape(Y.size, 1)
    return (X, Y)
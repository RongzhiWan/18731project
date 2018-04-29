import os
import numpy as np
import csv
import glob

def random_forest_data_gather_save(dir_name, data_save_name):
    X = np.array([])
    Y = np.array([])
    take_first_n = 50
    glob_find = '{}/*.txt'.format(dir_name)
    c = 0
    for file in glob.glob(glob_find):
        if file.find("README") != -1:
            continue
        with open(file) as f:
            for row in f.readlines():
                c += 1
                data = row.split('\t')
                
                class_num = int(data[0])
                Y = np.append(Y, class_num)

                row_num = int(data[1])
                col_num = int(data[2])
                flatten_matrix = np.fromstring(data[3], dtype=int, sep=' ')
                mat = np.array(flatten_matrix).reshape((row_num, col_num))
                mat = mat.astype(np.float)
                mat_part = mat[:, :take_first_n]
                if mat_part.shape[1] < take_first_n:
                    mat_part = np.pad(mat_part, ((0, 0), (0, take_first_n - mat_part.shape[1])), 'constant')
                mat_part = mat_part.reshape(1, mat_part.size)
                if (X.size == 0):
                    X = mat_part
                else:
                    X = np.append(X, mat_part, axis=0)
    Y = Y.reshape(Y.size, 1)
    np.savetxt('{}_Y.csv'.format(data_save_name), Y, delimiter=',', fmt='%i')
    np.savetxt('{}_X.csv'.format(data_save_name), X, delimiter=',')

def random_forest_data_gather_load(data_save_name):
    X = np.loadtxt('{}_X.csv'.format(data_save_name), delimiter=',')
    Y = np.loadtxt('{}_Y.csv'.format(data_save_name), delimiter=',').astype(int)
    return (X, Y)


if __name__ == '__main__':
    dir_name = '../data/output'
    rf_output_file = '../data/random_forest/y_out.csv'
    rf_output = np.loadtxt(rf_output_file, delimiter=',')
    glob_find = '{}/*.txt'.format(dir_name)
    c = 0
    for file in glob.glob(glob_find):
        if file.find("README") != -1:
            continue
        filename, _ = os.path.splitext(file)
        with open(file) as f:
            line_count = 0
            for row in f.readlines():
                line_count += 1
        rf_part = rf_output[c:c+line_count, :]
        np.savetxt('{}_y_out.csv'.format(filename), rf_part, delimiter=',')
import numpy as np
import sys
from os import path

device_type_num = 27
feature_num = 23


"""

Input format:

1 2 3 4 5 6 7 8
----------------
0 0 0 1 0 1 0 0
1 0 0 1 0 0 0 0


return:

- f_num: the number of potential device types

- data: device type each data entry belongs to
    e.g.
        4 6
        1 4


"""
def load_classification_res(path):

    first_entry = True
    data = []

    with open(path) as f:
        for line in f:
            entry = line.rstrip()
            cols = entry.split()

            if first_entry:
                # keep down the number of potential device type for classificaiton
                f_num = len(cols)

            list = []  # contain device type this device has been assigned to
            id = 0 # current device id
            for e in cols:#for each classification result
                id += 1
                if e == '0':
                    continue
                list.append(str(id))

            res_entry = " ".join(list)
            print res_entry
            data.append(res_entry)

    return f_num, data


'''
input: dir of training path

return a list of matrix list:

list[0]: matrix list for type 0

'''

def load_training_data(dir_path,device_type_num):

    list = []
    for device_type in range(0,device_type_num):

        file_path = path.join(dir_path,str(device_type))

        matrix_list = load_training_data_one_device_type(file_path)

        list.append(matrix_list)

    return list


'''
input:
0.txt ~ 26.txt

return: type number in str,  a list of matrix for that type

'''
def load_training_data_one_device_type(path):

    res = []

    with open(path) as f:
        for line in f:
            entry = line.rstrip()

            cols = entry.split('\t')

            num_row = int(cols[1])
            num_col = int(cols[2])
            flatten_matrix = cols[3]

            matrix = str2matrix(num_row, num_col, flatten_matrix)
            res.append(matrix)

    return res


"""

num_row: number of row of matrix
num_col: number of column of matrix
flatten_matrix: the flatten matrix, elements split by spaces

e.g.
    str2matrix(2, 3,  "0 1 1 0 0 2")

    gives:
        [[0 1 1]
        [0 0 2]]


    str2matrix(3, 2, "0 1 1 0 0 2")

    gives:
        [[0 1]
         [1 0]
         [0 2]]
"""

def str2matrix(num_row, num_col, flatten_matrix):

    flatten_matrix = np.fromstring(flatten_matrix, dtype=int, sep=' ')

    return np.reshape(flatten_matrix, (num_row, num_col))

"""
Assign type for each testing target

Input:

    classificaiton_result: each line consists of potential classification result for a testing target

    ground_truth_fingerprints: 27 lists of labeled fingerprints

Logic:
    For each testing target's fingerprint F:
        For all potential device types:
            randomly sample 5 fingerprints from that type's ground truth
            global_score = calculate 5 edit distance with F and sum up

        type = one gives the smallest global_score

"""

def assign_type(classificaiton_result, ground_truth_fingerprints, test_fingerprints):

    for entry in test_fingerprints:







def main():


    global device_type_num

    classificaiton_result_path = sys.argv[1]
    training_file_dir = sys.argv[2]
    test_fingerprints_path = sys.argv[3]

    classificaiton_result = load_classification_res(classificaiton_result_path)

    # 27 lists of fingerprints
    ground_truth_fingerprints = load_training_data(training_file_dir,device_type_num)

    test_fingerprints = load_training_data_one_device_type(test_fingerprints_path)








main()
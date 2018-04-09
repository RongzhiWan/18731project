import glob

import numpy as np
import sys
from os import path
import os
import random
import csv
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance

device_type_num = 27
feature_num = 23
sample_num = 5 # number of samples for type edit distance calculation


#debug print
unfinished_data_flag = False
log_flag = True


""""--------------------------------------------------- Loading data ------------------------------------------------"""

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

def csv2list(path):

    l = []
    np.set_printoptions(suppress=True)
    res = np.loadtxt(path, delimiter = ',')


    for e in res:
       # print e
        l.append(e)

    return l



def load_classification_res(path):

    data = []

    res = np.loadtxt(path, delimiter=',')

    unfinished_cnt = 0

    for line in res:

        #entry = line.rstrip()
        #cols = entry.split()

        list = []  # contain device type this device has been assigned to
        id = 0 # current device id

        finished_flag = False
        has_candicate_flag = False

        for e in line:#for each classification result

            id += 1

            if e != 0:
                finished_flag = True

            if e < 0.5: #if e == '0':
                continue
            else:
                has_candicate_flag = True
            list.append(str(id-1))

        if unfinished_data_flag:

            if finished_flag and not has_candicate_flag:
                print line
            if not finished_flag:
                unfinished_cnt +=1

        #res_entry = " ".join(list)
        #print res_entry

        data.append(list)

    if unfinished_data_flag:
        print "unfinished_cnt %d" %(unfinished_cnt)

    return data


'''
Args:
    dir_path(str): dir of training path
    device_type_num(int): number of potential device type

Returns:
    dict: a dict of matrix list:

    e.g.
        dict[0]: matrix list for type 0

'''

def load_training_data(dir_path):

    dict = {}
    true_label_list = []
    #for device_type in range(0,device_type_num):

    glob_find = '{}/*.txt'.format(dir_path)

    for file in glob.glob(glob_find):
        if file.find("README") != -1:
            continue

        device_type, matrix_list = load_training_data_one_device_type(file)

        # generate the list of true labels
        device_num = len(matrix_list)
        true_label_list += device_num * [device_type]

        if device_type not in dict:
            dict[device_type] = matrix_list
        else:
            dict[device_type].extend(matrix_list)


    return dict, true_label_list

'''

def load_training_data(dir_path):

    dict = {}
    #for device_type in range(0,device_type_num):

    for filename in os.listdir(dir_path):

        #file_path = path.join(dir_path,str(device_type)+".txt")

        file_path = path.join(dir_path,filename)

        device_type, matrix_list = load_training_data_one_device_type(file_path)

        if device_type not in dict:
            dict[device_type] = matrix_list
        else:
            dict[device_type].extend(matrix_list)


    return dict
'''



def load_testingFingerprint_data(dir_path):

    res = []
    #for device_type in range(0,device_type_num):

    glob_find = '{}/*.txt'.format(dir_path)
    c = 0
    for file in glob.glob(glob_find):
        if file.find("README") != -1:
            continue


        device_type, matrix_list = load_training_data_one_device_type(file)

        res.extend(matrix_list)

    return res


'''
Args:
    path(str): the path of training file of one device type

    e.g.
            0.txt ~ 26.txt

Returns:
    res(list): a list of fingerprints(matrix) for that type

'''
def load_training_data_one_device_type(path):

    res = []
    first_line_flag = True

    with open(path) as f:
        for line in f:
            entry = line.rstrip()

            cols = entry.split('\t')

            if first_line_flag:

                type = int(cols[0])
                first_line_flag = False

            num_row = int(cols[1])
            num_col = int(cols[2])
            flatten_matrix = cols[3]

            matrix = str2matrix(num_row, num_col, flatten_matrix)
            res.append(matrix)

    return type,res


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


"""-------------------------------------------------- Classification ------------------------------------------------"""

"""
Assign type for each testing target

Args:

    classificaiton_result(list of list of integer): each line consists of potential classification result for a testing target

        e.g.
            list[0] = [4 6]
            list[1] = [1 4]
            means testing target1's potential device type is 4 and 6,
                  testing target2's potential device type is 1 and 4

    ground_truth_fingerprints(list of list of nd_array): 27 lists of labeled fingerprints. Each fingerprint is an nd-array(matrix).


        e.g.
            list[0] = [[0 1 1], [0 0 2]], [[0 1 2], [0 0 2]] ...

            means device 0 has following fingerprints:
                fingerprint1: 0 1 1    fingerprint2:  0 1 2    ...
                              0 0 2                   0 0 2

    test_fingerprints(list of nd_array): fingerprints of the device to classify

        e.g.
            list = [[0 1 1], [0 0 2]], [[0 1 2], [0 0 2]] ...

            means the fingerprint to classify are:
                fingerprint1: 0 1 1    fingerprint2:  0 1 2    ...
                              0 0 2                   0 0 2

Logic:
    For each testing target's fingerprint F:
        For all potential device types:
            randomly sample 5 fingerprints from that type's ground truth
            global_score = calculate 5 edit distance with F and sum up

        type = one gives the smallest global_score

"""


def assign_type(classificaiton_result, ground_truth_fingerprints, test_fingerprints, true_label_list, log_path):

    if len(classificaiton_result) != len(test_fingerprints):
        print "ERROR! testing data number and classification result number not match!"
        return -1

    f_log = open(log_path, 'w')
    testing_device_num = len(classificaiton_result)

    res = [] # store all classification result for testing fingerprints

    # counters for classifcation error calcularion
    correct_cnt = 0.0
    for i in range(0,testing_device_num):
        fingerpirnt_to_classify = test_fingerprints[i]
        potential_type_list = classificaiton_result[i]

        if len(potential_type_list) == 0:#new device
            device_type = -1
        elif len(potential_type_list) == 1:
            device_type = int(potential_type_list[0])
        else:

            min_score = sys.maxint
            device_type = 0
            for type in potential_type_list:

                type = int(type)
                sampled_fingerprint_list = random_sample(sample_num,ground_truth_fingerprints[type])

                score = calculate_global_dissimilarity_score(fingerpirnt_to_classify,sampled_fingerprint_list)

                if score < min_score:
                    min_score = score
                    device_type = type

        correctness = "False"
        true_label =  true_label_list[i]
        no_correct_candidate = ""

        if device_type== -1 or device_type == true_label:
            correct_cnt += 1
            correctness = "True"
        elif str(true_label) not in potential_type_list:
            # no correct candidate
            no_correct_candidate = "alert: no correct candidate!"

        log = "\ttesting data %d: potential_type_list %s\ttrue type %d, classified as type %d\t%s\tcurrent accuracy %f\t%s" %(i,tuple(potential_type_list),true_label,device_type,correctness,correct_cnt/(i+1),no_correct_candidate)
        print log

        f_log.write(log + "\n")
        '''
        print "\ttesting data %d: " %(i),
        print "potential_type_list\t",
        print potential_type_list ,
        print "classified as type %d\t" %(device_type),
        print "current accuracy %f" %(correct_cnt/(i+1))
        '''
        res.append(device_type)

    return res


"""---------------------------------------

 Helper functions for classification

 --------------------------------------"""

"""

Randomly sample n elements from the input.

Args:
    n(int): number of samples to return
    fingerprint_list(list): a list of fingerprints to sample

Returns:
    res(list): a list of n sampled results
"""


def random_sample(n, fingerprint_list):
    res = []

    for i in range(0,n):
        selected = random.choice(fingerprint_list)
        res.append(selected)
    return res

"""

Calculate the dissimilarity score(distance) between fingerprint to classify and a list of sampled fingerprints that
represent the type.

Args:
    test_fingerprint(nd_array): a fingerprint matrix for the device to classify
    sampled_fingerprint_list(list of nd_array): a list of fingerprints sampled from a device type

Returns:
    global_score(float): a global dissimilarity score for testing fingerprint and a potential device type.

"""


def calculate_global_dissimilarity_score(test_fingerprint,sampled_fingerprints):

    scores_list = np.array([])# store scores between each test_fingerprint,sampled_fingerprint pajrs
    test_fingerprint_word,sampled_fingerprint_word_list = fingerprint2word(test_fingerprint, sampled_fingerprints)

    for sampled_fingerprint_word in sampled_fingerprint_word_list:

        distance = damerau_levenshtein_distance(sampled_fingerprint_word, test_fingerprint_word)
        scores_list = np.append(scores_list,distance)

    normalized_scores_list = scores_list / float(max(scores_list))

    global_score = sum(normalized_scores_list)
    return global_score

"""

Convert a set of fingerprint to 'words' that can be used to calculated edit distance.

Args:
    test_fingerprint(nd_array): a fingerprint matrix for the device to classify
    sampled_fingerprint_list(list of nd_array): a list of fingerprints sampled from a device type


Logic:
    1. All columns(packet) in fingerprint_set is mapped to a distinct int id.
    2. Represent the fingerprint with id e.g. [1,2,5,10]

Returns:
    test_fingerprint_word(list of int): id based representation of test fingerprint
    sampled_fingerprint_word_list(list of list of int): a list of id based representation of sampled fingerprint

"""


def fingerprint2word(test_fingerprint,sampled_fingerprint_list):

    # Generate id for each fingerprint
    whole_list = []
    whole_list.append(test_fingerprint)
    whole_list.extend(sampled_fingerprint_list)

    id_map = gen_packet2id_map(whole_list)

    # Map test fingerprint to id
    test_fingerprint_word = fingerprint2id(test_fingerprint,id_map)

    sampled_fingerprint_word_list = []
    for fingerprint in sampled_fingerprint_list:
        sampled_fingerprint_word_list.append(fingerprint2id(fingerprint,id_map))

    return test_fingerprint_word, sampled_fingerprint_word_list


"""

Generate a map that maps fingerprint matrix's column(packet) to distinct id.

    e.g. fingerprint1: 0 1 1    fingerprint2:  0 1 2
                       0 0 2                   0 0 2
                       0 0 0                   0 0 0

        distinct column(packet) are:   [0 0 0], [1 0 0], [1 2 0], [2 2 0]
        id:                             0         1        2         3

        return:
            [0 0 0] -> 0
            [1 0 0] -> 1
            [1 2 0] -> 2
            [2 2 0] -> 3

Args:
    fingerprint_list(set of fingerprint): a list of fingerprint to get packets from
Returns:
    map(dictionary): maps from packet(column) to int id

"""


def gen_packet2id_map(fingerprint_list):
    map = {}  # map from str

    for fingerprint in fingerprint_list:

        (row,col) = fingerprint.shape

        for i in range(0,col):
            packet_ary = fingerprint[:,i]
            packet_str = np.array2string(packet_ary)

            if packet_str not in map:
                id = len(map)
                map[packet_str] = id

    return map

"""
Given a map, maps a fingerprint matrix to a list of ids.

e.g.
         fingerprint1: 0 1 1    fingerprint2:  0 1 2
                       0 0 2                   0 0 2
                       0 0 0                   0 0 0

         map:
            [0 0 0] -> 0
            [1 0 0] -> 1
            [1 2 0] -> 2
            [2 2 0] -> 3

        returns:

        For fingerprint1: [0,1,2]
        For fingerprint2: [0,1,3]
Args:
    fingerprint(nd_array): the fingerprint to map to id
    map(dictionary): map column to id

Returns:
    list: a list of ids that representing the fingerprint

"""


def fingerprint2id(fingerprint,map):

    (row, col) = fingerprint.shape

    list = []
    for i in range(0, col):
        packet_ary = fingerprint[:, i]
        packet_str = np.array2string(packet_ary)
        id = map[packet_str]
        list.append(id)

    return list

"""
Generate one hot vector based on input list.

e.g.    input list : [0, 2, 1]
        length: 3


        output list: [[1,0,0],
                      [0,0,1],
                      [0,1,0]]

Args:
    idx_list(list): a list of int indicating index that need to set to 1
    length(int): the size of each one hot vector. Should be potential device type in this case

Returns:
    res(list): a list of one hot vector
"""


def gen_one_hot_vector(idx_list,length):

    n = len(idx_list)#550

    res = np.zeros((n, length))

    res[np.arange(n), idx_list] = 1

    list = []

    for l in res:
        cur_list = []
        for e in l:
            cur_list.append(str(int(e)))

        list.append(" ".join(cur_list))

    return list


'''
Store list to file. Each element will take one line.

Args:
    list(list): a list of elements that need to be stored to file
    out_path(str): a file path to store the content
'''


def persist(list,out_path):
    f_out = open(out_path,'w')
    for entry in list:
        f_out.write("%s\n"%(entry))

    f_out.close()


def my_print(str):
    sys.stdout.write(str)
    sys.stdout.flush()

"""--------------------------------------------------- main --------------------------------------------------------"""


def main():

    global device_type_num

    classification_result_path = sys.argv[1]
    training_file_dir = sys.argv[2]
    log_path = sys.argv[3]
    output_path = sys.argv[4]

    my_print("Loadint data...")
    # classification output of the trained model
    classification_result = load_classification_res(classification_result_path)
    # 27 lists of fingerprints
    ground_truth_fingerprints, true_label_list = load_training_data(training_file_dir)

    # list of fingerprints to classify
    test_fingerprints = load_testingFingerprint_data(training_file_dir)

    print "Done"

    print("Classifying...")
    # classify input data
    type_list = assign_type(classification_result, ground_truth_fingerprints, test_fingerprints, true_label_list,log_path)

    one_hot_vector_list = gen_one_hot_vector(type_list,device_type_num)
    print "Done"

    my_print("Storing result...")
    persist(one_hot_vector_list, output_path)
    print "Result stored in \"%s\"" %(output_path)

main()

import random
import sys
import argparse
import tensorflow as tf
import numpy as np
from random_forest import RandomForestClassify
from data_process import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Device-type identification from packet features')
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--data_file', dest='data_file', type=str, required=True)
    parser.add_argument('--data_folder', dest='data_folder', type=str, required=True)
    parser.add_argument('--data_gathered', dest='data_gathered', type=int, default=1)
    parser.add_argument('--model_folder', dest='model_folder', type=str)
    parser.add_argument('--test', dest='test', type=int, default=1)
    return parser.parse_args()

def main(args):
    args = parse_arguments()
    if not args.data_gathered:
        random_forest_data_gather_save(args.data_folder, args.data_file)
    (X, Y) = random_forest_data_gather_load(args.data_file)
    num_classes = np.amax(Y) + 1
    num_folds = 1
    num_data = Y.shape[0]
    y_out = np.zeros((num_data, num_classes))
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)
    for i in range(num_folds):
        # split each class of data by num_folds folds
        # test data is one fold
        test_idxs = np.empty((0, ))
        for j in range(num_classes):
            pos_data_idx = np.where(Y == j)
            num_fold_data = int(len(pos_data_idx) / num_folds)
            pos_data_idx_j = pos_data_idx[(i*num_fold_data):((i+1)*num_fold_data)]
            test_idxs = np.append(test_idxs, pos_data_idx_j)
        test_idxs = test_idxs.astype(int)
        x_test = X[test_idxs, :]
        y_test = Y[test_idxs, :]

        # train data is all other data
        train_idxs = np.setdiff1d(np.arange(num_data), test_idxs, assume_unique=True)
        X_train = X[train_idxs, :]
        Y_train = Y[train_idxs, :]

        exit(0)

        # for each class, train with all positive data and num_neg_data negative data points
        num_neg_data = 200
        classifiers = [None] * num_classes
        for j in range(num_classes):
            # create classifier
            classifiers[j] = RandomForestClassify(2, num_features, model_file='{}/class{}'.format(args.model_folder, j))

            pos_data_idx = np.where(Y_train == j)
            neg_data_idx = np.array(random.sample(range(Y_train.size), num_neg_data))
            train_data_idx = np.append(pos_data_idx, neg_data_idx)
            x_train = X_train[train_data_idx, :]
            y_train = Y_train[train_data_idx, :]
            y_train = (y_train == j).reshape(y_train.size, 1)
            train_i = np.reshape(y_train, y_train.shape[0])
            classifiers[j].train(x_train, y_train)

        # test 
        y_out_j = np.zeros((num_data, self.num_classes))
        for i in range(self.num_classes):
            y_out_j[:, j] = self.classifiers[i].test(x_test)[:, 1]
        y_out[test_idxs, :] = y_out_j
    np.savetxt('y_out.csv', y_out, delimiter=',')

if __name__ == '__main__':
    main(sys.argv)
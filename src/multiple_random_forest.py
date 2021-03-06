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
    parser.add_argument('--num_folds', dest='num_folds', type=int, default=10)
    parser.add_argument('--data_file', dest='data_file', type=str, default='../data/random_forest/v2/data')
    parser.add_argument('--data_folder', dest='data_folder', type=str, default='../data/output/v2')
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
    num_folds = args.num_folds
    num_splits = None
    num_data = Y.shape[0]
    num_features = X.shape[1]
    y_out = np.zeros((num_data, num_classes))
    if args.model_folder and (not os.path.exists(args.model_folder)):
        os.makedirs(args.model_folder)

    split_array = []
    for j in range(num_classes):
        data_idx = np.where(Y == j)[0]
        split_array.append(np.array_split(data_idx, num_folds))

    for i in range(num_folds):
        # split each class of data by num_folds folds
        # test data is one fold
        # special case if num_folds is 1, then there is no testing data.
        test_idxs = np.empty((0, ))
        if (num_folds > 1):
            for j in range(num_classes):
                test_idxs = np.append(test_idxs, split_array[j][i])
        # print(test_idxs)
        test_idxs = test_idxs.astype(int)
        x_test = X[test_idxs, :]
        y_test = Y[test_idxs]

        # train data is all other data
        train_idxs = np.setdiff1d(np.arange(num_data), test_idxs)
        X_train = X[train_idxs, :]
        Y_train = Y[train_idxs]

        # for each class, train with data points in the fold
        classifiers = [None] * num_classes
        for j in range(num_classes):
            # create classifier
            print('creating model for fold {} class {}'.format(i, j))
            if (args.model_folder):
                classifiers[j] = RandomForestClassify(2, num_features, model_dir='{}/fold{}_class{}'.format(args.model_folder, i, j), num_splits_to_consider=num_splits)
            else:
                classifiers[j] = RandomForestClassify(2, num_features, num_splits_to_consider=num_splits)

            if (args.train):
                classifiers[j].train(X_train, Y_train==j)
                print('finished training fold {} class {}'.format(i, j))

        # test 
        if (args.test):
            y_out_i = np.zeros((y_test.shape[0], num_classes))
            for j in range(num_classes):
                y_out_j = classifiers[j].predict(x_test)
                y_out_i[:, j] = y_out_j[:, 1]

            y_out_i_h = np.argmax(y_out_i, axis=1)
            accuracy = 1.0 * np.sum(y_out_i_h==y_test) / y_out_i_h.shape[0]
            print('accuracy {}'.format(accuracy))
            
            y_out[test_idxs, :] = y_out_i
            np.savetxt('../data/random_forest/y_out.csv', y_out, delimiter=',')
    
    # print overal test results
    if (args.test):
        y_out_h = np.argmax(y_out, axis=1)
        accuracy = 1.0 * np.sum(y_out_h==Y) / y_out_h.shape[0]
        print('total accuracy {}'.format(accuracy))

if __name__ == '__main__':
    main(sys.argv)
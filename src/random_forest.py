import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensor_forest.python import tensor_forest
import sys


class RandomForestClassify():
    '''
    Wrapper for TensorForest at https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tensor_forest
    Public functions: __init__, train, test
    '''
    def __init__(self, num_classes, num_features, num_trees=50, max_nodes=1000, model_dir=None, num_splits_to_consider=None):
        ''' Initialization function for class RandomForestClassify

        Args:
            num_classes  (int): number of classification classes.
            num_features (int): number of features each input includes
            num_trees    (int): The number of trees to create. 
            max_nodes    (int): The max number of nodes in each tree.
            model_dir    (str): The file to save or load the network. 
                                Train will save the trained network to model_dir
                                Test will load the network from model_dir

        Returns:
            None
        '''
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_trees = num_trees
        self.max_nodes = max_nodes
        self.model_dir = model_dir
        self.num_splits_to_consider = num_splits_to_consider
        tf.reset_default_graph()
        self._init_network()
        #self.saver = tf.train.Saver(max_to_keep=None)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # train
    def train(self, x_train, y_train):
        ''' Public method to train the network

        Args:
            x_train (float numpy array): Training data features
                                         If there are N data points, is a N*(num_features) numpy array
            y_train (int numpy array)  : Training data's classification
                                         If there are N data points, is a N*1 numpy array

        Returns:
            None
        '''
        # _, l = self.sess.run([self.train_op, self.loss_op], feed_dict={self.X: x_train, self.Y: y_train})
        x_train = x_train.astype(np.float32)
        self.classifier.fit(x=x_train, y=y_train)
        # if (self.model_file != None):
        #     self.saver.save(self.sess, self.model_file)
        return

    def test(self, x_test):
        ''' Public method to test the network

        Args:
            x_test (float numpy array): Training data features
                                        If there are M data points, is a M*(num_features) numpy array

        Returns:
            y_out  (float numpy array): The possibility of each data point in each of the classes
                                        If there are M data points, is a M*(num_classes) numpy array
        '''
        # if (self.model_file != None):
        #     self.saver.restore(self.sess, self.model_file)
        # [y_out] = self.sess.run([self.infer_op], feed_dict={self.X: x_test})
        x_test = x_test.astype(np.float32)
        y_out_gen = self.classifier.predict(x=x_test)
        y_out = np.zeros((x_test.shape[0], self.num_classes))
        i = 0
        for y in y_out_gen:
            y_out[i] = y['probabilities']
            i += 1
        return y_out

    def _init_network(self):
        ''' Private function to init the Tensorflow network

        init network with the class attributes set by __init__.
        '''
        if (self.num_splits_to_consider):
            hparams = tensor_forest.ForestHParams(
                num_classes=self.num_classes, 
                num_features=self.num_features, 
                regression=False,
                num_trees=self.num_trees, 
                max_nodes=self.max_nodes,
                num_splits_to_consider=self.num_splits_to_consider).fill()
        else:
            hparams = tensor_forest.ForestHParams(
                num_classes=self.num_classes, 
                num_features=self.num_features, 
                regression=False,
                num_trees=self.num_trees, 
                max_nodes=self.max_nodes).fill()

        # forest_graph = tensor_forest.RandomForestGraphs(hparams)
        # self.X = tf.placeholder(tf.float32, shape=[None, self.num_features])
        # self.Y = tf.placeholder(tf.int32, shape=[None])
        # self.train_op = forest_graph.training_graph(self.X, self.Y)
        # self.loss_op = forest_graph.training_loss(self.X, self.Y)
        # self.infer_op = forest_graph.inference_graph(self.X)
        # self.predict = self.infer_op / tf.reduce_sum(self.infer_op, axis=1, keep_dims=True)
        # correct_prediction = tf.equal(tf.argmax(self.infer_op, 1), tf.cast(self.Y, tf.int64))
        # self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(hparams, model_dir=self.model_dir)
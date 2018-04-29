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
        self.need_update_fast_predict = True
        self.fast_predict = None

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
        x_train = x_train.astype(np.float32)
        self.estimator.fit(x=x_train, y=y_train)
        self._update_predictor()
        return

    def predict(self, x_predict):
        ''' Public method to predict given x

        Args:
            x_test (float numpy array): Training data features
                                        If there are M data points, is a M*(num_features) numpy array

        Returns:
            y_out  (float numpy array): The possibility of each data point in each of the classes
                                        If there are M data points, is a M*(num_classes) numpy array
        '''
        n = x_predict.shape[0]
        x_predict = x_predict.astype(np.float32)
        y_out_gen = self.predictor({'x': x_predict})
        y_out = y_out_gen['probabilities']
        return y_out

    def _update_predictor(self):
        ''' Private function that updates self.predictor

        Should be called whenever needs to predict and the predictor has changed.
        '''
        def serving_input_fn():
            x = tf.placeholder(dtype=tf.float32, shape=[None, self.num_features], name='x')

            features = {'x': x}
            return tf.contrib.learn.utils.input_fn_utils.InputFnOps(
                   features, None, default_inputs=features)
        self.predictor = tf.contrib.predictor.from_contrib_estimator(self.estimator, serving_input_fn)

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


        self.estimator = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(hparams, model_dir=self.model_dir)
        self._update_predictor()
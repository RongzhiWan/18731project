import tensorflow as tf
import sys


class RandomForestClassify():
    '''
    Wrapper for TensorForest at https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tensor_forest
    Public functions: __init__, train, test
    '''
    def __init__(self, num_classes, num_features, num_trees=50, max_nodes=1000, model_file=None):
        ''' Initialization function for class RandomForestClassify

        Args:
            num_classes  (int): number of classification classes.
            num_features (int): number of features each input includes
            num_trees    (int): The number of trees to create. 
            max_nodes    (int): The max number of nodes in each tree.
            model_file   (str): The file to save or load the network. 
                                Train will save the trained network to model_file
                                Test will load the network from model_file

        Returns:
            None
        '''
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_trees = num_trees
        self.max_nodes = max_nodes
        self.model_file = model_file
        tf.reset_default_graph()
        self._init_network()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # train
    def train(x_train, y_train):
        ''' Public method to train the network

        Args:
            x_train (float numpy array): Training data features
                                         If there are N data points, is a N*(num_features) numpy array
            y_train (int numpy array)  : Training data's classification
                                         If there are N data points, is a N*1 numpy array

        Returns:
            None
        '''
        _, l = sess.run([self.train_op, self.loss_op], feed_dict={X: x_train, Y: y_train})
        if (self.model_file != None):
            self.saver.save(self.sess, self.model_file)

    def test(x_test):
        ''' Public method to test the network

        Args:
            x_test (float numpy array): Training data features
                                        If there are M data points, is a M*(num_features) numpy array

        Returns:
            y_out  (float numpy array): The possibility of each data point in each of the classes
                                        If there are M data points, is a M*(num_classes) numpy array
        '''
        if (self.model_file != None):
            self.saver.restore(self.sess, self.model_file)
        y_out = self.sess.run([self.predict], feed_dict={self.X: x_test})
        return y_out

    def _init_network():
        ''' Private function to init the Tensorflow network

        init network with the class attributes set by __init__.
        '''
        hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
            num_classes=self.num_classes, num_features=self.num_features, regression=False,
            num_trees=self.num_trees, max_nodes=self.max_nodes)

        forest_graph = tensor_forest.RandomForestGraphs(hparams)
        self.X = tf.placeholder(tf.float32, shape=[None, self.num_features])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_features])
        self.train_op = forest_graph.training_graph(self.X, self.Y)
        self.loss_op = forest_graph.training_loss(self.X, self.Y)
        infer_op, _, _ = forest_graph.inference_graph(self.X)
        self.predict = infer_op / tf.reduce_sum(infer_op, axis=1, keep_dims=True)
        correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(self.Y, tf.int64))
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
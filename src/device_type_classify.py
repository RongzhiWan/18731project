import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True
from random_forest import RandomForestClassify

class DeviceTypeClassifier():
    def __init__(self, model_folder, num_classes=27, num_features=336):
        self.num_classes = num_classes
        self.num_features = num_features
        self.classifiers = [None] * num_classes
        for j in range(num_classes):
            self.classifiers[j] = RandomForestClassify(2, num_features, model_dir='{}/fold0_class{}'.format(model_folder, j))
    
    def classify(self, x_classify):
        num_data = x_classify.shape[0]
        y_out = np.zeros((num_data, self.num_classes))
        for j in range(self.num_classes):
            y_out_j = self.classifiers[j].predict(x_classify)
            y_out[:, j] = y_out_j[:, 1]
        return y_out

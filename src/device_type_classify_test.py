import numpy as np
from device_type_classify import DeviceTypeClassifier

d = DeviceTypeClassifier(model_folder='../data/random_forest/v2/model')
print('DeviceTypeClassifier initiated')
X_data = np.loadtxt('../data/random_forest/v2/data_X.csv', delimiter=',')
y_out = d.classify(X_data[1:2, :])
print(y_out)
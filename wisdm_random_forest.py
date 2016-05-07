import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Reading the data

print('loading the dataset')

length = 200

f1 = open("data_processing/wisdm_data/basic_features_" + str(length) + ".csv")
f3 = open("data_processing/wisdm_data/basic_features_test_" + str(length) + ".csv")
f2 = open("data_processing/wisdm_data/answers_" + str(length) + ".csv")
f4 = open("data_processing/wisdm_data/answers_test_" + str(length) + ".csv")

data_train = np.loadtxt(fname = f1, delimiter = ',')
labels_train = np.loadtxt(fname = f2, delimiter = ',')
data_test = np.loadtxt(fname = f3, delimiter = ',')
labels_test = np.loadtxt(fname = f4, delimiter = ',')

f1.close(); f2.close(); f3.close(); f4.close()
print(str(length) + ", loading done")

# Classification

rf = RandomForestClassifier(n_estimators=100)
rf.fit(data_train, labels_train)

predictions = rf.predict(data_test)

print('accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, digits = 4))


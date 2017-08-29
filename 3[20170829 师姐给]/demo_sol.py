import numpy
import scipy
from sklearn import datasets
from sol_classifiers import ogd

iris=datasets.load_iris()
X_train=iris.data
Y_train=iris.target


X_test=iris.data[45:46]
Y_test=iris.target[45:46]


clf=ogd(eta=0.1)
clf.fit(X_train,Y_train)


print(clf.predict(X_test))




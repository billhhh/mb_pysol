import numpy
from sklearn import datasets
import mb_pysol

iris=datasets.load_iris()
X_train=numpy.concatenate([iris.data[1:40],iris.data[50:90]])
Y_train=numpy.concatenate([iris.target[1:40],iris.target[50:90]])

X_test=numpy.concatenate([iris.data[40:50],iris.data[90:100]])
Y_test=numpy.concatenate([iris.target[40:50],iris.target[90:100]])

# ogd
print("ogd")
mb_pysol.OGDClassifier();

# Check the versions of libraries

# Python version
import sys
print('Python: {}'.format(sys.version)) # to check the version 

# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))

# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))

# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))

# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

filename = 'C:/Users/Shubham Rawat/Dataset/iris.csv' 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] 
dataset = pd.read_csv(filename, names=names)

#Dimensions of Dataset
print(dataset.shape)

#peek at the data
print(dataset.head(20))

#Statistical Summary
print(dataset.describe())

#Class Distribution
print(dataset.groupby('class').size())



#DATA VISUALIZATION
#Univariate plots to better understand each attribute.
#Multivariate plots to better understand the relationships between attributes.

#Univariate plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histograms
dataset.hist()
plt.show()

#Multivariate Plots
scatter_matrix(dataset)
plt.show()

#Evaluate Some Algorithms
#Create a Validation Dataset
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#Test Harness
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# =============================================================================
# Build Models
#   Logistic Regression (LR)
#   Linear Discriminant Analysis (LDA)
#   K-Nearest Neighbors (KNN).
#   Classification and Regression Trees (CART).
#   Gaussian Naive Bayes (NB).
#   Support Vector Machines (SVM).
# =============================================================================
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show() #Box and Whisker Plot Comparing Machine Learning Algorithms on the Iris Flowers Dataset

#Make Predictions
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

















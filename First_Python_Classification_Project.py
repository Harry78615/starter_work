#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing Libraries that we will need for
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


# Load the Dataset, from URL in this case
url = "https://raw.githubusercontent.com/mikelogikbot/data/main/iris"
#url = "https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset"

names = ['sepal-lenght', 'sepal-width', 'petal-lenght', 'petal-width', 'class']
dataset = read_csv(url, names=names)


# In[ ]:


# Summarize the Dataset.
# Whats its shape?
print(dataset.shape)


# In[ ]:


# Summarize the DataSet.
# The head functions prints the top 20 rows. Rows are called observations in machine learning.
print(dataset.head(20))


# In[ ]:


# Summarize the Dataset
# the describe function provides us with summary statistics on the data.
print(dataset.describe())


# In[ ]:


# Summarize the Dataset.
# Class distribution. Class Distribution is massively important. we are perfect here
print(dataset.groupby('class').size())


# In[ ]:


# Visualize the Data.
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Loading the dataset
url = "https://raw.githubusercontent.com/mikelogikbot/data/main/iris"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = read_csv(url, names=names)

# Box and Whisker Plots
dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex =False, sharey = False)
pyplot.show()

# Histograms
dataset.hist()
pyplot.show()

#Scatter plot matris
scatter_matrix(dataset)
pyplot.show()


# In[ ]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.20, random_state = 1)


# In[ ]:


#Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver= 'liblinear', multi_class = 'ovr' )))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC (gamma='auto')))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s, %f, (%f)' %(name, cv_results.mean(), cv_results.std()))


# In[ ]:


# make predictions on Validated dataset with SVM as its gives high accuracy
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


# In[ ]:


# evaluating predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:


# Predictions with the lowest accuracy model i-e CART
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


# In[ ]:


print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:





# In[ ]:





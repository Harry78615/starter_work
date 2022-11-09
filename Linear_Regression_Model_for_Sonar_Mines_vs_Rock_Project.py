#!/usr/bin/env python
# coding: utf-8

# Binary Classification Problem in Machine Learning Sonar Miner vs Rocks from Machine Learning Repository

# In[2]:


# importing Libraries
import numpy
import pandas as pd
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[16]:


# loading the Dataset from URL

dataset = pd.read_csv(r'sonar.all-data', header = None)
#dataset = read_csv(header = None)


# In[17]:


# Shape pf dataset
print(dataset.shape)


# In[18]:


# types of Dataset
set_option('display.max_rows', 500)
print(dataset.dtypes)


# In[20]:


# head display
set_option('display.width', 100)
print(dataset.head(20))


# In[22]:


# Descriptions, change precison to place 3
set_option('precision', 3)
print(dataset.describe())


# In[23]:


# class distribution
print(dataset.groupby(60).size())


# In[27]:


# Visualize the Dataset
# histogram
dataset.hist(sharex = False, sharey = False, xlabelsize=1, ylabelsize=1)
pyplot.show()


# In[29]:


# Density ploting
dataset.plot(kind = 'density', subplots = True, layout = (8,8), sharex = False, legend = False, fontsize =0.5)
pyplot.show()


# In[32]:


# corelation metrics, multimedia data visualizations
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin = -1, vmax = 1, interpolation = 'none' )
fig.colorbar(cax)
pyplot.show()


# In[33]:


# Split out validation dataset
array = dataset.values
X = array[:,0:60].astype(float)
Y = array[:,60]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size= validation_size, random_state= seed)


# In[34]:


# tests options and evaluation metrics
num_folds = 10
seed = 7
scoring = 'accuracy'


# In[37]:


# Spot-check algorithms (models, alogorithms evokings)
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC))


# In[55]:


# To ignore Future Warnings,
import warnings
warnings.filterwarnings('ignore')

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle= True, random_state= seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)
    


# In[52]:


# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[69]:


# Standardize the Dataset

pipelines = []
pipelines.append(('Scaled-LR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('Scaled-LDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('Scaled-KNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('Scaled-CART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('Scaled-NB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('Scaled-SVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state= None)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)


# In[66]:


# Tune Scaled KNN

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
neighbors = [1,3,5,7,9,13,15,17,21]
param_grid = dict(n_neighbors = neighbors)
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds, shuffle=True,random_state=seed)
grid  = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s " % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params ):
    print("%f (%f) %r" %(mean, stdev, param))


# In[67]:


# tuning SVM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.9, 1.0, 1.3, 1.5, 1.7, 1.9, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel = kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds, shuffle=True,random_state=seed)
grid  = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s " % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params ):
    print("%f (%f) %r" %(mean, stdev, param))


# In[ ]:





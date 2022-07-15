#!/usr/bin/env python
# coding: utf-8

# # Analytics Academy of Data Corner
# 
# 
# 
# ## Machine Learning in Python 
# 
# ##### Decision Trees
# 
# 
# 

# In[ ]:





# Student name: Camila Gomes Vilaça
# 
# Submision date: 12th July 2022

# In[ ]:





# In[ ]:





# In[1]:


# Load libraries

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns


# In[ ]:





# In[2]:


col_names = ['pregnancies', 'glucose', 'bp', 'skinThickness', 'insulin', 'bmi', 'pedigree', 'age', 'label']


# load dataset
#Source: https://www.kaggle.com/datasets/saurabh00007/diabetescsv

df1 = pd.read_csv("diabetes.csv", header=None, names=col_names)


# In[3]:


df1.head()


# In[4]:


df1.info()


# In[5]:


df1.describe()


# ### Allocating the Train and Test datasets

# In[ ]:





# In[6]:


#split dataset in features and target variable


feature_cols = ['pregnancies', 'glucose', 'bp', 'skinThickness', 'insulin', 'bmi', 'pedigree', 'age']

X = df1[feature_cols] # Features (Independent Variables)
Y = df1.label # Target variable (Dependent variables)


# In[7]:


X.head()


# In[8]:


X.shape


# In[9]:


Y.shape


# In[10]:


Y.head()


# In[11]:


# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test


# In[12]:


# 70%
X_train.shape


# In[13]:


# 30%
X_test.shape


# In[14]:


# 70%
y_train.shape


# In[15]:


# 30%
y_test.shape


# In[16]:


# Create Decision Tree classifer object

clf = DecisionTreeClassifier()


# In[17]:


# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)


# In[18]:


X_test.head()


# In[19]:


#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[20]:


print (y_pred)


# In[21]:


print(y_test)


# In[22]:


y_pred.shape


# In[23]:


y_test.shape


# ### Measuring the accuracy of Model

# In[ ]:





# In[24]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[25]:


# Build confusion metrics
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
cm


# In[26]:


# Plot confusion matrix in a beautiful manner
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['Normal', 'Diabetes'], fontsize = 15)
ax.xaxis.tick_top()

ax.set_ylabel('Actual', fontsize=20)
ax.yaxis.set_ticklabels(['Normal', 'Diabetes'], fontsize = 15)
plt.show()


# In[ ]:





# In[ ]:





# ### Visualizaing the Tree

# You can use Scikit-learn's export_graphviz function for display the tree within a Jupyter notebook. For plotting tree, you also need to install graphviz and pydotplus.
# 
# pip install graphviz
# 
# pip install pydotplus

# In[27]:


# --- if needed:

import sys
get_ipython().system('{sys.executable} -m pip install graphviz')
get_ipython().system('{sys.executable} -m pip install pydotplus')


# In[28]:


import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} graphviz')


# In[29]:


get_ipython().system('pip install six')


# In[30]:


from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus


# In[31]:


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


# In[ ]:





# ## Pruning the tree
# 
# criterion="gini", max_depth=4

# In[ ]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", max_depth=4)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


# Build confusion metrics
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
cm


# In[ ]:


# Plot confusion matrix in a beautiful manner
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['Normal', 'Diabetes'], fontsize = 15)
ax.xaxis.tick_top()

ax.set_ylabel('Actual', fontsize=20)
ax.yaxis.set_ticklabels(['Normal', 'Diabetes'], fontsize = 15)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


# # Lab Activity
# 
# Rebuild the model with following callibrations:
# 
# 1. Consider 80% and 90% for training dataset
# 
# 2. Try the follwoing parameters and check the accuracy
# 
#     *criterion="gini" max_depth=5*
#     
#     *criterion="gini" max_depth=3*
#     
#     *criterion="entropy" max_depth=5*
#     
#     *criterion="entropy" max_depth=3*
#     
#     
# After finding the model with the highest accuracy:
# 
#     - Build the final decision tree
#     - Build the confusion Matrxi
#     - Calcuate teh accuracy of your model

# #### With 90% training

# In[ ]:


# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=1) # 90% training and 10% test


# In[ ]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


# Create Decision Tree classifer object
clf2 = DecisionTreeClassifier(criterion="gini", max_depth=3)

# Train Decision Tree Classifer
clf2 = clf2.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf2.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[ ]:


# Create Decision Tree classifer object
clf3 = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf3 = clf3.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf3.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


# Create Decision Tree classifer object
clf4 = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf4 = clf4.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf4.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Best model: 
# 
# clf2/clf4 - Accuracy: 0.7792207792207793

# #### With 80% training

# In[ ]:


# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=1) # 80% training and 20% test


# In[ ]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


# Create Decision Tree classifer object
clf2 = DecisionTreeClassifier(criterion="gini", max_depth=3)

# Train Decision Tree Classifer
clf2 = clf2.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf2.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[ ]:


# Create Decision Tree classifer object
clf3 = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf3 = clf3.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf3.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


# Create Decision Tree classifer object
clf4 = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf4 = clf4.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf4.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Best model: 
# 
# clf4, 80% training - Accuracy: 0.7987012987012987

# In[ ]:


# Build confusion metrics
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
cm


# In[ ]:


# Plot confusion matrix in a beautiful manner
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['Normal', 'Diabetes'], fontsize = 15)
ax.xaxis.tick_top()

ax.set_ylabel('Actual', fontsize=20)
ax.yaxis.set_ticklabels(['Normal', 'Diabetes'], fontsize = 15)
plt.show()


# In[ ]:


from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





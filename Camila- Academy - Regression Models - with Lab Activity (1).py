#!/usr/bin/env python
# coding: utf-8

# # Analytics Academy of Data Corner
# 
# Building and Validating the Regression models in Python

# In[1]:


pwd


# Student name: Camila Gomes Vilaça
# 
# Submission Date: 14th July 2022

# In[2]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

# Provided you are running IPython, the %matplotlib inline will make your plot outputs appear and be stored within the notebook.


# 

# # -------------------------- Linear Regression -------------------------------

# In[3]:


dataset = pd.read_csv('student_scores.csv')


# In[4]:


dataset.describe()


# In[18]:


dataset.shape


# In[6]:


dataset.head(10)


# In[7]:


dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.show()


# In[8]:


# Let´s separate the IVs and DV

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# In[9]:


y


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[12]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[13]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[ ]:





# In[14]:


print(regressor.intercept_)


# In[16]:


y_pred = regressor.predict(X_test)


# In[ ]:





# In[17]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[ ]:





# In[19]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:





# # -------------------------- Logistic Regression -------------------------------

# In[20]:



col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

dataset = pd.read_csv('Diabetes.csv',  header=None, names=col_names)


# In[21]:


dataset.head()


# ## Selecting Feature
# 

# In[22]:


#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = dataset[feature_cols] # Features (IV)
y = dataset.label # Target variable (DV)


# In[ ]:





# ## Splitting Data

# In[33]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:





# ## Model Development and Prediction
# 

# In[ ]:





# In[34]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)


# In[35]:


#
y_pred=logreg.predict(X_test)


# ## Model Evaluation using Confusion Matrix

# In[36]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[ ]:


# --- Visualizing Confusion Matrix using Heatmap


# In[ ]:





# In[ ]:





# ## Lab Activity
# 
# 1. Build the Confusion Matrix using Heatmap
# 2. Calculate the Accuracy of this model

# In[ ]:


# --- Write your code here


# In[27]:


from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix


# In[37]:


# Plot confusion matrix in a beautiful manner
ax= plt.subplot()
sns.heatmap(cnf_matrix, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['Normal', 'Diabetes'], fontsize = 15)
ax.xaxis.tick_top()

ax.set_ylabel('Actual', fontsize=20)
ax.yaxis.set_ticklabels(['Normal', 'Diabetes'], fontsize = 15)
plt.show()


# In[38]:


# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[30]:


# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[40]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[ ]:





# # -------------------------- Multiple Regression -------------------------------

# In[ ]:





# ## Loading the dataset
# 

# In[41]:


raw_data = pd.read_csv('Housing_Data.csv')


# ## Some EDA

# In[42]:


raw_data.shape


# In[43]:


raw_data.head(20)


# In[44]:


raw_data.describe()


# In[45]:


sns.pairplot(raw_data)


# In[46]:


# ---- Testing correlation

sns.heatmap(raw_data.corr(),annot=True,lw=1)


# In[47]:


# Let´s separate the IVs and DV

x = raw_data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population']]

y = raw_data['Price']


# In[ ]:





# ## Building the Train and Test datasets

# In[48]:





from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[ ]:





# ## Buidling the Regression Model

# In[49]:



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)


# In[50]:



print(model.coef_)


# In[51]:



print(model.intercept_)


# In[52]:


pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])


# ## Making the prediction

# In[53]:



predictions = model.predict(x_test)


# In[54]:


dfCompare = pd.DataFrame({'Actual': y_test, 'Predicted':predictions})
dfCompare


# ## Evaluating the model

# In[55]:






 plt.scatter(y_test, predictions)


# In[56]:




plt.hist(y_test - predictions)


# In[ ]:





# ## Interpreting the Regression Accuracy Measures
# 
# 
# Example:
# 
# MAE=10 implies that, on average, the forecast's distance from the true value is 10 (e.g true value is 200 and forecast is 190 or true value is 200 and forecast is 210 would be a distance of 10). 
# 
# 

# In[57]:


from sklearn import metrics


# In[58]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:





# ## Now, let´s apply our model to the new data

# In[ ]:





# In[59]:


x_test.head(10)


# In[60]:


# iterating the columns
for col in x_test.columns:
    print(col)


# In[61]:


# Creating a new fresh dataset

dfNew = pd.DataFrame(columns=['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population'])


# In[62]:


dfNew.shape


# In[63]:


dfNew = dfNew.append({'Avg. Area Income': 45000,'Avg. Area House Age':20,'Avg. Area Number of Rooms':3,'Avg. Area Number of Bedrooms':2, 'Area Population':100000}, ignore_index=True)
dfNew = dfNew.append({'Avg. Area Income': 54600,'Avg. Area House Age':5,'Avg. Area Number of Rooms':3,'Avg. Area Number of Bedrooms':2, 'Area Population':150000}, ignore_index=True)
dfNew = dfNew.append({'Avg. Area Income': 159000,'Avg. Area House Age':8,'Avg. Area Number of Rooms':5,'Avg. Area Number of Bedrooms':4, 'Area Population':23000}, ignore_index=True)
dfNew = dfNew.append({'Avg. Area Income': 25000,'Avg. Area House Age':40,'Avg. Area Number of Rooms':3,'Avg. Area Number of Bedrooms':2, 'Area Population':240000}, ignore_index=True)
dfNew = dfNew.append({'Avg. Area Income': 48000,'Avg. Area House Age':12,'Avg. Area Number of Rooms':2,'Avg. Area Number of Bedrooms':1, 'Area Population':30000}, ignore_index=True)


# In[64]:


dfNew.head (10)


# ## Applying the model to the fresh data to predict the DV

# In[65]:


predNew = model.predict(dfNew)


# In[ ]:





# In[66]:




dfNewWithPred = pd.concat([dfNew, pd.DataFrame(predNew)], axis=1)


# In[67]:


dfNewWithPred.columns = [*dfNewWithPred.columns[:-1], 'Predicted Price']


# In[68]:


dfNewWithPred.head(10)


# In[ ]:





# # <<<  Implementing Multiple Regression with Categorical variable >>>

# In some situations we may have some categorical variables as well, so we need to add dummy variables, you might be wondering what is dummy variable?

# ## What is a Dummy Variable?
# 
# 
# A dummy variable (is, an indicator variable) is a numeric variable that represents categorical data, such as gender, race, etc.

# In[ ]:





# Let´s load a new dataset with categorical varibales!

# In[69]:


raw_data2 = pd.read_csv('Housing_Data_withCategorical.csv')


# In[70]:


raw_data2.shape


# In[71]:


raw_data2.head (5)


# In[72]:


sns.pairplot(raw_data2)


# In[73]:


# ---- Testing correlation

sns.heatmap(raw_data2.corr(),annot=True,lw=1)


# In[ ]:





# In[76]:


raw_data2.head(30)


# In[ ]:





# In[75]:


raw_data2 = pd.get_dummies(raw_data2, columns=["Area Type", "Public Transporation Quality"], drop_first=True)


# In[77]:


raw_data2.head(5)


# In[78]:


raw_data2.dtypes


# In[79]:


# Let´s separate the IVs and DV

x = raw_data2[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population', 'Area Type_Urban','Public Transporation Quality_Bad','Public Transporation Quality_Excellent', 'Public Transporation Quality_Good']]

y = raw_data2['Price']


# In[ ]:





# ## Lab Activity
# 
# Complete the model as follows:
# 
# 1. Sepearte the Train and Test datasets
# 2. Build the model
# 3. Show the coefficients
# 4. Compatre the predicted and actual values in graph
# 5. Calculate the accuracy of the model
# 6. Build a fresh dataset
# 7. Apply the model to predict the DV with the fresh dataset

# ### 1. Separate the Train and Test datasets

# In[85]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# ### 2. Build the model

# In[86]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)


# ### 3. Show the coefficients

# In[87]:


print(model.coef_)


# In[88]:


print(model.intercept_)


# In[89]:


pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])


# ### 4. Compatre the predicted and actual values in graph

# In[90]:


predictions = model.predict(x_test)


# In[91]:


dfCompare = pd.DataFrame({'Actual': y_test, 'Predicted':predictions})
dfCompare


# In[92]:


plt.scatter(y_test, predictions)


# In[93]:


plt.hist(y_test - predictions)


# ### 5. Calculate the accuracy of the model
# 
# 
# Example:
# 
# MAE=10 implies that, on average, the forecast's distance from the true value is 10 (e.g true value is 200 and forecast is 190 or true value is 200 and forecast is 210 would be a distance of 10). 
# 
# 

# In[94]:


from sklearn import metrics


# In[95]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ### 6. Build a fresh dataset

# In[96]:


x_test.head(10)


# In[97]:


# iterating the columns
for col in x_test.columns:
    print(col)


# In[98]:


# Creating a new fresh dataset
dfNew = pd.DataFrame(columns = [
    'Avg. Area Income',
    'Avg. Area House Age',
    'Avg. Area Number of Rooms',
    'Avg. Area Number of Bedrooms', 
    'Area Population',
    'Area Type_Urban',
    'Public Transporation Quality_Bad',
    'Public Transporation Quality_Excellent',
    'Public Transporation Quality_Good'
    ]
)


# In[99]:


dfNew.shape


# In[100]:


dfNew = dfNew.append({'Avg. Area Income': 45000,'Avg. Area House Age':20,'Avg. Area Number of Rooms':3,'Avg. Area Number of Bedrooms':2, 'Area Population':100000, 'Area Type_Urban': 0,'Public Transporation Quality_Bad': 0, 'Public Transporation Quality_Excellent': 0, 'Public Transporation Quality_Good': 1}, ignore_index=True)
dfNew = dfNew.append({'Avg. Area Income': 54600,'Avg. Area House Age':5,'Avg. Area Number of Rooms':3,'Avg. Area Number of Bedrooms':2, 'Area Population':150000, 'Area Type_Urban': 0,'Public Transporation Quality_Bad': 1, 'Public Transporation Quality_Excellent': 0, 'Public Transporation Quality_Good': 0}, ignore_index=True)
dfNew = dfNew.append({'Avg. Area Income': 159000,'Avg. Area House Age':8,'Avg. Area Number of Rooms':5,'Avg. Area Number of Bedrooms':4, 'Area Population':23000, 'Area Type_Urban': 1,'Public Transporation Quality_Bad': 1, 'Public Transporation Quality_Excellent': 0, 'Public Transporation Quality_Good': 0}, ignore_index=True)
dfNew = dfNew.append({'Avg. Area Income': 25000,'Avg. Area House Age':40,'Avg. Area Number of Rooms':3,'Avg. Area Number of Bedrooms':2, 'Area Population':240000, 'Area Type_Urban': 0,'Public Transporation Quality_Bad': 0, 'Public Transporation Quality_Excellent': 1, 'Public Transporation Quality_Good': 0}, ignore_index=True)
dfNew = dfNew.append({'Avg. Area Income': 48000,'Avg. Area House Age':12,'Avg. Area Number of Rooms':2,'Avg. Area Number of Bedrooms':1, 'Area Population':30000, 'Area Type_Urban': 1,'Public Transporation Quality_Bad': 0, 'Public Transporation Quality_Excellent': 1, 'Public Transporation Quality_Good': 0}, ignore_index=True)


# In[101]:


dfNew.head (10)


# ### 7. Apply the model to predict the DV with the fresh dataset

# In[102]:


predNew = model.predict(dfNew)


# In[103]:


dfNewWithPred = pd.concat([dfNew, pd.DataFrame(predNew)], axis=1)


# In[104]:


dfNewWithPred.columns = [*dfNewWithPred.columns[:-1], 'Predicted Price']


# In[105]:


dfNewWithPred.head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





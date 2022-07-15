#!/usr/bin/env python
# coding: utf-8

# # Analytics Academy of Data Corner
# 
# #### Introduction to Python Programming

# In[ ]:





# ## Lab Activity 1
# 
# Add an introduction using Markdown with following info:
# 
# * Your full name
# * Today Date
# * Your course group
# 
# 
# Write a Line of python code to print “Hello World” 

# In[1]:


#write your code here


# * ####Camila Gomes Vilaça
# 
# * #5th July 2022
# * ##Data science group 

# In[2]:


print ("Hello World ")


# In[3]:


# how to find your jupyter path


# In[4]:


pwd


# # varibales

# In[5]:


numOfBoxes=12
OwnerName="Sofia"
print("numOfBoxes=",numOfBoxes)
print("OwnerName=",OwnerName)


# # Lists

# In[6]:


#To define an integer, use the following syntax:
myint=7
print(myint)

#To define a floating point number, you may use one of the following notations
myfloat=8.0
print(myfloat)
myfloat=float(8)
print(myfloat)


# In[7]:


#Strings are defined either with a single quote or a double quote
mystring='hello'
print (mystring)
mystring="hello"
print(mystring)


# ## Lab Activity 2
# 
# In this exercise, you will need to add numbers and strings to the correct lists using the "append" list method. You must add the numbers 1,2, and 3 to the "numbers" list, and the words 'hello' and 'world' to the string variable.
# 
# You will also have to fill in the variable second_name with the second name in the names list, using the brackets operator []. Note that the index is zero-based, so if you want to access the second item in the list, its index will be 1.
# 
# 
# 

# In[8]:


numbers =[]
strings= []
names = ["John","Eric","Jessica"]


# In[9]:


#write your code here

a=6
b=7 

print("a", a , "b,b ")


# In[10]:


second_name= None  ## write your code here


# In[11]:


#this code should  write out the filled arrays and the second name in the names list (Eric)

print (numbers)
print (strings)
print ("The second name on the names list is %s" % second_name)


# # Basic Operators

# In[ ]:


# Lists- create a list 

mylist = []
mylist.append(1)
mylist.append(2)
mylist.append(3)
print(mylist[0]) # prints 1
print(mylist[1]) # prints 2
print(mylist[2]) # prints 3

 

print ("===: se====")

# prints out 1,2,3

for x in mylist:
print(x)


# In[ ]:


numbers = []
strings = []
names = ["John", "Eric", "Jessica"]

# write your code here
second_name None


# this code should write out the filled arrays and the second name in the names List (Eric).
print (numbers)

print (strings)

print("The second name on the names list is %s" % second_name)


# In[ ]:





# In[ ]:





# # Conditions

# In[ ]:





# In[ ]:





# In[ ]:





# ## Lab Activity 3
# 
# In the following code, change the variables in the first section, so that each if statement resolves as True.
# You should generarte the follwoing output:
# 
# 1
# 
# 2
# 
# 3
# 
# 4
# 

# In[12]:


number =12
second_number=10
first_array =
second_array= [1,2,3]


# In[ ]:


if number > 15:
    print ("1")


# In[ ]:


if first_array:
    print ("2")


# In[ ]:


if len (second_array)==2 :
    print ("3")


# In[ ]:


if len (first_array) + len (second_array)==5: 
    print ("4")


# In[ ]:





# # Loops

# In[ ]:





# In[ ]:





# In[ ]:





# # Lab Activity 4
# 
# Write a loop to calculate the factorial for the value stored in varibale inputVar

# In[ ]:


inputVar =5


# In[ ]:


#--- write your code here


# In[ ]:





# In[ ]:





# # Functions

# In[ ]:





# In[ ]:





# In[ ]:





# # Lab Activity 5
# 
# Write a function ( Facto) that 
# 
# * receive a N as parameter
# * Calculate and return the N!
# 
# Then, call your function with:
# 
# * Facto (5)
# * Facto (10)

# In[ ]:





# In[ ]:





# In[ ]:





# 
# 
# 
# # Data Cleaning and working with data frames in Python
# 
# 
# 
# 

# In[ ]:



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as ms
import numpy as np


# In[ ]:





# In[ ]:


#---- if you need to install any python packagaes

#import sys
#!{sys.executable} -m pip install numpy


# ## Loading data into a dataframe

# In[ ]:


dfMain = pd.read_csv("Loans_DataSet.csv")


# In[13]:


dfMain.shape


# In[ ]:


dfMain.head(10)


# In[ ]:


dfMain["Employment.Type"].head()


# In[ ]:


dfMain["asset_cost"].describe()


# In[ ]:


dfMain.dtypes


# # Filtering based on Conditions:
# 
# 
# Datasets can be filtered using different conditions, which can be implemented using logical operators in python. For example, == (double equal to), ≤ (less than or equal to), ≥(greater than or equal to), etc.

# In[ ]:


dfMain[(dfMain['Employment.Type'] == "Salaried")]


# In[ ]:


dfMain[(dfMain['Employment.Type'] == "Salaried") & (dfMain['branch_id'] == 100)]


# # Lab Activity 6
# 
# Filter the dfMain based on teh follwoing conditions:
#     
#   asset_cost > 8000 and loan_default=1
# 
# How many records exists with this condition?

# In[ ]:


#---- write your code here


# In[ ]:





# # Data Cleaning in Python

# In[ ]:





# In[ ]:


### #1: Handling null values:


# In[ ]:





# In[ ]:


dfMain.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


#Showing records with null values


# In[ ]:


# Method 1

dfMain[dfMain['Employment.Type'].isnull()]


# In[ ]:


# Method 2
null_series = pd.isnull(dfMain['Employment.Type'])  

dfMain[null_series] 


# In[ ]:





# ## Handling Null values

# In[ ]:


dataCopy = dfMain


# In[ ]:


print(dataCopy.shape)


# ### Method #1: Removing the records with null values

# In[ ]:





# In[ ]:


dataClean = dataCopy.dropna()


# In[ ]:


print(dataClean.shape)


# In[ ]:


dataClean[dataClean['Employment.Type'].isnull()]


# In[ ]:





# ### Method #2: replacing the Null values with another value

# In[ ]:





# In[ ]:



dataCopy['Employment.Type'] = dataCopy['Employment.Type'].replace(np.nan, "-----")


# In[ ]:


dataCopy[dataCopy['Employment.Type'].isnull()]


# In[ ]:


print(dataCopy.shape)


# In[ ]:





# ###  Method #3: replacing the Null values with previous valid value

# In[ ]:





# In[ ]:


dataCopy = dfMain
dataCopy = dataCopy.fillna(method='ffill') # forward fill


# In[ ]:





# In[ ]:


dataCopy[dataCopy['Employment.Type'].isnull()]


# In[ ]:


print (dataCopy.shape)


# In[ ]:





# # Handling duplicate records

# In[ ]:


dataCopy = dfMain


# In[ ]:


duplicate = dataCopy[dataCopy.duplicated()] 
  
print("Duplicate Rows :") 

duplicate


# In[ ]:


duplicate = dataCopy[dataCopy.duplicated(['branch_id', 'supplier_id'])] 
  
print("Duplicate Rows based on branch_id and supplier_id :") 
  
# Print the resultant Dataframe 
duplicate 


# In[ ]:


# dropping duplicate values 
duplicate.drop_duplicates(['branch_id', 'supplier_id'],keep= 'last')


# In[ ]:





# # Handling outliers

# Quantile-based Flooring and Capping
# 
# In this technique, we will do the flooring (e.g., the 10th percentile) for the lower values and capping (e.g., the 90th percentile) for the higher values.

# In[ ]:


print(dfMain['disbursed_amount'].quantile(0.10))
print(dfMain['disbursed_amount'].quantile(0.90))


# In[ ]:


dfTemp = dfMain


# In[ ]:


dfTemp.drop(dfTemp[dfTemp['disbursed_amount'] < 39794].index, inplace = True)
dfTemp.drop(dfTemp[dfTemp['disbursed_amount'] > 68882].index, inplace = True) 


# In[ ]:


print(dfTemp.shape)


# In[ ]:





# In[ ]:





# # Preparing the Training & Test datasets in DS projects

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


training_data, testing_data = train_test_split(dfMain, test_size=0.2, random_state=25)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


# In[ ]:


training_data.shape


# In[ ]:


testing_data.shape


# In[ ]:





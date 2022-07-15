#!/usr/bin/env python
# coding: utf-8

# # Analytics Academy of Data Corner
# 
# #### Introduction to Python Programming

# In[1]:


pwd


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

# In[2]:


#write your code here


# # Camila Gomes Vilaça
# ## 5th July 2022
# ### Data Science Course (Group A)

# In[3]:


# Camila Gomes Vilaça
## 5th July 2022
### Data Science Course (Group A)


# In[4]:


print("Hello World")


# In[ ]:





# In[5]:


# how to find your jupyter path


# In[6]:


pwd


# # varibales

# In[7]:


numOfBoxes = 12
ownerName = "Sofia"
print("numOfBoxes = ", numOfBoxes)
print("ownerName = ", ownerName)


# In[8]:


myint = 7
print(myint)

myfloat = 8.0
print(myfloat)
myfloat = float(8)
print(myfloat)


# In[9]:


mystring = 'hello'
print(mystring)
mystring = "hello"
print(mystring)


# In[10]:


a = 6
b = 7

print("a:", a, " b:", b)


# In[11]:


a, b = 9, 10

print("a:", a, " b:", b)


# In[ ]:





# # Lists

# In[12]:


#Lists

mylist = []
mylist.append(1)
mylist.append(2)
mylist.append(3)
print(mylist[0]) # prints 1
print(mylist[1]) # prints 2
print(mylist[2]) # prints 3

print("===========")

# prints out 1,2,3
for x in mylist:
    print(x)


# ## Lab Activity 2
# 
# In this exercise, you will need to add numbers and strings to the correct lists using the "append" list method. You must add the numbers 1,2, and 3 to the "numbers" list, and the words 'hello' and 'world' to the string variable.
# 
# You will also have to fill in the variable second_name with the second name in the names list, using the brackets operator []. Note that the index is zero-based, so if you want to access the second item in the list, its index will be 1.
# 
# 
# 

# In[13]:


numbers =[]
strings= []
names = ["John","Eric","Jessica"]


# In[14]:


#write your code here
for i in range(1, 4):
    numbers.append(i)

strings.append("hello")
strings.append("world")


# In[15]:


second_name= None  ## write your code here

second_name = names[1]


# In[16]:


#this code should  write out the filled arrays and the second name in the names list (Eric)

print (numbers)
print (strings)
print ("The second name on the names list is %s" % second_name)


# In[ ]:





# # Basic Operators

# In[17]:


number = 1 + 2 * 3 / 4.0
print(number)


# In[18]:


remainder = 11 % 3
print(remainder)


# In[19]:


squared = 7 ** 2
cubed = 2 **3
print(squared)
print(cubed)


# In[ ]:





# In[20]:


lotsofhellos = 'hello ' * 10
print(lotsofhellos)


# In[21]:


even_numbers = [2, 4, 6, 8]
odd_numbers = [1, 3, 5, 7]
all_numbers = odd_numbers + even_numbers 
print(all_numbers)


# In[22]:


print([1, 2, 3] * 3)


# # Conditions

# In[23]:


x = 2
print(x == 2) # prints out True
print(x == 3) # prints out False
print(x < 3) # prints out True


# In[24]:


name = "John"
age = 23
if name == "John" and age == 23:
    print("Your name is John, and you are also 23 years old")

if name == "John" or name == "Rick":
    print("Your name is either John or Rick")


# In[25]:


name = "John"
if name in ["John", "Rick"]:
    print("Your name is either John or Rick")


# In[26]:


statement = False
another_statement = True

if statement is True:
    # do something
    
    print("Statement is True!")
    pass
elif another_statement is True:
    print("Another Statement is True!")
    pass
else:
    pass


# In[27]:


print(not False)
print()


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

# In[28]:


number = 16
second_number = 2
first_array = [3, 4, 5]
second_array= [1, 2]


# In[29]:


if number > 15:
    print ("1")


# In[30]:


if first_array:
    print ("2")


# In[31]:


if len (second_array) == 2 :
    print ("3")


# In[32]:


if len (first_array) + len (second_array) == 5: 
    print ("4")


# In[ ]:





# # Loops

# In[33]:


primes = [2, 3, 5, 7]
for prime in primes:
    print(prime)


# In[34]:


count = 0
while count < 5:
    print(count)
    count += 1


# In[35]:


count = 0
while True:
    print(count)
    count += 1
    if count >= 5:
        break
        
for x in range(10):
    if x % 2 == 0:
        continue
    print(x)


# # Lab Activity 4
# 
# Write a loop to calculate the factorial for the value stored in varibale inputVar

# In[36]:


inputVar = 5


# In[37]:


#--- write your code here


# In[38]:


outputVar = 1
for i in range(1, inputVar + 1):
    outputVar = outputVar * i
print(outputVar)


# In[ ]:





# # Functions

# In[39]:


def my_function():
    print("Hello From My Function!")
    
def my_function_with_args(username, greeting):
    print("Hello, %s, From My Function!, I wish you %s" %(username, greeting))

def sum_two_numbers(a, b):
    return a + b

my_function()

my_function_with_args("John Doe", "a great year!")

x = sum_two_numbers(1, 2)
print(x)


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

# In[40]:


def Facto(N):
    outputVar = 1
    for i in range(1, N + 1):
        outputVar = outputVar * i
    return outputVar


# In[41]:


print(Facto(5))
print(Facto(10))


# In[ ]:





# 
# 
# 
# # Data Cleaning and working with data frames in Python
# 
# 
# 
# 

# In[42]:


get_ipython().system('pip install missingno')


# In[43]:



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import missingno as ms


# In[ ]:





# In[44]:


#---- if you need to install any python packagaes

#import sys
#!{sys.executable} -m pip install numpy


# ## Loading data into a dataframe

# In[45]:


dfMain = pd.read_csv("Loans_DataSet.csv")


# In[46]:


dfMain.shape


# In[47]:


dfMain.head(10)


# In[48]:


dfMain["Employment.Type"].head()


# In[49]:


dfMain["asset_cost"].describe()


# In[50]:


dfMain.dtypes


# # Filtering based on Conditions:
# 
# 
# Datasets can be filtered using different conditions, which can be implemented using logical operators in python. For example, == (double equal to), ≤ (less than or equal to), ≥(greater than or equal to), etc.

# In[51]:


dfMain[(dfMain['Employment.Type'] == "Salaried")]


# In[52]:


dfMain[(dfMain['Employment.Type'] == "Salaried") & (dfMain['branch_id'] == 100)]


# # Lab Activity 6
# 
# Filter the dfMain based on teh follwoing conditions:
#     
#   asset_cost > 8000 and loan_default=1
# 
# How many records exists with this condition?

# In[53]:


#---- write your code here


# In[54]:


dfMain[(dfMain['asset_cost'] > 8000) & (dfMain['loan_default'] == 1)]


# # Data Cleaning in Python

# In[ ]:





# In[55]:


### #1: Handling null values:


# In[ ]:





# In[56]:


dfMain.apply(lambda x: sum(x.isnull()),axis=0)


# In[57]:


#Showing records with null values


# In[58]:


# Method 1

dfMain[dfMain['Employment.Type'].isnull()]


# In[59]:


# Method 2
null_series = pd.isnull(dfMain['Employment.Type'])  

dfMain[null_series] 


# In[ ]:





# ## Handling Null values

# In[60]:


dataCopy = dfMain


# In[61]:


print(dataCopy.shape)


# ### Method #1: Removing the records with null values

# In[ ]:





# In[62]:


dataClean = dataCopy.dropna()


# In[63]:


print(dataClean.shape)


# In[64]:


dataClean[dataClean['Employment.Type'].isnull()]


# In[ ]:





# ### Method #2: replacing the Null values with another value

# In[ ]:





# In[65]:



dataCopy['Employment.Type'] = dataCopy['Employment.Type'].replace(np.nan, "-----")


# In[66]:


dataCopy[dataCopy['Employment.Type'].isnull()]


# In[67]:


print(dataCopy.shape)


# In[ ]:





# ###  Method #3: replacing the Null values with previous valid value

# In[ ]:





# In[68]:


dataCopy = dfMain
dataCopy = dataCopy.fillna(method='ffill') # forward fill


# In[ ]:





# In[69]:


dataCopy[dataCopy['Employment.Type'].isnull()]


# In[70]:


print (dataCopy.shape)


# In[ ]:





# # Handling duplicate records

# In[71]:


dataCopy = dfMain


# In[72]:


duplicate = dataCopy[dataCopy.duplicated()] 
  
print("Duplicate Rows :") 

duplicate


# In[73]:


duplicate = dataCopy[dataCopy.duplicated(['branch_id', 'supplier_id'])] 
  
print("Duplicate Rows based on branch_id and supplier_id :") 
  
# Print the resultant Dataframe 
duplicate 


# In[74]:


# dropping duplicate values 
duplicate.drop_duplicates(['branch_id', 'supplier_id'], keep = 'last')


# In[ ]:





# # Handling outliers

# Quantile-based Flooring and Capping
# 
# In this technique, we will do the flooring (e.g., the 10th percentile) for the lower values and capping (e.g., the 90th percentile) for the higher values.

# In[75]:


print(dfMain['disbursed_amount'].quantile(0.10))
print(dfMain['disbursed_amount'].quantile(0.90))


# In[76]:


dfTemp = dfMain


# In[77]:


dfTemp.drop(dfTemp[dfTemp['disbursed_amount'] < 39794].index, inplace = True)
dfTemp.drop(dfTemp[dfTemp['disbursed_amount'] > 68882].index, inplace = True) 


# In[78]:


print(dfTemp.shape)


# In[ ]:





# In[ ]:





# # Preparing the Training & Test datasets in DS projects

# In[79]:


from sklearn.model_selection import train_test_split


# In[80]:


training_data, testing_data = train_test_split(dfMain, test_size=0.2, random_state=25)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


# In[81]:


training_data.shape


# In[82]:


testing_data.shape


# In[ ]:





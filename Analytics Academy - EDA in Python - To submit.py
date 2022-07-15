#!/usr/bin/env python
# coding: utf-8

# # Analytics Academy of Data Corner
# 
# 
# Exploratory Data Analysis (EDA) in Python

# In[ ]:





# In[3]:


#Importing the required libraries


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:





# ## *Step 1. Loading the dataset*

# In[6]:


dfRawData = pd.read_excel("Singapore Airbnb - Raw.xlsx")


# In[7]:


dfRawData.shape


# In[8]:


#--- List of columns

dfRawData.dtypes


# In[9]:


dfRawData.head(5)


# In[ ]:





# ## *Step 2. Data Cleaning*

# In[ ]:


#-- Copying a backup from rawdata

dfToClean = dfRawData

dfRawData.shape


# ### 1: Handling null values:

# In[ ]:


# Listing the columns with null values

dfToClean.apply(lambda x: sum(x.isnull()),axis=0)


# ###  Showing records with null values

# In[ ]:





# Let´s starts with name and id

dfToClean[dfToClean['name'].isnull()]



# In[ ]:


dfToClean.shape


# In[ ]:


dfToClean[dfToClean['host_id'].isnull()]


# In[ ]:


#-- we decide to remove these rows because they are just a few records and the columns are improtant for us to have value


# In[ ]:


dfToClean = dfToClean.dropna(subset=['name', 'host_id'])


# In[ ]:


dfToClean.shape


# In[ ]:


# lets get sure if we any null value remained in name and host_id

dfToClean.apply(lambda x: sum(x.isnull()),axis=0)


# ## Lab Activity:
# 
# Now, Let´s solve the nulls in latitude  and longitude
# 
# 1. Show the rows with null values 
# 2. put '' in the null cells for these two cells
# 3. check the null values again for the whole dataframe

# In[ ]:


# write your code here


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# How about the number_of_reviews?

dfToClean[dfToClean['number_of_reviews'].isnull()]


# In[ ]:


# let´s set them with the average number_of_reviews

varMean = dfToClean['number_of_reviews'].mean()


# In[ ]:


dfToClean['number_of_reviews'] = dfToClean['number_of_reviews'].replace(np.nan, round(varMean))


# In[ ]:


# lets check the null values again

dfToClean.apply(lambda x: sum(x.isnull()),axis=0)


# ## Lab Activity:
# 
# Solve the nulls issue for last_review and reviews_per_month
# 
# 1. Show the rows with null values 
# 2. put 0 for reviews_per_month
# 3. put '' for last_review
# 3. check the null values again for the whole dataframe

# In[ ]:





# In[ ]:





# In[ ]:





# #### ===== Congrats! We handled the Null values! =====

# ## Now, it´s turn to handle the dublicates!

# In[ ]:





# ## Lab Activity:
# 
# 1. Find and show the duplicates
# 2. Remove teh dublicates
# 3. check again the whole dataframe to see if we have dublicates

# In[ ]:


#write your code here


# In[ ]:





# In[ ]:





# In[ ]:





# #### ===== Congrats! We handled the duplicates! =====

# ## The last step of Data Cleaning: Outliers

# In[ ]:


#-- Let´s a have a more detailed look at the data

dfToClean.describe()


# In[ ]:


#-- max of minimum_nights (1000) seems unreasonable, let´s check it out!


# In[ ]:


print(dfToClean['minimum_nights'].quantile(0.05))
print(dfToClean['minimum_nights'].quantile(0.95))


# In[ ]:


#how many recorde we have with minimum_nights > 90.0


# In[ ]:


dfToClean[(dfToClean['minimum_nights'] > 90)]


# In[ ]:



# --- and > 365 ?

dfToClean[(dfToClean['minimum_nights'] > 365)]


# In[ ]:


# ------ Ok let´s remove the records ['minimum_nights'] > 365 as the outliers

dfToClean.drop(dfToClean[dfToClean['minimum_nights'] > 365].index, inplace = True) 


# In[ ]:


dfToClean.shape


# ## Lab Activity:
# 
# 1. find and remove teh outliers for price
# 

# In[ ]:


#-- witet your code here


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Here we go!!! 
# 
# Now, we have all clean records

# In[ ]:


dfClean= dfToClean


# In[ ]:


dfClean.info()


# ## *Step 3. Exploratory Data Analysis (EDA)*

# In[ ]:





# In[ ]:





# Now lets take a look at how the price is distributed

# In[ ]:


print(dfClean['price'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(dfClean['price'], color='g', bins=100, hist_kws={'alpha': 0.4});


# ## Numerical data distribution
# 

# In[ ]:


# lets first list all the types of our data from our dataset and take only the numerical ones:

list(set(dfClean.dtypes.tolist()))


# In[ ]:





# In[ ]:


df_num = dfClean.select_dtypes(include = ['float64', 'int64'])
df_num.head()


# In[ ]:


#-- let´s just select the improtant numerical columns


# In[ ]:


df_num2 = df_num[["price", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365"]]


# In[ ]:


df_num2.head()


# In[ ]:


# Now lets plot them all:

df_num2.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); 


# In[ ]:





# In[ ]:


dfClean.plot(kind='scatter', x='reviews_per_month', y='price', title='price vs reviews_per_month');


# In[ ]:





# ## Lab Activity:
# 
# Create the Scatter Plot for  x='minimum_nights', y='price'

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Correlation testing

# In[ ]:


sns.heatmap(df_num2.corr());


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





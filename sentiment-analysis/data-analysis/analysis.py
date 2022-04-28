#!/usr/bin/env python
# coding: utf-8

# ## This python notebook has the data analysis of Twitter Airline Sentiment dataset

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("../data/Tweets.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


columns = df.columns
columns


# ### Remove any duplicate tweets in the data

# In[6]:


duplicates = df[df.duplicated(keep=False)]
duplicates.sort_values("tweet_id", inplace = True)
duplicates.shape


# From the above we can see that there are 72 duplicate rows.

# In[7]:


duplicates.head(6)


# In[12]:


#removing duplicates
df.drop_duplicates(keep='first',inplace=True)


# In[13]:


duplicates = df[df.duplicated(keep=False)]
duplicates.sort_values("tweet_id", inplace = True)
duplicates.shape


# In[8]:


# size of dataset after removing duplicates
df.shape


# In[16]:


figsize=(20, 5)

ticksize = 14
titlesize = ticksize + 8
labelsize = ticksize + 5

params = {'figure.figsize' : figsize,
          'axes.labelsize' : labelsize,
          'axes.titlesize' : titlesize,
          'xtick.labelsize': ticksize,
          'ytick.labelsize': ticksize}

plt.rcParams.update(params)

plt.subplot(121)
col = "airline"
xlabel = "Airlines"
ylabel = "Count"

sns.countplot(x=df[col])
plt.title("Airlines Review Count")
plt.xticks(rotation=90)
plt.xlabel(xlabel)
plt.ylabel(ylabel)


# #### Observations:
# 1. United Airlines has most reviews
# 2. Virgin America has minimum reviews.

# In[9]:


plt.subplot(122)
col = "airline_sentiment"
xlabel = "Sentiment"
ylabel = "Count"
sns.countplot(df[col])
plt.title("Review Sentiment Count")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xticks(rotation=90)
plt.plot()


# #### Observations
# 1. there are more negative reviews then positive and neutral combined.
# 2. positive reviews are very less.

# In[10]:


figsize=(20, 5)

ticksize = 14
titlesize = ticksize + 8
labelsize = ticksize + 5

xlabel = "Airlines"
ylabel = "Count"


params = {'figure.figsize' : figsize,
          'axes.labelsize' : labelsize,
          'axes.titlesize' : titlesize,
          'xtick.labelsize': ticksize,
          'ytick.labelsize': ticksize}

plt.rcParams.update(params)

plt.figure(figsize=figsize)
col1 = "airline"
col2 = "airline_sentiment"
sns.countplot(x=df[col1], hue=df[col2])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xticks(rotation=90)
plt.plot()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Load the data
df=pd.read_csv(r'D:\data science and deepl learning 20 case studies\datascienceforbusiness-master\olympic100m.csv')


# In[4]:


# check the firt 5 rows of th data
df.head()


# In[5]:


x=df['year']


# In[6]:


x.shape


# In[7]:


# reshape the data
x_train=np.array(x).reshape((-1,1))


# In[8]:


x_train.shape


# In[9]:


# Target variable separation
y_train=np.array(df['time'])


# In[10]:


# Import the model
from sklearn.linear_model import LinearRegression


# In[11]:


model=LinearRegression()


# In[12]:


# fit the data to train the model
model.fit(x_train,y_train)


# In[13]:


# predict on the trained data
y_pred=model.predict(x_train)


# In[14]:


plt.scatter(x_train,y_train)
plt.plot(x,y_pred,color='red')


# In[15]:


##lets predict for the upcoming years

x_2020=np.array([2020]).reshape(-1,1)


# In[16]:


model.predict(x_2020)


# In[17]:


x_2024=np.array([2024]).reshape(-1,1)
model.predict(x_2024)


# In[18]:


x_2028=np.array([2028]).reshape(-1,1)
model.predict(x_2028)


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[2]:


st= pd.read_csv('F:/Dataset/50_Startups (1).csv')


# In[3]:


st


# In[4]:


sf=st.rename(columns ={'R&D Spend':'rd','Administration':'ad','Marketing Spend':'ms'}, inplace =False)


# In[5]:


sd=sf.drop(['State'], axis = 1)


# In[6]:


sd.info()


# In[8]:


sd.corr()


# In[9]:


sns.set_style(style='darkgrid')
sns.pairplot(sd)


# In[10]:


import statsmodels.formula.api as smf 
model = smf.ols('Profit~rd+ad+ms',data=sd).fit()


# In[11]:


model.params


# In[12]:


print(model.tvalues, '\n', model.pvalues)


# In[13]:


(model.rsquared,model.rsquared_adj,model.aic)


# In[14]:


new_data=pd.DataFrame({'rd':165444,"ad":90000,"ms":300000},index=[1])


# In[15]:


model.predict(new_data)


# In[ ]:





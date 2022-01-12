#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[2]:


toyota = pd.read_csv('F:/Dataset/ToyotaCorolla.csv')


# In[3]:


toyota


# In[5]:


toyota.info()


# In[8]:


toyota2=pd.concat([toyota.iloc[:,2:4],toyota.iloc[:,6:7],toyota.iloc[:,8:9],toyota.iloc[:,12:14],toyota.iloc[:,15:18]],axis=1)
toyota2


# In[9]:


toyota3=toyota2.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)


# In[10]:


toyota3


# In[11]:


toyota3[toyota3.duplicated()]


# In[12]:


toyota4=toyota3.drop_duplicates().reset_index(drop=True)
toyota4


# In[13]:


toyota4.describe()


# In[14]:


toyota4.corr()


# In[16]:


sns.set_style(style='darkgrid')
sns.pairplot(toyota4)


# In[17]:


model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyota4).fit()


# In[18]:


model.params


# In[19]:


model.tvalues , np.round(model.pvalues,5)


# In[20]:


model.rsquared , model.rsquared_adj


# In[22]:


slr_c=smf.ols('Price~CC',data=toyota4).fit()
slr_c.tvalues , slr_c.pvalues


# In[23]:


slr_d=smf.ols('Price~Doors',data=toyota4).fit()
slr_d.tvalues , slr_d.pvalues


# In[24]:


mlr_cd=smf.ols('Price~CC+Doors',data=toyota4).fit()
mlr_cd.tvalues


# In[26]:


rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=toyota4).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=toyota4).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=toyota4).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=toyota4).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=toyota4).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=toyota4).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=toyota4).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=toyota4).fit().rsquared
vif_WT=1/(1-rsq_WT)


# In[27]:


d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[29]:


sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[30]:


list(np.where(model.resid>6000))


# In[31]:


list(np.where(model.resid<-6000))


# In[32]:


def standard_values(vals) : return (vals-vals.mean())/vals.std() 


# In[33]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 


# In[34]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plt.show()


# In[35]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plt.show()


# In[36]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plt.show()


# In[37]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'CC',fig=fig)
plt.show()


# In[38]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plt.show()


# In[39]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Gears',fig=fig)
plt.show()


# In[40]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'QT',fig=fig)
plt.show()


# In[41]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Weight',fig=fig)
plt.show()


# In[42]:


(c,_)=model.get_influence().cooks_distance
c


# In[44]:


fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(toyota4)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[45]:


np.argmax(c) , np.max(c)


# In[46]:


fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)


# In[48]:


k=toyota4.shape[1]
n=toyota4.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[49]:


toyota4[toyota4.index.isin([80])]


# In[51]:


toyota_new=toyota4.copy()
toyota_new


# In[52]:


toyota5=toyota_new.drop(toyota_new.index[[80]],axis=0).reset_index(drop=True)
toyota5


# In[ ]:


while np.max(c)>0.5 :
   model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyota5).fit()
   (c,_)=model.get_influence().cooks_distance
   c
   np.argmax(c) , np.max(c)
   toyotao5=toyota5.drop(toyota5.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
   toyota5
else:
   final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toytao5).fit()
   final_model.rsquared , final_model.aic
   print("Thus model accuracy is improved to",final_model.rsquared)


# In[ ]:


if np.max(c)>0.5:
    model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyota5).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    toyota5=toyota5.drop(toyota5.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    toyota5 
elif np.max(c)<0.5:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyota5).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)


# In[ ]:


final_model.rsquared


# In[ ]:


toyota5


# In[ ]:


new_data=pd.DataFrame({'Age':12,"KM":40000,"HP":80,"CC":1300,"Doors":4,"Gears":5,"QT":69,"Weight":1012},index=[0])
new_data


# In[ ]:


final_model.predict(new_data)


# In[ ]:


pred_y=final_model.predict(toyota5)
pred_y


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[16]:


pip install pandas-profiling


# In[59]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,RidgeCV,LassoCV,ElasticNet,ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport


# In[ ]:





# In[3]:


df = pd.read_csv('Admission_Prediction.csv')
df


# In[4]:


pf = ProfileReport(df)
pf.to_widgets()


# In[5]:


df


# In[6]:


df['GRE Score'] = df['GRE Score'].fillna(df['GRE Score'].mean())


# In[7]:


df['TOEFL Score'] = df['TOEFL Score'].fillna(df['TOEFL Score'].mean())
df['University Rating'] = df['University Rating'].fillna(df['University Rating'].mean())


# In[8]:


df.describe()


# In[12]:


df.isnull().sum()


# In[13]:


df.drop(columns = ['Serial No.'],inplace = True)
df


# In[14]:


y = df['Chance of Admit']
x = df.drop(columns=['Chance of Admit'])


# In[15]:


x


# In[16]:


y


# In[17]:


scaler = StandardScaler()
arr = scaler.fit_transform(x)


# In[18]:


arr


# In[20]:


df1 = pd.DataFrame(arr)
df1.profile_report().to_widgets()


# In[21]:


df1.describe()


# In[22]:


arr


# In[23]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_df = pd.DataFrame()


# In[25]:


vif_df['vif'] = [variance_inflation_factor(arr,i) for i in range(arr.shape[1])]


# In[26]:


vif_df['feature'] = x.columns


# In[27]:


vif_df


# In[28]:


arr


# In[32]:


x_train,x_test,y_train,y_test = train_test_split(arr,y,test_size = 0.25, random_state = 100)
x_train


# In[34]:


lr = LinearRegression()
lr.fit(x_train,y_train)


# In[37]:


import pickle


# In[42]:


pickle.dump(lr,open('admission_lr_model.pickle','wb'))


# In[44]:


get_ipython().system('dir')


# In[47]:


lr.predict([[337.000000,118.0,4.0,4.5,4.5,9.65,1]])


# In[46]:


df


# In[49]:


test1 = scaler.transform([[337.000000,118.0,4.0,4.5,4.5,9.65,1]])
model = pickle.load(open('admission_lr_model.pickle','rb'))
model.predict(test1)


# In[50]:


lr.score(x_test,y_test)


# In[51]:


def adj_r2(x,y):
    r2 = lr.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[52]:


adj_r2(x_test,y_test)


# In[53]:


lr.coef_


# In[54]:


lr.intercept_


# Lasso

# In[55]:


lassocv = LassoCV(cv = 10,max_iter = 2000000, normalize = True)
lassocv.fit(x_train,y_train)


# In[56]:


lassocv.alpha_


# In[60]:


from sklearn.linear_model import Lasso


# In[61]:


lasso = Lasso(alpha = lassocv.alpha_)
lasso.fit(x_train,y_train)


# In[62]:


lasso.score(x_test,y_test)


# Ridge

# In[66]:


ridgecv = RidgeCV(alphas=np.random.uniform(0,10,50),cv = 10 , normalize = True)
ridgecv.fit(x_train,y_train)


# In[67]:


ridgecv.alpha_


# In[68]:


ridge_lr = Ridge(alpha = ridgecv.alpha_)
ridge_lr.fit(x_train,y_train)


# In[69]:


ridge_lr.score(x_test,y_test)


# ELasticNet

# In[70]:


elastic = ElasticNetCV(alphas = None,cv = 10)
elastic.fit(x_train,y_train)


# In[71]:


elastic.alpha_


# In[72]:


elastic.l1_ratio_


# In[74]:


elastic_lr = ElasticNet(alpha = elastic.alpha_ , l1_ratio=elastic.l1_ratio_)


# In[75]:


elastic_lr.fit(x_train,y_train)


# In[76]:


elastic_lr.score(x_test,y_test)


# In[ ]:





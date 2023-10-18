#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.model_selection import TimeSeriesSplit
from sklearn import linear_model, metrics, model_selection, pipeline, tree
import xgboost as xgb
from yellowbrick import regressor
import math

# In[4]:


# import food waste data
food_waste = pd.read_csv('df_foodwaste-Copy1.csv')

cols = ['timestamp', 'weight', 'client_id', 'category_label']
food_wate_clean = (food_waste
 [cols]
 .assign(date=pd.to_datetime(food_waste.timestamp))
 .astype({'weight':'float16', 'client_id' : 'category', 'category_label':'category'})
 .drop(columns=('timestamp'))
)


# In[5]:


food_waste_lumc = (food_wate_clean 
  .loc[lambda d:d['client_id'] == '3511_VERMAAT_VODAFONEHOOGCATHARIJNE_KEUKEN']
 .loc[lambda d:d['date'] >= '2020-12-11']
 .set_index('date')
 .groupby('client_id')
 ['weight']
.resample('D')
 .sum()
.reset_index()
  .set_index('date')
.assign(
       sum_weekly=lambda d:d['weight'].shift(1).rolling('7D', min_periods=1).sum().astype('float16'),
       sum_monthly=lambda d:d['weight'].shift(1).rolling('30D', min_periods=1).sum().astype('float16'))
                   .reset_index()
)


# In[12]:


food_waste_lumc


# ### EDA

# In[8]:


fig, ax = plt.subplots(figsize=(8,4))
(food_waste_lumc
.set_index('date')
.weight
.plot(ax=ax, title='Food waste over time'));


# ### Prediction

# In[9]:


from sklearn import base
class TweakDirtyTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, ycol=None):
        self.ycol = ycol
        self.y_val = None
        
    def transform(self, X):
        return tweak_dirty(X)
    
    def fit(self, X, y=None):
        return self


# In[23]:


def cyclic_encode(adf, col, x_suffix='_x', y_suffix='_y'):
    return (adf
                .assign(**{f'{col}{x_suffix}':
                    np.sin((2*np.pi*adf[col])/(adf[col].nunique())),
                        f'{col}{y_suffix}':
                    np.cos((2*np.pi*adf[col])/(adf[col].nunique())),
                })
            )

def tweak_dirty(adf):
    return (adf
            .assign(dow=adf.date.dt.day_of_week,
            day=adf.date.dt.day,
            month=adf.date.dt.month,
            doy=adf.date.dt.day_of_year
            )
            .pipe(cyclic_encode, col='dow')
            .pipe(cyclic_encode, col='day')
            .pipe(cyclic_encode, col='month')
            .pipe(cyclic_encode, col='doy')
            .loc[:, ['sum_weekly', 'sum_monthly', 'dow', 'day', 'month', 'doy', 'dow_x',
            'dow_y', 'day_x', 'day_y', 'month_x',
            'month_y', 'doy_x', 'doy_y']]
            )


# In[24]:


pl = pipeline.Pipeline([('tweak', TweakDirtyTransformer())])


# In[27]:


X = food_waste_lumc.drop(columns='weight')
y = food_waste_lumc['weight']

# drop missing y values
y = y[~y.isna()]
X = X.loc[y.index]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, shuffle=False)
X_train = pl.fit_transform(X_train)
X_test = pl.transform(X_test)

# refilter y
y_train = y_train.loc[X_train.index]
y_test = y_test.loc[X_test.index]
X = pd.concat([X_train, X_test], axis='index')
y = pd.concat([y_train, y_test], axis='index')


# In[31]:


lr = linear_model.LinearRegression()
lr.fit(X_train.fillna(0), y_train)
lr.score(X_test.fillna(0), y_test)


# In[32]:


dt = tree.DecisionTreeRegressor(max_depth=3)
dt.fit(X_train.fillna(0), y_train)
dt.score(X_test.fillna(0), y_test)


# In[40]:


xg = xgb.XGBRegressor(max_depth=3, early_stopping_rounds=10)
evaluation = [(X_train, y_train),
(X_test, y_test)]
xg.fit(X_train, y_train, eval_set=evaluation)
xg.score(X_test, y_test)


# In[41]:


fig, ax = plt.subplots(figsize=(8,4))
viz = regressor.residuals_plot(lr, X_train.fillna(0), y_train, X_test.fillna(0), y_test, qqplot=True, hist=False)

fig, ax = plt.subplots(figsize=(8,4))
viz = regressor.residuals_plot(dt, X_train.fillna(0), y_train, X_test.fillna(0), y_test, qqplot=True, hist=False)

fig, ax = plt.subplots(figsize=(8,4))
viz = regressor.residuals_plot(xg, X_train.fillna(0), y_train, X_test.fillna(0), y_test, qqplot=True, hist=False)


# In[ ]:





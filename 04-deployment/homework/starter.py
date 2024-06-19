#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


# get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd
import sys
import numpy as np


# In[4]:
year = int(sys.argv[1])
month = int(sys.argv[2])

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[5]:


dv, model


# In[6]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


# In[8]:


df.head()


# In[9]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[10]:


y_pred[4000:4025]


# In[11]:


y = df['duration']
print('Standard Deviation', np.std(y_pred))


# In[12]:


# year = 2023
# month = 3
# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# # In[15]:


# # Convert y_pred to a pandas Series
# y_pred_series = pd.Series(y_pred, name='predicted_duration')

# # Concatenate df['ride_id'] with y_pred_series
# df_result = pd.concat([df['ride_id'], y_pred_series], axis=1)

# # Display the result
# # print(df_result)


# # In[16]:


# output_file = "output.parquet"
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )


# # In[ ]:





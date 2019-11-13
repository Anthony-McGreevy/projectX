#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Anthony-McGreevy/projectX/blob/master/Inventory_Stock_Prediction_Model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


get_ipython().system('pip install -U -q PyDrive')
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

file_id = '1RgoONjxIOzYxZl36cE8V47dLjdp5yL72'
downloaded = drive.CreateFile({'id': file_id})

# Download the file to a local disk as 'exported.xlsx'.
downloaded.GetContentFile('online_retail(2009-2011).xlsx')

get_ipython().system('pip install -q xlrd')

import pandas as pd
df = pd.read_excel('online_retail(2009-2011).xlsx')
df


# Things to do 
# 
# •Data sanity check and preprocessing 
# - Use data visualisation techniques
# - Missing Data
# - Feature Selection techniques 
# - Group data into hourly
# - Add season (1,2,3,4)
# - Add day of the week in separate column
# - Deriving temperol features 
# - Transformation using StandardScaler
# 
# 
# 
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

retail_df = pd.read_excel('online_retail(2009-2011).xlsx',usecols=['Invoice', 'StockCode', 'Quantity', 'InvoiceDate', 'Price'])

retail_df.head()


# Only using the columns that will provide value to the model. 
# 

for column in retail_df.columns:
    null_vals = retail_df[column].isnull().values.sum()
    if(null_vals!=0):
        print (column,null_vals)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

retail_df.describe()


# Do we want to drop values that are below zero?

#dropping all value that have na or are duplications
retail_df.dropna(inplace=True)

retail_df.drop_duplicates(inplace=True)

retail_df.shape

retail_df.StockCode = retail_df.StockCode.astype('category')

retail_df['InvoiceDate'] = pd.to_datetime(retail_df['InvoiceDate'], format = "%m/%d/%Y %H:%M")

retail_df.head()

min(retail_df.InvoiceDate), max(retail_df.InvoiceDate) 


# Taking out any values that are below zero and any values that begin with a C in Invoice as that seems to be for products that have been cancelled.

retail_df['Invoice'] = retail_df['Invoice'].astype('str')

non_c_df = retail_df[retail_df["Invoice"].str.startswith('C')==False]

mask = (non_c_df["Quantity"]>0) & (non_c_df["Price"] >0)

grouped_hourly = non_c_df.loc[mask]

grouped_hourly.head()

grouped_hourly.describe()

#changing the time format into hourly as opposed to just minute 
from datetime import datetime
grouped_hourly['InvoiceDate'] = grouped_hourly['InvoiceDate'].apply(lambda x: pd.to_datetime(x).strftime("%y-%m-%d %H") ) 

grouped_hourly.head()

grouped_hourly_stock = grouped_hourly.groupby(['InvoiceDate','StockCode'])["Quantity"].sum().unstack()
grouped_hourly_stock.head()
grouped_hourly_stock.fillna(0,inplace=True)
grouped_hourly_stock.columns = grouped_hourly_stock.columns.astype("object")
grouped_hourly_stock.reset_index(inplace=True)
grouped_hourly_stock.head()

# **Deriving temporal features**
grouped_hourly_stock.InvoiceDate = pd.to_datetime(grouped_hourly_stock.InvoiceDate,format="%y-%m-%d %H")
grouped_hourly_stock['month'] = grouped_hourly_stock.InvoiceDate.dt.month
grouped_hourly_stock['Day_of_week'] = grouped_hourly_stock.InvoiceDate.dt.dayofweek

seasons = [1,1,2,2,2,3,3,3,4,4,4,1]
month_to_season = dict(zip(range(1,13), seasons))
season = grouped_hourly_stock.month.map(month_to_season)
grouped_hourly_stock['season'] = season
grouped_hourly_stock.head()

# Now we have added seasonality columns at the end: month, day of week, season
grouped_hourly_stock.shape

# I added a new column that will separate the days into morning, afternoon and night

grouped_hourly_stock = grouped_hourly_stock.assign(session=pd.cut(grouped_hourly_stock.InvoiceDate.dt.hour,[0,6,12,16,20],labels=['Night','Morning','Afternoon','Evening']))
grouped_hourly_stock.head()
grouped_hourly_stock.drop(['InvoiceDate'],axis=1,inplace=True)
grouped_hourly_stock.columns
category_cols = ['month', 'season', 'Day_of_week','session']

for column in category_cols:
    grouped_hourly_stock[column] = grouped_hourly_stock[column].astype("category")

expanded_trans_df = pd.get_dummies(grouped_hourly_stock)
expanded_trans_df.shape
target_labels = 5304
expanded_trans_df.columns[:target_labels]

expanded_trans_df.head()

# **Transformation using StandardScaler**
from sklearn.preprocessing import StandardScaler
expanded_scaler = StandardScaler(with_mean=False)     #to keep sparse matrix intact
expanded_scaled = expanded_scaler.fit_transform(expanded_trans_df)

# look back 24 timestamps back and 1 stamp ahead
n_hours = 24
n_features = expanded_trans_df.shape[1]
expanded_scaler.inverse_transform(expanded_scaled)

# **LSTM Modelling**
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.utils import np_utils
from keras.optimizers import sgd,adam


#Referenced from : https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


reframed = series_to_supervised(expanded_scaled, n_hours,1)


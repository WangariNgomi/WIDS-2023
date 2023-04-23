#!/usr/bin/env python
# coding: utf-8

# C

# # 1.1) Problem Statement
# 
# Accurate long-term forecasts of temperature and precipitation are an essential tool for helping people and communities prepare for and adapt to extreme weather events.
# 
# Currently, purely physics-based models dominate short-term weather forecasting. But these models have a limited forecast horizon. The availability of meteorological data offers an opportunity for data scientists to improve sub-seasonal forecasts by blending physics-based forecasts with machine learning.
# 
# Sub-seasonal forecasts for weather and climate conditions (lead-times ranging from 15 to more than 45 days) would help communities and industries adapt to the challenges brought on by climate change.
# 
# In this project,focus on longer-term weather forecasting to help communities adapt to extreme weather events caused by climate change by generating forecasts of temperature and precipitation for one year.

# # 1.2) Objectives
# ## a) Main objective
# The objective of this project is to develop a predictive model that will forecast sub-seasonal temperatures (temperatures over a two-week period) within the United States.
# 
# ## b) Specific objectives
# 
# 1. To create a machine learning model that will make the predictions.
# 
# 2. The model should can provide an indication of how the average temperatures in the next 14 days
# 
# 3. 
# 
# 4. 

# # 1.3)Specifying the question
# 
# 
# Predict the arithmetic mean of the maximum and minimum temperature over the next 14 days, for each location and start date

# # 1.4) Defining the metric of success
# 
# 1. Perform Exploratory Data Analysis on [Adapting to Climate Change Data](https://www.kaggle.com/competitions/widsdatathon2023/data).
# 
# 
# 2. Project will be considered successful if we train a machine learning model that will be able to predict Predict the arithmetic mean of the maximum and minimum temperature over the next 14 days, for each location
# 
# 
# 3. Use Root Mean Squared Error as the evaluation metric.

# # 1.5) Data Relevance and Validation
# 
# The data available is relevant for the intended analysis . More information can be found here:
# 
# [Kaggle.](https://www.kaggle.com/competitions/widsdatathon2023/overview) 
# 
# 

# # 1.6) Understanding the Context
# 
# The data set we are to work with contains the following columns:
# 

# # 1.7) The Experimental Design Taken
# 
#  Below are the steps taken in this analysis
#  
#  1. loading the required libraries
#  2. Loading and previewing data
#  3. Cleaning the data
#  4. Feature engineering
#  5. Univariate analysis
#  6. Bivariate analysis
#  7. Multivriate analysis
#  8. Implementing the solution by training a model that will give accurate predictions
#  9. Challenging the solution and giving insights on how improvements can be made.

# # 2) Reading the data

# In[6]:


# dataframe and plotting
import pandas as pd #used to analyze data
import numpy as np #used for working with arrays
import seaborn as sns #helps you explore and understand your data using stattistical graphs, it is built on matplotlib
import plotly as px

# In[7]:


# Loading train the data into a pandas dataframe
train = pd.read_csv(r'C:\Users\USER\Desktop\PROJECTS\train_data.csv')


# In[8]:


# Loading test the data into a pandas dataframe
test = pd.read_csv(r'C:\Users\USER\Desktop\MIRIAM\WIDS 2023\test_data.csv')


# In[9]:


ss = pd.read_csv(r'C:\Users\USER\Desktop\MIRIAM\WIDS 2023\sample_solution.csv')


# # 3) Understanding the data

# In[10]:


ss.head()


# In[11]:


train.head()


# In[12]:


train.tail()


# In[13]:


train.shape


# In[14]:


test.head()


# In[15]:


#Type column to be used to split train and test set from the combined dataframe
train['type'] = '0'
test['type'] = '1'

# setting the maximum number of columns that will be displayed
pd.set_option('display.max_columns', 246)

# Combine train and test set
combined_df = pd.concat((train, test)).reset_index(drop =True)
combined_df


# In[16]:


train.columns


# In[17]:


#Checking the number of entries in train and test sets:

print("The combined dataset contains {} rows, and {} columns".format(combined_df.shape[0], combined_df.shape[1]))


# In[18]:


combined_df.columns.tolist()


# In[19]:


# checking for missing values
pd.set_option('display.max_rows', 247)

train.isnull().sum()


# In[20]:


# checking for missing values
pd.set_option('display.max_rows', 247)

test.isnull().sum()


# In[21]:


# checking for missing values
pd.set_option('display.max_rows', 247)

combined_df.isnull().sum()


# In[22]:


#check for duplicate rows
sum(combined_df.duplicated())


# In[23]:


#Checking the number of records in our train dataset
combined_df.count()


# In[24]:


combined_df.dtypes


# In[25]:


#Checking for the value counts for regions column
combined_df['climateregions__climateregion'].value_counts()


# # 2) Combining Features

# In[26]:


# getting the mean of  'wind-vwnd-925-2010-1',
combined_df['avg_wind-vwnd-925-2010'] = combined_df.iloc[:,225:245].mean(axis=1)
combined_df = combined_df.drop(combined_df.iloc[:,225:246], axis = 1)
combined_df.head()


# In[27]:


# getting the mean of  'wind-hgt-100-2010-1',
combined_df['avg_wind-hgt-100-2010'] = combined_df.iloc[:,216:225].mean(axis=1)
combined_df = combined_df.drop(combined_df.iloc[:,216:225], axis = 1)
combined_df.head()


# In[28]:


# getting the mean of  'wind-hgt-10',
combined_df['avg_wind-hgt-10-2010'] = combined_df.iloc[:,206:215].mean(axis=1)
combined_df = combined_df.drop(combined_df.iloc[:,206:216], axis = 1)
combined_df.head()


# In[29]:


# getting the mean of  'wind-uwnd-925-2010',
combined_df['avg_wind-uwnd-925-2010'] = combined_df.iloc[:,186:205].mean(axis=1)
combined_df = combined_df.drop(combined_df.iloc[:,186:206], axis = 1)
combined_df.head()


# In[30]:


# getting the mean of  'icec-2010',
combined_df['avg_icec-2010'] = combined_df.iloc[:,176:185].mean(axis=1)
combined_df = combined_df.drop(combined_df.iloc[:,176:186], axis = 1)
combined_df.head()


# In[31]:


# getting the mean of  wind-hgt-500-2010,
combined_df['avg_wind-hgt-500-2010'] = combined_df.iloc[:,166:175].mean(axis=1)
combined_df =combined_df.drop(combined_df.iloc[:,166:176], axis = 1)
combined_df.head()


# In[32]:


# getting the mean of sst-2010,
combined_df['avg_sst-2010'] = combined_df.iloc[:,156:165].mean(axis=1)
combined_df = combined_df.drop(combined_df.iloc[:,156:166], axis = 1)
combined_df.head()


# In[33]:


# getting the mean of wind-hgt-850,
combined_df['avg_wind-hgt-850'] = combined_df.iloc[:,146:155].mean(axis=1)
combined_df = combined_df.drop(combined_df.iloc[:,146:156], axis = 1)
combined_df.head()


# In[34]:


# getting the mean of hgt wind-uwnd-250,
combined_df['avg_wind-uwnd-250'] = combined_df.iloc[:,121:140].mean(axis=1)
combined_df = combined_df.drop(combined_df.iloc[:,121:141], axis = 1)
combined_df.head()


# In[35]:


# getting the mean of wind-vwnd-250-2010,
combined_df['avg_wind-vwnd-250-2010'] = combined_df.iloc[:,101:120].mean(axis=1)
combined_df = combined_df.drop(combined_df.iloc[:,101:121], axis = 1)
combined_df.head()


# In[36]:


combined_df.shape


# In[37]:


pd.set_option('display.max_rows', 247)
combined_df.columns.tolist()


# # 3) Feature Engineering

# In[38]:


# Convert the 'startdate' column to datetime format
combined_df['startdate'] = pd.to_datetime(combined_df['startdate'])


# In[40]:


# Convert the 'startdate' column to datetime format
train['startdate'] = pd.to_datetime(train['startdate'])


# In[41]:


# Convert the 'startdate' column to datetime format
test['startdate'] = pd.to_datetime(test['startdate'])


# In[42]:


print('Starting date of the whole data: ', combined_df.startdate.min())
print('Ending date of the whole data: ', combined_df.startdate.max())
print('Total number of days in available in whole data: ', combined_df.startdate.nunique())


# In[43]:


print('Starting date of the data: ', train.startdate.min())
print('Ending date of the data: ', train.startdate.max())
print('Total number of days in available train data: ', train.startdate.nunique())


# In[44]:


print('Starting date of the data: ', test.startdate.min())
print('Ending date of the data: ', test.startdate.max())
print('Total number of days in available train data: ', test.startdate.nunique())


# In[ ]:


# combined_df['two_week_interval'] = combined_df.groupby(pd.Grouper(freq='2W')).ngroup()


# In[45]:


# Extract year, month and day as additional features
# Extract year, month and day as additional features
combined_df['year'] = pd.DatetimeIndex(combined_df['startdate']).year
combined_df['month'] = pd.DatetimeIndex(combined_df['startdate']).month
combined_df['day_of_month'] = pd.DatetimeIndex(combined_df['startdate']).day
combined_df['day_of_week'] = pd.DatetimeIndex(combined_df['startdate']).dayofweek
combined_df['day_of_year'] = pd.DatetimeIndex(combined_df['startdate']).dayofyear
combined_df['week_of_year'] = pd.DatetimeIndex(combined_df['startdate']).weekofyear   
       


# In[46]:


combined_df.drop(['startdate'], inplace = True, axis = 1)


# In[47]:


combined_df.drop(['climateregions__climateregion'], inplace = True, axis = 1)


# In[49]:


combined_df.columns.tolist()


# In[185]:


new_df = combined_df[['index','lat','lon','contest-pevpr-sfc-gauss-14d__pevpr','nmme0-tmp2m-34w__nmme0mean',
                     'contest-wind-h10-14d__wind-hgt-10','nmme-tmp2m-56w__nmmemean','contest-rhum-sig995-14d__rhum',
                     'nmme-prate-34w__nmmemean','contest-wind-h100-14d__wind-hgt-100','nmme0-prate-56w__nmme0mean',
                     'nmme0-prate-34w__nmme0mean','contest-tmp2m-14d__tmp2m','contest-slp-14d__slp',
                     'contest-wind-vwnd-925-14d__wind-vwnd-925','nmme-prate-56w__nmmemean',
                     'contest-pres-sfc-gauss-14d__pres','contest-wind-uwnd-250-14d__wind-uwnd-250','nmme-tmp2m-34w__nmmemean',
                     'contest-prwtr-eatm-14d__prwtr','contest-wind-vwnd-250-14d__wind-vwnd-250','contest-precip-14d__precip',
                     'contest-wind-h850-14d__wind-hgt-850','contest-wind-uwnd-925-14d__wind-uwnd-925',
                     'contest-wind-h500-14d__wind-hgt-500','nmme0mean','elevation__elevation', 'mjo1d__phase',
                     'mjo1d__amplitude','mei__mei','mei__meirank','mei__nip','type','avg_wind-vwnd-925-2010',
                     'avg_wind-hgt-100-2010','avg_wind-hgt-10-2010','avg_wind-uwnd-925-2010','avg_icec-2010',
                     'avg_wind-hgt-500-2010', 'avg_sst-2010','avg_wind-hgt-850','avg_wind-uwnd-250','avg_wind-vwnd-250-2010',
                     'year','month','day_of_month','day_of_week','day_of_year','week_of_year']]


# In[186]:


new_df.shape


# In[132]:


# get_ipython().system('pip install sweetviz')


# In[134]:


import sweetviz as sv

# Create a sweetviz report for the dataset
report = sv.analyze(new_df)

# Display the report
report.show_html()


# In[188]:


# Separate train and test data from the combined dataframe
train_df = new_df[new_df['type'] == '0']
test_df = new_df[new_df['type'] == '1']

# Check the shapes of the split dataset
train_df.shape, test_df.shape


# In[189]:


#drop type column  from train as it doesnt have any use now
train_df.drop(['type'], axis = 1, inplace = True)
train_df.info()


# In[190]:


#drop target and type column from test
test_df.drop(['type'], axis = 1, inplace= True)
test_df.info()


# In[191]:


#drop target and type column from test
test_df.drop(['contest-tmp2m-14d__tmp2m'], axis = 1, inplace= True)


# # 5)Implementing the Solution with all Identified Features

# In[193]:


# dropping rows with missing values
# Drop rows with missing values
train_df = train_df.dropna()


# In[194]:


# assigning our independent variables
features = train_df.drop(['contest-tmp2m-14d__tmp2m','index'], axis = 1)
target = train_df['contest-tmp2m-14d__tmp2m']


# In[199]:


# This step entails dividing the datasets into training and test sets
# We start by importing the neccessary libray for the same

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# In[200]:


# lets print the shapes again 
print("Shape of the x Train :", X_train.shape)
print("Shape of the y Train :", y_train.shape)
print("Shape of the x Test :", X_test.shape)
print("Shape of the y Test :", y_test.shape)


# In[201]:


import lightgbm as lgb

# # storing training and validation data in LightGBM Datasets.
# train_data = lgb.Dataset(X_train, label = y_train)
# valid_data = lgb.Dataset(X_valid, label = y_valid)

#create a model using lgbm
model = LGBMRegressor()

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
           eval_metric='l2',
          early_stopping_rounds=100)


# In[202]:


# make predictions
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration_)

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# calculate the rmse
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("RMSE:", rmse)


# In[175]:


# # Create the DMatrix from the training data
# dtrain = xgb.DMatrix(X_train, label=y_train)

# # Define the XGBoost model
# model = xgb.XGBRegressor()

# # Perform k-fold cross validation with 3 folds
# kfold = xgb.cv(model.get_params(), dtrain, num_boost_round=10, nfold=3, metrics="rmse", as_pandas=True, seed=42)

# # Print the cross-validation results
# print(kfold)


# In[176]:


# import xgboost as xgb
# from sklearn.metrics import mean_squared_error

# # # Assign your input and target data
# # X_train = your_input_train_data
# # y_train = your_target_train_data
# # X_test = your_input_test_data
# # y_test = your_target_test_data

# # Create the DMatrix from the training data
# dtrain = xgb.DMatrix(X_train, label=y_train)

# # Perform k-fold cross validation with 3 folds
# # kfold = xgb.cv(model.get_params(), dtrain, num_boost_round=10, nfold=3, metrics="rmse", as_pandas=True, seed=42)

# # Define the XGBoost model
# model = xgb.XGBRegressor()

# # Fit the model on the training data
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)


# # # Evaluate the model's accuracy on the test set
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error: %.2f" % mse)
# rmse = np.sqrt(mse)
# print("Root Mean Squared Error: %.2f" % rmse)


# In[177]:


# # # Evaluate the model's accuracy on the test set
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error: %.2f" % mse)
# rmse = np.sqrt(mse)
# print("Root Mean Squared Error: %.2f" % rmse)


# In[178]:


# rmse = np.sqrt(mse)
# print("Root Mean Squared Error: %.2f" % rmse)


# In[203]:


# Training base model
# Import XGBoost Regressor
import xgboost as xgb

from xgboost import XGBRegressor

#Create a XGBoost Regressor
reg = XGBRegressor(n_estimators = 1000)

# Train the model using the training sets 
reg.fit(X_train, y_train,
        eval_set = [(X_train, y_train), (X_test, y_test)],
        verbose = 100)


# In[182]:


# Model prediction on train data
y_pred = reg.predict(X_test)


# In[205]:


pd.DataFrame(data = reg.feature_importances_,
             index = reg.feature_names_in,
             columns = ['importance'])


# In[ ]:





# In[155]:


# Model Evaluation
from sklearn import metrics

print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# # 3) Removing Redundancy

# In[97]:


# a function to get all columns of object type
obj_list = new_df.select_dtypes(include = "object").columns
print (obj_list)


# In[71]:


#Label Encoding for object to numeric conversion
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in obj_list:
    new_df[feat] = le.fit_transform(new_df[feat])

print (new_df.info())


# In[98]:


# we begin by drawing a correlation matrix using the .corr
# wedevelop a mask since the correlation matrix will repeat itselt along tha main diagonal
# and duplicate itself. So, we will slice it to remove the duplicate data
# first we create an array filled with ones, assign upper side 1 and lower side 0
# and a one will be applied where 1 is set

# generate a mask for the upper triangle

corr = new_df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))


# In[99]:


# set up matplotlib figure
import plotly as plt
f, ax = plt.subplots(figsize =(11,9))

# generate a custom diverging colormap to show different color tones
cmap = sns.diverging_palette(230,20, as_cmap = True)

# draw the heatmap with mask and correct aspect ratio
sns.heatmap(corr, mask = mask, cmap= cmap, vmax=0.3,
           square = True, linewidth = .5, cbar_kws = {'shrink' : .5})


# In[100]:


# get correlations, it will get diagonaland lower triangular pairs of correlation matrix
def get_redundant_pairs(new_df):
    pairs_to_drop = set()
    cols = new_df.columns
    for i in range(0, new_df.shape[1]):
        for k in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[k]))
    return pairs_to_drop


# In[101]:


# sort correlations in decsending order
def get_top_correlations(new_df, n =5):
    au_corr = new_df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(new_df)
    au_corr = au_corr.drop(labels = labels_to_drop).sort_values(ascending = False)
    return au_corr[0:n]


# In[103]:


print('Top Absolute Correlations')
print(get_top_correlations(new_df, 50))


# In[116]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

features = new_df.columns

def compute_vif(features):
    
    vif = pd.DataFrame()
    vif["features"] = new_df.columns
    vif["vif_factor"] = [variance_inflation_factor(train_df[features].values, i) for i in range(len(features))]
    return vif.sort_values(by =['vif_factor'])


# In[117]:


vif_data = compute_vif(features)
vif_data


# A VIF score of 1 indicates no collinearity, while a score greater than 1 indicates that the predictor is correlated with one or more other predictors in the model. A VIF score of 5 or greater is generally considered to indicate high collinearity, and suggests that the predictor may not be providing any additional information to the model above and beyond what is already provided by the other correlated predictors. In general, a lower VIF score is better.
# 
# We are going to drop features with a vif of > 5

# In[125]:


def vif4(vif_factor):
    vifd = pd.DataFrame()
    vifd = vif_data.loc[vif_data['vif_factor'] <= 4.0]
    return vifd
    

result = vif4('vif_factor')
result


# # fitting the model after removing multicolinearity

# In[135]:


# assigning our independent variables
X = train_df[['avg_wind-vwnd-925-2010','day_of_week','contest-precip-14d__precip','contest-wind-vwnd-925-14d__wind-vwnd-925',
            'contest-wind-uwnd-925-14d__wind-uwnd-925','avg_wind-vwnd-250-2010','contest-wind-vwnd-250-14d__wind-vwnd-250']]
Y = train_df['contest-tmp2m-14d__tmp2m']


# In[148]:


# This step entails dividing the datasets into training and test sets
# We start by importing the neccessary libray for the same

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[149]:


# lets print the shapes again 
print("Shape of the x Train :", X_train.shape)
print("Shape of the y Train :", y_train.shape)
print("Shape of the x Test :", X_test.shape)
print("Shape of the y Test :", y_test.shape)


# In[137]:


# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train_norm = norm.transform(X_train)

# transform testing data
X_test_norm = norm.transform(X_test)


# In[138]:


# Training base model

# Import XGBoost Regressor
import xgboost as xgb

from xgboost import XGBRegressor

#Create a XGBoost Regressor
reg = XGBRegressor()

# Train the model using the training sets 
reg.fit(X_train_norm, y_train)


# In[142]:


# Model prediction on train data
y_pred = reg.predict(X_test_norm)


# In[144]:


# Model Evaluation
from sklearn import metrics

print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_test_norm.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# In[ ]:





# In[ ]:





# In[107]:


new_df[['avg_icec-2010', 'type']].info()


# In[ ]:


from sklearn.decomposition import PCA

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# getting summary statisctics for both categorical and numerical columns
pd.set_option('display.max_rows', 55)

combined_df.describe(include = 'all')


# In[ ]:


# !pip install sweetviz


# In[ ]:


# import sweetviz as sv

# # Create a sweetviz report for the dataset
# report = sv.analyze(train)

# # Display the report
# report.show_html()


# In[ ]:


new_df.columns.tolist()


# In[ ]:


new_df.shape


# In[ ]:


# df_copy.drop('startdate', inplace = True, axis = 1)


# In[ ]:


new_df.dtypes


# In[ ]:


# filling missing values in age column
new_df.fillna(method='ffill',inplace=True)

pd.set_option('display.max_rows', 246)
new_df.isnull().sum()


# In[ ]:


# Separate train and test data from the combined dataframe
train_df = new_df[new_df['type'] == 0]
test_df = new_df[new_df['type'] == 1]

# Check the shapes of the split dataset
train_df.shape, test_df.shape


# In[ ]:


#drop type column  from train as it doesnt have any use now
train_df.drop(['type','startdate'], axis = 1, inplace = True)
train_df.info()


# In[ ]:


#drop target and type column from test
test_df.drop(['contest-tmp2m-14d__tmp2m','type','startdate'], axis = 1, inplace= True)
test_df.info()


# In[ ]:


# #drop target and type column from test
# test_df.drop(['contest-tmp2m-14d__tmp2m'], axis = 1, inplace= True)
# test_df.info()


# In[ ]:


# Visualizing the differences between actual prices and predicted values
plt.scatter(y_train, y_pred)
plt.xlabel("Actual Avg Temp")
plt.ylabel("Predited Avg Temp")
plt.title("Actual vs Predicted Temperature")
plt.show()


# In[ ]:


# Checking residuals
plt.scatter(y_pred,y_train-y_pred)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()


# In[ ]:


#Predicting Test data with the model
y_test_pred = reg.predict(X_test)


# In[ ]:


# Model Evaluation
acc_xgb = metrics.r2_score(y_test, y_test_pred)
print('R^2:', acc_xgb)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# # 6)Implementing the Solution with all Features,normalised data

# In[ ]:


# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train_norm = norm.transform(X_train)

# transform testing data
X_test_norm = norm.transform(X_test)


# In[ ]:


# Training the model with normalised data
# Import XGBoost Regressor
import xgboost as xgb

from xgboost import XGBRegressor

#Create a XGBoost Regressor
model = XGBRegressor()

# Train the model using the training sets 
model.fit(X_train_norm, y_train)


# In[ ]:


# Model prediction on train data
y_pred_norm = model.predict(X_train_norm)


# In[ ]:


# Model Evaluation
from sklearn import metrics

print('R^2:',metrics.r2_score(y_train, y_pred_norm))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred_norm))*(len(y_train)-1)/(len(y_train)-X_train_norm.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred_norm))
print('MSE:',metrics.mean_squared_error(y_train, y_pred_norm))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred_norm)))


# In[ ]:


#Predicting Test data with the model
y_test_pred_norm = model.predict(X_test_norm)


# In[ ]:


# Model Evaluation
acc_xgb = metrics.r2_score(y_test, y_test_pred_norm)
print('R^2:', acc_xgb)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred_norm))*(len(y_test)-1)/(len(y_test)-X_test_norm.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred_norm))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred_norm))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_norm)))


# ## 7. Optimizing the model using optuna

# In[ ]:


# setting up k-fold cross validation
from sklearn.model_selection import StratifiedKFold,cross_val_score
kfold = StratifiedKFold(n_splits= 10, random_state=True)


# In[ ]:


# get_ipython().system('pip install optuna')


# In[ ]:


import optuna


# In[ ]:


# Next we’ll use Optuna to tune the hyperparameters of the XGBRegressor model. 
# Optuna lets you tune the hyperparameters of any model, not just XGBoost models.
# The first step in the process is to define an objective function. The objective function is 
# the function that Optuna will try to optimize. In our case, we’re trying to minimize the mean squared error
from sklearn.metrics import mean_squared_error

def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 100),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'random_state': trial.suggest_int('random_state', 1,42)
    }
    model = xgb.XGBRegressor(**param)
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    return mean_squared_error(y_test, y_pred)


# In[ ]:


# Create the study

study = optuna.create_study(direction='minimize', study_name='regression')
study.optimize(objective, n_trials=100)


# In[ ]:


# Print the best parameters
print('Best parameters', study.best_params)


# In[ ]:


# Print the best value
print('Best value', study.best_value)


# In[ ]:


# Print the best trial
print('Best trial', study.best_trial)


# In[ ]:


# Now that we have the optimal hyperparameters, we can use them to train the model.
# We’ll use the XGBRegressor() function to create a model with the optimal hyperparameters
# by passing in **study.best_params.

model = xgb.XGBRegressor(**study.best_params)
model.fit(X_train_norm, y_train)
y_pred = model.predict(X_test_norm)

print('MSE: ', mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))


# In[ ]:


# creating submission file
test_pred = np.expm1(model.predict(test_df))


# In[ ]:


test_pred


# In[ ]:


test_df["contest-tmp2m-14d__tmp2m"] = test_pred
test_df.to_csv("results.csv", columns=['contest-tmp2m-14d__tmp2m'])


# In[ ]:


import os
os.getcwd()


# In[ ]:


combined_df.shape


# In[ ]:


combined_df.columns.tolist()


# In[ ]:


combined_df.shape


# In[ ]:


# !pip install fancyimpute


# In[ ]:


# # filling missing values in contest-tmp2m-14d__tmp2m column
# import fancyimpute
# from fancyimpute import KNN, IterativeImputer
# knn = KNN()
# df_copy['contest-tmp2m-14d__tmp2m'] = knn.fit_transform(df_copy)


# In[ ]:


import lightgbm as lgb
from lightgbm import LGBMRegressor


# In[ ]:


#create a model using lgbm
model = LGBMRegressor()
model.fit(X_train, y_train,
         eval_set=[(X_test, y_test)],
        eval_metric='l2',
        early_stopping_rounds=1000)

# # Make predictions
# y_pred = model.predict(X_valid)
# print(f'F1 score on the X_test is: {f1_score(y_valid, y_pred)}')


# In[ ]:


y_pred = model.predict(X_train, num_iteration=model.best_iteration_)


# In[ ]:


import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# calculate the rmse
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print("RMSE:", rmse)


# In[ ]:


test_pred = np.expm1(model.predict(test_df, num_iteration=model.best_iteration_))


# In[ ]:


test_df["contest-tmp2m-14d__tmp2m"] = test_pred
test_df.to_csv("results.csv", columns=['contest-tmp2m-14d__tmp2m'])


# # 7)Implementing the Solution with selected Features

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
select_univariate = SelectKBest(f_regression,k=10).fit(X,Y)
feature_mask = select_univariate.get_support()
X.columns[feature_mask]
#select_univariate.scores_


# In[ ]:


with open('scores.txt','w') as f:
    f.write(str(pd.DataFrame({'FeatureName':X.columns,
              'Scores': select_univariate.scores_}).sort_values(by='Scores',
                                                                ascending=False)))
#pd.DataFrame({'FeatureName':features.columns,
            #  'Scores': select_univariate.scores_}).sort_values(by='Scores',
                           #                                     ascending=False)


# In[ ]:


uni_df = pd.DataFrame({'Univariate Method': X.columns[feature_mask]})
uni_df


# In[ ]:


Selected_features =train_df['nmme-tmp2m-56w__ccsm4','nmme-tmp2m-56w__cfsv2','nmme-tmp2m-56w__gfdlflora',
                    'nmme-tmp2m-56w__gfdlflorb','nmme-tmp2m-56w__nmmemean','nmme-tmp2m-34w__ccsm4',
                    'nmme-tmp2m-34w__cfsv2','nmme-tmp2m-34w__gfdlflora','nmme-tmp2m-34w__gfdlflorb',
                    'nmme-tmp2m-34w__nmmemean']


# In[ ]:


target = ['contest-tmp2m-14d__tmp2m']


# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_df[Selected_features], train_df[target], test_size=0.2, random_state = 42)


# In[ ]:


# Training the model
# Import XGBoost Regressor
import xgboost as xgb

from xgboost import XGBRegressor

#Create a XGBoost Regressor
reg1 = XGBRegressor()

# Train the model using the training sets 
reg1.fit(X_train, y_train)


# In[ ]:


# Model prediction on train data
y_pred1 = reg1.predict(X_train)


# In[ ]:


# Model Evaluation
acc_xgb = metrics.r2_score(y_test, y_pred1)
print('R^2:', acc_xgb)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_pred1))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_pred1))
print('MSE:',metrics.mean_squared_error(y_test, y_pred1))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))


# In[ ]:





# In[78]:


# submission = pd.DataFrame({'contest-tmp2m-14d__tmp2m': y_test_pred}, index=test_data['index'])
# submission.to_csv('submission_27v2.csv', index=True)


# In[ ]:






# coding: utf-8

# # Environmental setting

# In[4]:

import os
import sys
import math
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt # for drawing purpose
import seaborn as sns # for drawing purpose
color = sns.color_palette() # for coloring purpose
#import xgboost as xgb

# working directory
os.chdir("C:\\Users\\popeg\\Desktop\\Big_Data_Project\\Sberbank")


# In[8]:

train = pd.read_csv("train.csv",parse_dates=['timestamp'])
macro = pd.read_csv("macro.csv",parse_dates=['timestamp']) # i will put it to trainning set when the backbone is formed:)
test = pd.read_csv("test.csv",parse_dates=['timestamp'])
train.to_pickle('train.pkl') # easy to extract file without reading again:)   


# # Viewing the data

# In[3]:

train.loc[0:5,:]


# In[9]:

# check the distribution of our target variable: price_doc
plt.figure(figsize=(12,8))
sns.distplot(train.price_doc.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()


# In[43]:

# Since our metric is Root Mean Square Logarithmic error, let us plot the log of price_doc variable.
plt.figure(figsize=(12,8))
sns.distplot(np.log(train.price_doc.values), bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()


# In[10]:

# check the relations between specific time-period and price_doc

train_df = train

# year and month #
train_df["yearmonth"] = train["timestamp"].dt.year*100 + train["timestamp"].dt.month # 201101, which is Jan, 2011

# year and week #
train_df["yearweek"] = train["timestamp"].dt.year*100 + train["timestamp"].dt.weekofyear # 201101, which is first week of 2011

# year #
train_df["year"] = train["timestamp"].dt.year

# month of year #
train_df["month_of_year"] = train["timestamp"].dt.month

# week of year #
train_df["week_of_year"] = train["timestamp"].dt.weekofyear

# day of week #
train_df["day_of_week"] = train["timestamp"].dt.weekday


############
# ploting:0
############

plt.figure(figsize=(12,8))
sns.pointplot(x='yearweek', y='price_doc', data=train_df)
plt.ylabel('price_doc', fontsize=12)
plt.xlabel('yearweek', fontsize=12)
plt.title('Median Price distribution by year and week_num')
plt.xticks(rotation='vertical')
plt.show() # according to the plot, it seems like the price_doc has the trend of getting expensive.

plt.figure(figsize=(12,8))
sns.boxplot(x='month_of_year', y='price_doc', data=train_df)
plt.ylabel('price_doc', fontsize=12)
plt.xlabel('month_of_year', fontsize=12)
plt.title('Median Price distribution by month_of_year')
plt.xticks(rotation='vertical')
plt.show()

plt.figure(figsize=(12,8))
sns.pointplot(x='week_of_year', y='price_doc', data=train_df)
plt.ylabel('price_doc', fontsize=12)
plt.xlabel('week of the year', fontsize=12)
plt.title('Median Price distribution by week of year')
plt.xticks(rotation='vertical')
plt.show() # somehow, in the beggining of the year, the average price is relatively higher.

plt.figure(figsize=(12,8))
sns.boxplot(x='day_of_week', y='price_doc', data=train_df)
plt.ylabel('price_doc', fontsize=12)
plt.xlabel('day_of_week', fontsize=12)
plt.title('Median Price distribution by day of week')
plt.xticks(rotation='vertical')
plt.show()


# In[13]:

grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,8))
sns.barplot(grouped_df.yearmonth.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Year Month', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[64]:

# check the relation between floor number and price_doc


grouped_df = train_df.groupby('floor')['price_doc'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,8))
sns.pointplot(grouped_df.floor.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show() # seems like at floor number 33, the median price is the highest


# In[14]:

# visualize the frequeency table of sub_area

plt.figure(figsize=(12,8))
sns.countplot(x="sub_area", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('sub_area', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[12]:

# check the relation between sub_area and price_doc

plt.figure(figsize=(12,8))
sns.pointplot(x='sub_area', y='price_doc', data=train_df)
plt.ylabel('price_doc', fontsize=12)
plt.xlabel('sub_area', fontsize=12)
plt.title('Median Price distribution by sub_area')
plt.xticks(rotation='vertical')
plt.show() # somehow, in the beggining of the year, the average price is relatively higher.


# In[47]:

# check the missing values within columns

missing_df = train.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0] # pick up columns with missing values
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# # Now train the training data:)

# In[ ]:

# retrieve the original trainning dataset:0

train = pd.read_pickle('train.pkl')


# In[ ]:

# check every feature in train-set
TrainFeatures = [i for i in train.colutrain = pd.read_pickle('train.pkl')mns]
TrainFeatures


# In[ ]:

# check the correlation between price_doc and other features 
# maybe i can drop features with very low variance

FeatureCorrelate = train.corr()["price_doc"]
FeatureCorrelate


# In[ ]:

FeatureCorrelate.describe()


# # # now fill all the NA cell within columns of train-set 
# 

# In[197]:

# make all the Nan rows be filled with mean of its column feature
# [train[i].fillna(train[i].mean(),inplace = True) for i in col_has_nul]
for col in col_has_nul:
    train[col].fillna(train[col].mean(),inplace = True)


# # Handle binary variables 

# In[185]:

# find out all non-numeric variable :)
non_numeric_col = [train.columns[i] for i in range(len(train.dtypes)) if train.dtypes[i] =='object']
non_numeric_col = [i for i in non_numeric_col if i not in ["timestamp","sub_area"]]


# In[193]:

number = preprocessing.LabelEncoder()
for col in non_numeric_col:
    train[col] = number.fit_transform(train[col].astype('str'))


# #  This fuction extract and prune the features of trainning data set we want:)

# In[44]:

# Features Refiner:)

def featureRefiner(df):
    
    # required module: sklearn.preprocessing and pandas   
    
    #df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    props = {}
    for prop in ["month", "day"]:
        props[prop] = getattr(df["timestamp"].dt, prop)
    
    carryover = [p for p in df.columns if p not in ["id", "timestamp","price_doc"]] # drop id and timestamp 
    for prop in carryover:
        props[prop] = df[prop]
    
    refined_df = pd.DataFrame(props)
    refined_df["price_doc"] = df["price_doc"] # put price_doc to the last column:)
    
    # handle columns which contain Na/missing values
    col_has_nul = [col for col in refined_df.columns if refined_df[col].isnull().values.any() == True] # find out which column contains na   
    for col in col_has_nul: # make all the Nan rows be filled with mean of its column feature
        refined_df[col].fillna(refined_df[col].mean(),inplace = True)
    
    # handle non-numeric columns
    non_numeric_col = [refined_df.columns[i] for i in range(len(refined_df.dtypes)) if refined_df.dtypes[i] =='object']
    # non_numeric_col = [i for i in non_numeric_col if i not in ["timestamp"]] # it's redundant
    number = preprocessing.LabelEncoder()
    for col in non_numeric_col:
        refined_df[col] = number.fit_transform(refined_df[col].astype('str'))
        
    return refined_df


# In[45]:

pipi = featureRefiner(train) # pipi is just a pseudo name of to-be-evaluated DF.

pipi.to_csv('train_ready.csv')


# #  Execution :)

# In[70]:

# Execution :)
pipi.price_doc = np.log1p(pipi.price_doc.values) #Since our metric is "RMSLE", let us use log of the target variable for model building rather than using the actual target variable.

predictors = [i for i in pipi.columns if i not in ["price_doc"]]
myregressor = RandomForestRegressor(n_estimators= int(math.sqrt(len(pipi.columns)-1)), min_weight_fraction_leaf=0.1)
# int(math.sqrt(len(pipi.columns)-1))

# evaluation
scores = cross_validation.cross_val_score(myregressor, pipi[predictors], pipi['price_doc'], cv=10) # algorithm default setting, X, Y, number of folds...

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[5]:

train = pd.read_pickle('train.pkl')
freq_table = table(train.sub_area)
freq_table


# In[ ]:

#[random.normalvariate(0,2) for i in range(142)]
hehe = np.random.normal(0,2,142)
hehe.sort()
hehe


# In[ ]:

enc = preprocessing.OneHotEncoder()
number = preprocessing.LabelEncoder()
#train.thermal_power_plant_raion = number.fit_transform(train.thermal_power_plant_raion.astype('str'))
train.sub_area = number.fit_transform(train.sub_area.astype('str'))
train.sub_area = enc.fit_transform(train.sub_area.astype('str'))
#train['sub_area'] = enc.fit_transform(train['sub_area'].astype('str'))   
#train.sub_area


# In[16]:

train.sub_area.sort_values(ascending=True)


# In[14]:

# this line can make frequency table:)
train.sub_area.value_counts()


# In[ ]:

# get 146 dummies features first, then use PCA to extract the essence:) This may work better?

from sklearn.decomposition import PCA
sub_area = pd.get_dummies(train.sub_area)
pca = PCA(n_components=3)
dest_small = pca.fit_transform(sub_area[[int("{}".format(i)) for i in range(sub_area.shape[1])]])
dest_small = pd.DataFrame(dest_small)


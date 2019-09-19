# PUBG Machine Learning
# Import script
# Imports, chunks, splits, and then saves the data for training purposes

# imports
import pandas as pd
import numpy as np
from sklearn import model_selection
import os

# parameters ===========================================================================================================
#features for all data points pulled, y is always the last value listed
#subset filters the applicable instances (e.g. solo matches, duos, team)
#subset for filtering, currently set to matchtype

features =['kills','headshotKills','heals','damageDealt','DBNOs','walkDistance','winPlacePerc']
subset='solo'

predictors=len(features)-1
# types dictionary

#types ={'kills':np.int64,'headshotKills':np.int64,'walkDistance':np.int64}

#types = {'DBNOs':int , 'assists':int , 'boosts':int, 'damageDealt':float,
#         'headshotKills':int,'heals':int,'Id':str,'killPlace':int, 'killPoints':int,
#         'killStreaks':int, 'kills':int,'longestKill':float,'matchDuration':int,
#         'matchID':int,'matchType':str,'rankPoints':int,'revives':int,'rideDistance':float,
#         'roadKills':int, 'swimDistance':float,'teamKills':int,'vehicleDestroys':int,
#         'walkDistance':float,'weaponsAcquired':int,'winPoints':int,'groupID':int,
#         'numGroups':int,'maxPlace':int,'winPlacePerc':float}

# Chunking =============================================================================================================

train_data= pd.read_csv(r'.\pubg-finish-placement-prediction\train_V2.csv',chunksize=100000)

data = pd.DataFrame()

for chunk in train_data:
    chunk = chunk[chunk.matchType==subset]
    chunk = chunk[features]
    data=pd.concat([data,chunk])

# pre-cleaning==========================================================================================================

#splitting data into x and y
x=data[features[:predictors]]
y=data[features[-1]].to_frame()

#bucketing by groups of 5
#y['winPlaceInt']=(y['winPlacePerc']*100)-((y['winPlacePerc']*100)%5)

#converting to ints, seeing as their stored as floats for some reason
y['winPlaceInt']=(y['winPlacePerc']*100)

x.dropna()
y.dropna()

# saving raw csv========================================================================================================
try:
    os.chdir(r'./cleaned_data')
except:
    os.mkdir(r'./cleaned_data')
    os.chdir(r'./cleaned_data')

x.to_csv('x_data.csv')
y.to_csv('y_data.csv')

# Splitting into test and train ========================================================================================
y=y.drop(columns=['winPlacePerc'])
y=y.astype(int)

# setting type
x = x.astype(np.int)
y = y.astype(np.int)

# converting to numppy array to allow for auto splitting
x = x.to_numpy()
y = y.to_numpy()

y=np.ravel(y)

print(x.shape)
print(y.shape)
#splitting
x_train, x_test, y_train,y_test = model_selection.train_test_split(x,y,test_size=.3)

# Saving split to csv files ============================================================================================
x_train_pd=pd.DataFrame(x_train, columns=features[:predictors])
y_train_pd=pd.DataFrame(y_train, columns=[features[-1]])

x_train_pd.to_csv('x_train_data.csv')
y_train_pd.to_csv('y_train_data.csv')

x_test_pd=pd.DataFrame(x_test, columns=features[:predictors])
y_test_pd=pd.DataFrame(y_test, columns=[features[-1]])

x_test_pd.to_csv('x_test_data.csv')
y_test_pd.to_csv('y_test_data.csv')

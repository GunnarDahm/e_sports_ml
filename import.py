# PUBG Machine Learning
# Ultimate purpose is to determine the placement of each player in a match

# imports
import pandas as pd
import numpy as np


# parameters ==========================================================================================================
#attributes for all data points pulled, y is always the last value listed
#subset for filtering, currently set to matchtype

attributes =['kills','headshotKills','heals','damageDealt','DBNOs','walkDistance','winPlacePerc']
subset='solo'

predictors=len(attributes)-1
# types dictionary

#types ={'kills':np.int64,'headshotKills':np.int64,'walkDistance':np.int64}

#types = {'DBNOs':int , 'assists':int , 'boosts':int, 'damageDealt':float,
#         'headshotKills':int,'heals':int,'Id':str,'killPlace':int, 'killPoints':int,
#         'killStreaks':int, 'kills':int,'longestKill':float,'matchDuration':int,
#         'matchID':int,'matchType':str,'rankPoints':int,'revives':int,'rideDistance':float,
#         'roadKills':int, 'swimDistance':float,'teamKills':int,'vehicleDestroys':int,
#         'walkDistance':float,'weaponsAcquired':int,'winPoints':int,'groupID':int,
#         'numGroups':int,'maxPlace':int,'winPlacePerc':float}

#Chunking==============================================================================================================

train_data= pd.read_csv(r'.\pubg-finish-placement-prediction\train_V2.csv',chunksize=100000)

data = pd.DataFrame()

for chunk in train_data:
    chunk = chunk[chunk.matchType==subset]
    chunk = chunk[attributes]
    data=pd.concat([data,chunk])

# pre-cleaning==========================================================================================================

#splitting data into x and y
x=data[attributes[:predictors]]
y=data[attributes[-1]].to_frame()

#bucketing by groups of 5
#y['winPlaceInt']=(y['winPlacePerc']*100)-((y['winPlacePerc']*100)%5)

#converting to ints
y['winPlaceInt']=(y['winPlacePerc']*100)

x.dropna()
y.dropna()

y=y.drop(columns=['winPlacePerc'])
y=y.astype(int)

# saving csv============================================================================================================
x.to_csv('x_data.csv')
y.to_csv('y_data.csv')

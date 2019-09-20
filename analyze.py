# PUBG Machine Learning
# analysis script
# Imports a trained model and analyzes the performance by comparing each error rate

#imports
from sklearn import model_selection, ensemble, metrics
import pandas as pd
import joblib
import numpy as np

# model selected =======================================================================================================
# omit extension name
model_name = 'gradient_boosting_model'

# importing train data =================================================================================================

x_train = pd.read_csv(r'./cleaned_data/x_train_data.csv')
y_train = pd.read_csv(r'./cleaned_data/y_train_data.csv')

x_train = x_train.drop(x_train.columns[0], axis=1)
y_train = y_train.drop(y_train.columns[0], axis=1)

# converting to numppy array
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

y_train = np.ravel(y_train)

# importing test data ==================================================================================================
x_test=pd.read_csv(r'./cleaned_data/x_test_data.csv')
y_test=pd.read_csv(r'./cleaned_data/y_test_data.csv')

# cleaning
x_test=x_test.drop(x_test.columns[0], axis=1)
y_test=y_test.drop(y_test.columns[0], axis=1)

x_test_np = x_test.to_numpy()
y_test_np = y_test.to_numpy()

# loading model ========================================================================================================
model=joblib.load((r'./models/'+model_name+'.joblib'))

# evaluating mean squared error ========================================================================================
test_mse = metrics.mean_absolute_error(y_test_np, model.predict(x_test_np))

print('Test Mean Squared Error: ',test_mse)

train_mse = metrics.mean_absolute_error(y_train, model.predict(x_train))

print('Train Mean Squared Error: ',train_mse)

# creating the prediction vs. actual data set ==========================================================================

predictions=[]

for n in range(len(x_test)):
    slice=x_test.loc[n]
    slice=slice.to_numpy()
    slice=slice.reshape(1,-1)
    predictions.append(float(model.predict(slice)))

predictions_pd=pd.DataFrame({'predicted_winPlacePerc':predictions})

# combining dataframes and saving ======================================================================================

test_evaluation=pd.merge(x_test,predictions_pd,left_index=True, right_index=True)

test_evaluation=pd.merge(test_evaluation,y_test,left_index=True, right_index=True)

test_evaluation['error']=test_evaluation.winPlacePerc-test_evaluation.predicted_winPlacePerc

test_evaluation.to_csv((r'./models/'+model_name+'_evaluation.csv'))
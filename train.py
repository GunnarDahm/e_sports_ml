# PUBG Machine Learning
# training script
# Imports the pre-split data and trains/tests a model. Exports models to models directory. Be sure to rename the
# exported model on line 17

# imports
from sklearn import ensemble, metrics, linear_model
import pandas as pd
import joblib
import numpy as np
import os

# Model Utilized =======================================================================================================
# exchange for other models in the SK learn library as necessary

model = linear_model.LassoLars(alpha=.00001)
model_name = 'lars_lasso_model'

# Gradient Boosting example
# model = ensemble.GradientBoostingRegressor(n_estimators=10000,
#                                              learning_rate=0.1,
#                                              max_depth=4,
#                                              min_samples_leaf=9,
#                                              max_features=0.1,
#                                              loss='huber')

# importing train data =================================================================================================
# setting dir
os.chdir(r'.\cleaned_data')

# retrieval
x_train = pd.read_csv(r'x_train_data.csv')
y_train = pd.read_csv('y_train_data.csv')

x_train = x_train.drop(x_train.columns[0], axis=1)
y_train = y_train.drop(y_train.columns[0], axis=1)

# cleaning
x_train = x_train.astype(np.int)
y_train = y_train.astype(np.int)

# converting to numppy array
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

y_train = np.ravel(y_train)

# importing test data ==================================================================================================
x_test = pd.read_csv('x_test_data.csv')
y_test = pd.read_csv('y_test_data.csv')

x_test = x_test.drop(x_test.columns[0], axis=1)
y_test = y_test.drop(y_test.columns[0], axis=1)

# cleaning
x_test = x_test.astype(np.int)
y_test = y_test.astype(np.int)

# converting to numppy array
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

y_test = np.ravel(y_test)

print('Data set, training model.')

# training the model ===================================================================================================

model.fit(x_train, y_train)

# saving the model =====================================================================================================

os.chdir(os.path.dirname(os.getcwd()))

try:
    os.chdir(r'./models')
except:
    os.mkdir(r'./models')
    os.chdir(r'./models')

# RENAME MODEL FOR EACH MODEL USED
joblib.dump(model, (model_name + '.joblib'))

# Evaluating ===========================================================================================================
mse = metrics.mean_absolute_error(y_train, model.predict(x_train))

print('Mean Squared Error: ',mse)


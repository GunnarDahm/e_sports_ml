
import matplotlib.pyplot as plt
from sklearn import model_selection, ensemble, metrics
import pandas as pd
import joblib
import numpy as np


# importing data
x=pd.read_csv('x_data.csv')
y=pd.read_csv('y_data.csv')

x=x.drop(x.columns[0], axis=1)
y=y.drop(y.columns[0], axis=1)

# cleaning
x = x.astype(np.int)
y = y.astype(np.int)

# converting to numppy array
x = x.to_numpy()
y = y.to_numpy()

y=np.ravel(y)

print(x.shape)
print(y.shape)

#randomizing traning and testing data

x_train, x_test, y_train,y_test = model_selection.train_test_split(x,y,test_size=.3)

x_test_pd=pd.DataFrame(x_test)
y_test_pd=pd.DataFrame(y_test)

x_test_pd.to_csv('x_test_data.csv')
y_test_pd.to_csv('y_test_data.csv')

print('Data set. Training model.')

#training the model

model = ensemble.GradientBoostingRegressor(n_estimators=10000,
                                              learning_rate=0.1,
                                              max_depth=4,
                                              min_samples_leaf=9,
                                              max_features=0.1,
                                              loss='huber')

model.fit(x_train,y_train)

# saving the model

joblib.dump(model,'model.joblib')

mse = metrics.mean_absolute_error(y_test, model.predict(x_test))

print(mse)



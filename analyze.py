#Analyzing the Data

#imports
from sklearn import model_selection, ensemble, metrics
import pandas as pd
import joblib
import numpy as np

# importing data
x=pd.read_csv('x_test_data.csv')
y=pd.read_csv('y_test_data.csv')

# cleaning
x=x.drop(x.columns[0], axis=1)
y=y.drop(y.columns[0], axis=1)

x = x.astype(np.int)
y = y.astype(np.int)

x = x.to_numpy()
y = y.to_numpy()

# loading model
model=joblib.load('model.joblib')


# evaluating mean squared error
mse = metrics.mean_absolute_error(y, model.predict(x))

print(mse)

# creating the prediction vs. actual data set

prediction=[]
error=[]
for n in x:
    prediction.append(model.predict(n))
    error.append((model.predict(n)-n))

prediction=np.array(prediction)
error=np.array(error)

comparison = np.append(y,prediction)

comparison = np.append(comparison,error)

print(comparison)
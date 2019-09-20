# PUBG Machine Learning
# plot script
# Visualizes the regression errors

#imports
import pandas as pd
import matplotlib.pyplot as plt

# model to be analyzed =================================================================================================
# omit the filetype when choosing which model to evaluate
model_name='gradient_boosting_model_evaluation'

# importing the data ===================================================================================================
data=pd.read_csv(r'./models/'+model_name+'.csv')

errors=[]

for n in range (0,100):
    error_slice = data[data.winPlacePerc==n]
    error_slice=error_slice['error']
    error_slice= error_slice.to_list()

    errors.append(error_slice)



# initializing plot=====================================================================================================
plt.style.use('fivethirtyeight')

fig = plt.figure()

fig.suptitle('Error by Placement')

ax1 = fig.add_subplot(1, 1, 1)

ax1.boxplot(errors)

ax1.set_xlabel('Placement')

ax1.set_ylabel('Error')

plt.xticks(rotation = 90)

plt.show()
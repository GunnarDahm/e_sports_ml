<h1> E-Sports Machine Learning Model </h1>

<h2> Status: Active </h2>

<h2> Purpose </h2>
<p> The purpsose of this project is to create versatile platform to manipulate the data of and train a model on the 
metrics of player performance in the competitive E-sport game, Player Unkown's Battle Grounds (PUBG).

<h2> Methods Used </h2>

<ul>
<li>Machine Learning</li>
<li>Predictive Modeling</li>
<li>Data Visualization</li>
</ul>

<h2> Technologies </h2>
<ul>
<li>Python</li>
</ul>

<h2> Required Libraries </h2>
<ul>
<li>Pandas</li>
<li>NumPy</li>
<li>Sklearn</li>
<li>joblib</li>
</ul>

<h2> Project Description </h2>

<p>
This project utilizes the base data as published by PUBG Corp on Kaggle for a competition. See the dataset here:
<a href="https://www.kaggle.com/c/pubg-finish-placement-prediction">
https://www.kaggle.com/c/pubg-finish-placement-prediction</a>
. This project allows for features and targeted prections to be easily interchanged. Additionally, the project also 
allows for quick exchanging of models used to train the data. Lastly, the project includes a basic visualization of 
predictions for purposes of seeing additional patterns.
</p>

<p>
The import.py file imports the data from the specified features and subsets as specified within the file
 from the pubg-finish-placement-prediction directory. The import.py file then splits the file into train and test sets
 and saves them as csvs in the data directory.
</p>

<p>
The train.py file imports the data from the data directory, trains a model, and then saves a model.joblib file in the 
models directory. This train.py file also spits out a MSE value on the test data.
</p>

<p>
The analyze.py file imports a model and then generates a comparison csv of predicted versus actual values. From this
csv, this script creates a visualization of errors and differences per the predicted values.
</p>


<h2> To Do: </h2>
<ul>
<li>Complete visualizations within the analyze.py file</li>
<li>Run regressions for Naive Bayes</li>
<li>Run regressions for Random Forest</li>
<li>Run regressions for K-Means</li>
</ul>

<h2> Author:</h2>
<p> Gunnar Dahm </p>
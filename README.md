<h1> E-Sports Machine Learning Model </h1>

<h2> Status: Completed </h2>

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
<li>Python (v.3.7)</li>
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
predictions for purposes of seeing additional patterns. These scripts were written in a generalized format to be easily 
repurposed for other datasets and files.
</p>

<p>
For those unaware, Player Unkown's Battlegrounds is a battle royale style game, often compared to Epic's Fortnite. The 
game involves approximately one hundred players dropping on to an island, and the game concludes when one player is left
standing. The creator's of PUBG published this dataset with each observation representing a player's stats including 
distance traversed, damage dealt, knockdowns, etc.
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
csv, this script creates a visualization of errors and differences per the predicted values.  The plot.py file merely 
plots the erros by placement in a box and whiskers plot.
</p>

<h2> To Use: </h2>

<p>
To run the process from the start, open the import.py file. Change the parameters as noted within the first few lines 
of the file to adjust the preferred parameters and subsets. Run the import.py file. All training and testing datasets 
are saved in the cleaned_data directory.
</p>

<p>
Then open the train.py file. Models can be exchanged as necessary from the SKLearn library and hyper-parameters may be
tuned as necessary. Trained models are saved to the models directory. Be sure to update the model name parameter as well
to ensure that old models are not saved over.
</p>

<p>
Running the analyze file will export a csv of all features as well as the errors between actual versus predicted values.
This script will also export the train vs. test MSE. The CSV will be exported to the models directory. To use, ensure 
the correct model_name is being referrenced.
</p>

<p>
The plot.py script, when run, will generate a box and whisker plot for the error for each discrete instance 
of the predicted value (i.e. each player's actual final placement). Similar to the analyze script, be sure to change the 
name of the model_name to model_evaluation.csv file you would like to analyze.
</p>


<h2> Author:</h2>
<p> Gunnar Dahm </p>
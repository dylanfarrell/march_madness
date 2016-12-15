# march_madness

<<<<<<< HEAD
# Data

- raw data found in /datasets/kaggle_data
- predictors found in /datasets/our_data
    - each predictor is a matrix with the rows being the year and the column being the team ID
- predictions for 2015 and 2016 are in /datasets/predictions

# Modules
- these are helper files that allow us to easily setup infrastructure to train/test models and run tournaments

- march_madness_classes        : code to setup tournaments, simulations, and ensembles
- march_madness_games          : code to query the database 
- march_madness_models         : code to faciliate head to head models (used in combo with march_madness classes)
- march_madness_train_and_tune : code to do train test split

# Data Exploration/Cleaning
- these files are used to generate predictors from the regular season data, and to explore relation with the response

- Kaggle Database : code to take our first crack of exploration
- Markov          : code to model the ncaa as a markov chain/ explore connection with response
- Advanced        : code to test some more predictors

# Modeling 
- Baseline : code to train a baseline head to head model
- Model Selection : code to choose predictors and train our final head to head model
- Optimize for tourney structure : code to induce upsets/ maximize expected score
- Ensemble : code to predict many unique brackets for submission

# Prediction
- Our final 2 years of prediction - no tuning was done, to see how our method works.
=======
This file will teach you how to move around this project.

1) Data

2) Modules

3) Infrastructure

4) Data Cleaning/Exploration

5) Modeling

6) Prediction


# 1) Data

  a) Raw Data - The raw data are help at datasets/kaggle_data. See https://www.kaggle.com/c/march-machine-learning-mania-2016/data for a detailed explanation of each file. We extracted all of our predictors from this dataset.
  
  b) Predictor Database - The predictor database is held at datasets/our_data. Each file is a regular season statistic. Each file is setup the same way. The columns are team IDs and the rows are years. Thus, they are all 31x282 matrix, where the (i,j) entry is the value in year 1985 + i for team 1100 + j.
  
  c) Predictions - The predictions for 2015 and 2016 (our "test set") are found at datasets/predictions. We have multiple types of files here. There is the single_bracket_predictions, which are the predicitons for a bracket pool, where you are only allowed to submit one bracket. The are also ensemble_bracket_i_predictions, which is the predictions from each individual bracket. 
  
# 2) Modules
  
  a) March Madness Classes - This file has 3 classes - Tournament, Simulator, Ensemble. Each class allows you to make a prediction of some type. Tournament lets you pass in a Model for predicting head to head games and the seeding and it generates an entire bracket. Simulator takes in a head to head model and runs a simulation n times, predicting a bracket based on max expected points scores. Ensemble creates 10 different brackets that maximizes diversity to submit for ESPN bracket challenge. 
  
  b) March Madness Models - This module contains var
  
  c) March Madness Games
  
  d) March Madness Train and Tune
  
# 3) Infrastructure

  a) Test Files
  
# 4) Data Cleaning/Exploration

  a) Basic
  
  b) Markov
  
  c) Advanced
  
# 5) Modeling

  a) Head To Head Model
  
  b) Optimize For Tournament Structure
  
  c) Ensemble of Brackets  
  
# 6) Prediction
>>>>>>> dbcff8935006f6758fe4c1f8322da51e5f41ebb1

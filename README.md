# march_madness

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

# march_madness

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

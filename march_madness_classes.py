import pandas as pd
import numpy as np
import march_madness_models as mmm

### This module contains classes that are used for structuring the tournament 

### CLASSES
# 1) Tournament -> Creates a tournament structure based on the seeds, slots, and model of a given year

# 2) Simulator -> Runs a tournament under some model n times, uses expected points scored as metric to optimize points scored

# 3) Ensemble -> Runs 10 tournaments, each differing from eachother to maximize diversity and uniqueness

########################################################################################################

### Tournament Class
### arguments = seeds, slots, and model
### proceeds to generate the entire tournament based on the model that is passed
### creates a dataframe of all games with predictions

### PROPERTIES
# 1) round_i_df --> df of games in round i

# 2) entire_bracket --> df of all games in the tournament

### METHODS
# 1) init --> runs the entire tournament, taking results of prev round 

# 2) score_tournament --> takes in the actual results of the tournament year and scores it according to ESPN

# 3) compare_to_dif_tournament --> computes the number of games different from another tournament 

# 4) get_predicted_points --> returns a dataframe of how many points each team scores under the model

class Tournament(object):
    # init function
    def __init__(self, seeds, slots, model, include_scoring_dif=False):
        self.seeds = seeds
        self.slots = slots
        self.model = model
        self.include_scoring_dif = include_scoring_dif
        
        games = []
       
        # slots
        round_1_slots = slots[slots["Slot"].str.contains("R1")]
        
        # generate first round games
        for index, slot in round_1_slots.iloc[:32, :].iterrows():
      
            # get seeds
            team_1_seed = slot["Strongseed"]
            team_2_seed = slot["Weakseed"] 
            
            # lookup team id
            team_1 = seeds.loc[seeds["Seed"] == team_1_seed, "Team"].values
            team_2 = seeds.loc[seeds["Seed"] == team_2_seed, "Team"].values
            
            # play in game
            if len(team_1) == 0:
                # get seeds 
                team_1_a_seed = team_1_seed + "a"
                team_1_b_seed = team_1_seed + "b"
                
                # lookup team id
                team_1_a = seeds.loc[seeds["Seed"] == team_1_a_seed, "Team"].values[0]
                team_1_b = seeds.loc[seeds["Seed"] == team_1_b_seed, "Team"].values[0]
                
                # predict winner of play ing
                if include_scoring_dif:
                    team_1, x = self.model.predict(team_1_a, team_1_b)
                else:
                    team_1 = self.model.predict(team_1_a, team_1_b)
            
            # not a play in game
            else:
                # extract value
                team_1 = team_1[0]
             
            # play in game
            if len(team_2) == 0:
                # get seeds 
                team_2_a_seed = team_2_seed + "a"
                team_2_b_seed = team_2_seed + "b"
                
                # lookup team id
                team_2_a = seeds.loc[seeds["Seed"] == team_2_a_seed, "Team"].values[0]
                team_2_b = seeds.loc[seeds["Seed"] == team_2_b_seed, "Team"].values[0]
                
                # predict winner of play in
                if include_scoring_dif:
                    team_2, x = self.model.predict(team_2_a, team_2_b)
                else:
                    team_2 = self.model.predict(team_2_a, team_2_b)
                
            # not a play in game
            else:
                # exrtract value
                team_2 = team_2[0]
                
            # predict winner under our model
            if include_scoring_dif:
                cur_game_pred_team, cur_game_pred_scoring_dif = self.model.predict(team_1, team_2)
            else:
                cur_game_pred_team = self.model.predict(team_1, team_2)
            
            # predict winner seed under our model
            if cur_game_pred_team ==  team_1:
                cur_game_pred_seed = team_1_seed
            else:
                cur_game_pred_seed = team_2_seed

            if self.include_scoring_dif:
                # append games
                games.append((slot["Slot"], 
                              team_1_seed, 
                              team_1, 
                              team_2_seed, 
                              team_2, 
                              cur_game_pred_team, 
                              cur_game_pred_seed,
                              cur_game_pred_scoring_dif))
            else:
                # append games
                games.append((slot["Slot"], 
                              team_1_seed, 
                              team_1, 
                              team_2_seed, 
                              team_2, 
                              cur_game_pred_team, 
                              cur_game_pred_seed))

        # convert to dataframe
        if self.include_scoring_dif:
            self.round_1_df = pd.DataFrame(data=np.array(games), 
                                           columns=["Slot", 
                                                    "Strongseed", 
                                                    "Strongseed Team", 
                                                    "Weakseed", 
                                                    "Weakseed Team", 
                                                    "Prediction", 
                                                    "Prediction Seed",
                                                    "Prediction Scoring Dif"])
        else:
            self.round_1_df = pd.DataFrame(data=np.array(games), 
                                           columns=["Slot", 
                                                    "Strongseed", 
                                                    "Strongseed Team", 
                                                    "Weakseed", 
                                                    "Weakseed Team", 
                                                    "Prediction", 
                                                    "Prediction Seed"])
        
        self.round_2_df = pd.DataFrame()
        self.round_3_df = pd.DataFrame()
        self.round_4_df = pd.DataFrame()
        self.round_5_df = pd.DataFrame()
        self.round_6_df = pd.DataFrame()
        
        # run entire tournament
        self.run_tournament()
            
    # run a particular round
    def generate_round_games(self, round_n):
        games = []
        
        n_games_in_prev_round = {2: 32, 3: 16, 4: 8, 5:4, 6:2}
        
        prev_round_df_dic = {2: self.round_1_df,
                         3: self.round_2_df,
                         4: self.round_3_df,
                         5: self.round_4_df,
                         6: self.round_5_df}
    
        # slots of previous round
        round_n_slots = self.slots[self.slots["Slot"].str.contains("R{}".format(round_n))]
        
        # prev round df
        prev_round_df = prev_round_df_dic.get(round_n)
        
        # generate games in a round
        for index, slot in round_n_slots.iloc[:n_games_in_prev_round.get(round_n), :].iterrows():
            # get seeds
            team_1_seed = slot["Strongseed"]
            team_2_seed = slot["Weakseed"]
            
            # teams
            team_1 = prev_round_df.loc[prev_round_df["Slot"] == team_1_seed, "Prediction"].values[0]
            team_2 = prev_round_df.loc[prev_round_df["Slot"] == team_2_seed, "Prediction"].values[0]

            # predict winner under our model
            if self.include_scoring_dif:
                cur_game_pred_team, cur_game_pred_scoring_dif = self.model.predict(team_1, team_2)
            else:
                cur_game_pred_team = self.model.predict(team_1, team_2)                                                                         
            # predict winner seed under our model
            if int(cur_game_pred_team) == int(team_1):
                cur_game_pred_seed = team_1_seed
            else:
                cur_game_pred_seed = team_2_seed

            # append games, include scoring dif if necessary
            if self.include_scoring_dif:
                games.append((slot["Slot"], 
                              team_1_seed, 
                              team_1, 
                              team_2_seed, 
                              team_2, 
                              cur_game_pred_team, 
                              cur_game_pred_seed,
                              cur_game_pred_scoring_dif))
            else:
                games.append((slot["Slot"], 
                          team_1_seed, 
                          team_1, 
                          team_2_seed, 
                          team_2, 
                          cur_game_pred_team, 
                          cur_game_pred_seed))

        # convert to datafram
        if self.include_scoring_dif:
            cur_round_df = pd.DataFrame(data=np.array(games), 
                                           columns=["Slot", 
                                                    "Strongseed", 
                                                    "Strongseed Team", 
                                                    "Weakseed", 
                                                    "Weakseed Team", 
                                                    "Prediction", 
                                                    "Prediction Seed",
                                                    "Prediction Scoring Dif"])
        else:
            cur_round_df = pd.DataFrame(data=np.array(games), 
                                           columns=["Slot", 
                                                    "Strongseed", 
                                                    "Strongseed Team", 
                                                    "Weakseed", 
                                                    "Weakseed Team", 
                                                    "Prediction", 
                                                    "Prediction Seed"])
        
        if round_n == 2:
            self.round_2_df = cur_round_df
        elif round_n == 3:
            self.round_3_df = cur_round_df
        elif round_n == 4:
            self.round_4_df = cur_round_df
        elif round_n == 5:
            self.round_5_df = cur_round_df
        elif round_n == 6:
            self.round_6_df = cur_round_df  
            
     
    # simulate an entire tournament
    def run_tournament(self):  
        for n in range(2,7):
            self.generate_round_games(n)
            
        self.entire_bracket = pd.concat([self.round_1_df, 
                                              self.round_2_df,
                                              self.round_3_df,
                                              self.round_4_df,
                                              self.round_5_df,
                                              self.round_6_df])
        self.entire_bracket.reset_index(inplace = True, drop=True)
        
    # compare our tournament to other    
    def compare_to_dif_tournament(self, act_results, dif_results, print_res=True, scoring='ESPN'):
        
        # dictionaries
        act_results_dict = {0 : act_results.entire_bracket,
                            1 : act_results.round_1_df,
                            2 : act_results.round_2_df, 
                            3 : act_results.round_3_df, 
                            4 : act_results.round_4_df, 
                            5 : act_results.round_5_df, 
                            6 : act_results.round_6_df}
        
        dif_results_dict = {0 : dif_results.entire_bracket,
                            1 : dif_results.round_1_df,
                            2 : dif_results.round_2_df, 
                            3 : dif_results.round_3_df, 
                            4 : dif_results.round_4_df, 
                            5 : dif_results.round_5_df, 
                            6 : dif_results.round_6_df}
        
        our_results_dict = {0 : self.entire_bracket,
                            1 : self.round_1_df,
                            2 : self.round_2_df, 
                            3 : self.round_3_df, 
                            4 : self.round_4_df, 
                            5 : self.round_5_df, 
                            6 : self.round_6_df}
        
        differences = []
        
        # do it for each round
        for i in range(7):
            dif_results_df = dif_results_dict.get(i)
            our_results_df = our_results_dict.get(i)
            act_results_df = act_results_dict.get(i)
            
            # indexes of different choices
            dif_index = dif_results_df.loc[dif_results_df["Prediction"] != our_results_df["Prediction"]].index
            dif_n = dif_index.shape[0]
            
            # T/F used for sorting
            our_model_correct_tf = our_results_df.loc[dif_index, "Prediction"] == act_results_df.loc[dif_index, "Prediction"]
            dif_model_correct_tf = dif_results_df.loc[dif_index, "Prediction"] == act_results_df.loc[dif_index, "Prediction"]
            
            # get number correct by each model, when there is disagreement
            our_model_correct_n = our_results_df.loc[dif_index].loc[our_model_correct_tf].shape[0]
            dif_model_correct_n = dif_results_df.loc[dif_index].loc[dif_model_correct_tf].shape[0]
            
            # append to our array
            differences.append((dif_n, our_model_correct_n, dif_model_correct_n))
            
            # if we want to print the results
        if print_res:
            print "Number Correct Our Model     : {}, Number Correct Dif Model : {}".format(differences[0][1], differences[0][2])
            print "R1: Number Correct Our Model : {}, Number Correct Dif Model : {}".format(differences[1][1], differences[1][2])
            print "R2: Number Correct Our Model : {}, Number Correct Dif Model : {}".format(differences[2][1], differences[2][2])
            print "R3: Number Correct Our Model : {}, Number Correct Dif Model : {}".format(differences[3][1], differences[3][2])
            print "R4: Number Correct Our Model : {}, Number Correct Dif Model : {}".format(differences[4][1], differences[4][2])
            print "R5: Number Correct Our Model : {}, Number Correct Dif Model : {}".format(differences[5][1], differences[5][2])
            print "R6: Number Correct Our Model : {}, Number Correct Dif Model : {}".format(differences[6][1], differences[6][2])
            
        return differences[0]
        
    # score model vs true results
    def score_tournament(self, actual_results, print_res=False, scoring='ESPN'):
        # extract the df from the actual results tournament
        actual_results_df      = actual_results.entire_bracket
        actual_results_round_1 = actual_results.round_1_df
        actual_results_round_2 = actual_results.round_2_df
        actual_results_round_3 = actual_results.round_3_df
        actual_results_round_4 = actual_results.round_4_df
        actual_results_round_5 = actual_results.round_5_df
        actual_results_round_6 = actual_results.round_6_df
        
        # count correct answers
        tot_correct = actual_results_df[actual_results_df["Prediction"] == self.entire_bracket["Prediction"]].shape[0]
        r_1_correct = actual_results_round_1[actual_results_round_1["Prediction"] == self.round_1_df["Prediction"]].shape[0]
        r_2_correct = actual_results_round_2[actual_results_round_2["Prediction"] == self.round_2_df["Prediction"]].shape[0]
        r_3_correct = actual_results_round_3[actual_results_round_3["Prediction"] == self.round_3_df["Prediction"]].shape[0]
        r_4_correct = actual_results_round_4[actual_results_round_4["Prediction"] == self.round_4_df["Prediction"]].shape[0]
        r_5_correct = actual_results_round_5[actual_results_round_5["Prediction"] == self.round_5_df["Prediction"]].shape[0]  
        r_6_correct = actual_results_round_6[actual_results_round_6["Prediction"] == self.round_6_df["Prediction"]].shape[0]
         
        # total games
        tot_games = actual_results_df.shape[0]
        r_1_games = actual_results_round_1.shape[0]
        r_2_games = actual_results_round_2.shape[0]
        r_3_games = actual_results_round_3.shape[0]
        r_4_games = actual_results_round_4.shape[0]
        r_5_games = actual_results_round_5.shape[0]
        r_6_games = actual_results_round_6.shape[0]
        
        # accuracy
        tot_accuracy = float(tot_correct) / tot_games      
        r_1_accuracy = float(r_1_correct) / r_1_games
        r_2_accuracy = float(r_2_correct) / r_2_games
        r_3_accuracy = float(r_3_correct) / r_3_games
        r_4_accuracy = float(r_4_correct) / r_4_games
        r_5_accuracy = float(r_5_correct) / r_5_games
        r_6_accuracy = float(r_6_correct) / r_6_games
        
        # score depending on scoring system
        if scoring=='ESPN':
            r_1_points = r_1_correct * 10
            r_2_points = r_2_correct * 20
            r_3_points = r_3_correct * 40
            r_4_points = r_4_correct * 80
            r_5_points = r_5_correct * 160
            r_6_points = r_6_correct * 320
        
        tot_points = r_1_points + r_2_points + r_3_points + r_4_points + r_5_points + r_6_points
        
        # if we want to print the results
        if print_res:
            print "Total Points  : {}\n".format(tot_points)
            print "Total Accuracy: {} / {} = {}".format(tot_correct, tot_games, tot_accuracy)
            print "R1    Accuracy: {} / {} = {}".format(r_1_correct, r_1_games, r_1_accuracy)
            print "R2    Accuracy: {} / {} = {}".format(r_2_correct, r_2_games, r_2_accuracy)
            print "R3    Accuracy: {} / {} = {}".format(r_3_correct, r_3_games, r_3_accuracy)
            print "R4    Accuracy: {} / {} = {}".format(r_4_correct, r_4_games, r_4_accuracy)
            print "R5    Accuracy: {} / {} = {}".format(r_5_correct, r_5_games, r_5_accuracy)
            print "R6    Accuracy: {} / {} = {}".format(r_6_correct, r_6_games, r_6_accuracy)

        return (tot_points, tot_accuracy)
    
    # individual team points scored
    def get_predicted_points_for_team(self, team, scoring='ESPN'):
        # counts number of wins in the projected bracket
        team = str(team)
        wins = self.entire_bracket[self.entire_bracket["Prediction"] == team].shape[0]
         
        # TODO: allow for other scoring systems    
        # 10 points for R1, 20 for R2    
        points = 0                           
        for i in range(int(wins)):
            points = points + 10 * 2 ** i
                         
        return points
                                   
    # points scored for all teams
    def get_predicted_points(self, scoring="ESPN"):
        # setup buffers
        teams = self.seeds["Team"]
        points = np.zeros(teams.shape[0])

        i = 0
        for team in teams:
            # get points of team i
            points[i] = self.get_predicted_points_for_team(team)
            i = i + 1
              
        return points
    
########################################################################################################

### Simulator Class
### arguments = seeds, slots, and model
### proceeds to generate the entire tournament based on the model that is passed n times
### calculates expected points scored by each team
### produces final bracket, predicting games based on argmax(E[points_i], E[points_j])

### METHODS
# 1) init --> sets up model, seeds, slots

# 2) simulate_tournament --> takes in n_iterations, predicts bracket (used probability from logistic model to predict each game... sends team 1 over team 2 with some proabbiltu p based on the logistic model of the head to head results of the game) n times, calculates expected score of each team                                                      

# 3) predict_tournament --> deterministically predicts bracket based on expected score of each team

# 4) score_tournament --> compares predicted bracket to the results of the actual tournament
    

# used to simulate a tournament n times and output the combined results
class Simulator(object):
    def __init__(self, seeds, slots, model):
        self.seeds = seeds
        self.slots = slots
        self.model = model
    
    # run the tournament
    def run_tournament(self, scoring="ESPN"):
        self.model.game_count=0
        
        # generate and run a tournament
        tournament = Tournament(self.seeds, self.slots, self.model)
        
        # get the number of predicted points that each team accumulates
        predicted_points = tournament.get_predicted_points(scoring=scoring)
        
        return predicted_points
        
    # simulate the tournament n times
    def simulate_tournament(self, n_iterations, scoring="ESPN"):
        predicted_points = np.zeros(self.seeds.shape[0])
        
        # calculate total points score over the entire simulation
        for i in range(n_iterations):
            predicted_points = predicted_points + self.run_tournament(scoring="ESPN")
            
        # make a dataframe for safe keeping
        self.predicted_points = pd.DataFrame(data=predicted_points, index=self.seeds["Team"], columns=["pred_points"])
        
        return self.predicted_points
    
    # output the final predictions based on the simulation
    def predict_tournament(self):
        # setup model for prediction
        expected_points_model = mmm.ExpectedPointsPredictor(self.predicted_points)
         
        # run tourney using our model as a prediction
        self.tournament_prediction = Tournament(self.seeds, self.slots, expected_points_model)
        self.tournament_prediction.run_tournament()
        
        # return the created tournament object
        return self.tournament_prediction
    
    # score the tournament
    def score_tournament(self, actual_results, print_res=True, scoring="ESPN"):
        return self.tournament_prediction.score_tournament(actual_results, print_res=print_res, scoring=scoring)    

    
    
########################################################################################################

### Ensemble Class
### arguments = seeds, slots, and head to head model
### runs the tournament 10 times with 10 diverse brackets

### METHODS
# 1) init --> sets up model, seeds, slots
#             creates 3 starter tourneys ---> unbiased, late round upsets, early round upsets
#             generates 7 derivated tourneys ---> with cost functions that induce bias based on predictions of 3 starters

# 2) score_tournament ---> gets the max score of the 10 brackets

# 3) compute_dif_matrix ---> computes number of games different between each pair of the 10 brackets

# 4) compute_dif_vect ---> computes dif from an argument bracket for each of the 10 brackets

    
class Ensemble(object):
    def __init__(self, 
                 seeds, 
                 slots, 
                 head_to_head_model, 
                 scaler, 
                 predictor_dfs, 
                 year):
        
        self.seeds = seeds
        self.slots = slots
        self.head_to_head_model = head_to_head_model
        self.scaler = scaler
        self.predictor_dfs = predictor_dfs
        self.year = year
        self.tourney_arr = []
        
        # setup low seed tourney
        top_seed_model = mmm.BasicPredictor()
        top_seed_tourney = Tournament(self.seeds, self.slots, top_seed_model)
        
        # setup unbiased tourney
        unbiased_model = mmm.ModelPredictor(self.head_to_head_model, 
                                            self.scaler, 
                                            self.predictor_dfs, 
                                            self.year, 
                                            self.seeds)
        unbiased_tourney = Tournament(self.seeds, self.slots, unbiased_model)
            
        # setup top_seed tourney dif 1
        top_seed_dif_1_model = mmm.ModelPredictor(self.head_to_head_model, 
                                            self.scaler, 
                                            self.predictor_dfs, 
                                            self.year, 
                                            self.seeds,
                                            other_bracket_arr=[top_seed_tourney.entire_bracket])
        top_seed_dif_1_tourney = Tournament(self.seeds, self.slots, top_seed_dif_1_model)
        
        # setup top_seed tourney dif 2
        top_seed_dif_2_model = mmm.ModelPredictor(self.head_to_head_model, 
                                            self.scaler, 
                                            self.predictor_dfs, 
                                            self.year, 
                                            self.seeds,
                                            other_bracket_arr=[top_seed_tourney.entire_bracket,
                                                           unbiased_tourney.entire_bracket])
        top_seed_dif_2_tourney = Tournament(self.seeds, self.slots, top_seed_dif_2_model)
        
        # setup top_seed tourney dif 3
        top_seed_dif_3_model = mmm.ModelPredictor(self.head_to_head_model, 
                                          self.scaler,
                                          self.predictor_dfs,
                                          self.year, 
                                          self.seeds,
                                          other_bracket_arr=[top_seed_tourney.entire_bracket,
                                                             unbiased_tourney.entire_bracket,
                                                             top_seed_dif_1_tourney.entire_bracket])
        top_seed_dif_3_tourney = Tournament(self.seeds, self.slots, top_seed_dif_3_model)
        
        # setup early round bias
        early_round_bias = {6:0, 5:0, 4:0, 3:10, 2:20, 1:20}
        early_round_bias_model = mmm.ModelPredictor(self.head_to_head_model, 
                                                    self.scaler,
                                                    self.predictor_dfs,
                                                    self.year, 
                                                    self.seeds,
                                                    higher_seed_bias=True,
                                                    higher_seed_bias_delta=.005, 
                                                    cooling = early_round_bias)
        early_round_bias_tourney = Tournament(self.seeds, self.slots, early_round_bias_model)
        
        # setup early round_bias dif 1
        early_round_bias_model_dif_1 = mmm.ModelPredictor(self.head_to_head_model, 
                                            self.scaler,
                                            self.predictor_dfs,
                                            self.year, 
                                            self.seeds,
                                            other_bracket_arr=[early_round_bias_tourney.entire_bracket])
        
        early_round_bias_tourney_dif_1 = Tournament(self.seeds, 
                                                        self.slots, 
                                                        early_round_bias_model_dif_1)
        # setup early round_bias dif 1
        early_round_bias_model_dif_2 = mmm.ModelPredictor(self.head_to_head_model, 
                                    self.scaler,
                                    self.predictor_dfs,
                                    self.year, 
                                    self.seeds,
                                    other_bracket_arr=[early_round_bias_tourney.entire_bracket,
                                                       early_round_bias_tourney_dif_1.entire_bracket])
        
        early_round_bias_tourney_dif_2 = Tournament(self.seeds, 
                                                        self.slots, 
                                                        early_round_bias_model_dif_2)
        
        
        # setup later round bias
        later_round_bias= {6:10, 5:10, 4:20, 3:30, 2:0, 1:0}
        later_round_bias_model = mmm.ModelPredictor(self.head_to_head_model, 
                                                    self.scaler,
                                                    self.predictor_dfs,
                                                    self.year, 
                                                    self.seeds,
                                                    higher_seed_bias=True,
                                                    higher_seed_bias_delta=.005, 
                                                    cooling = later_round_bias)
        later_round_bias_tourney = Tournament(self.seeds, self.slots, later_round_bias_model)
        
        # setup later round_bias dif 1
        later_round_bias_model_dif_1 = mmm.ModelPredictor(self.head_to_head_model, 
                                            self.scaler,
                                            self.predictor_dfs,
                                            self.year, 
                                            self.seeds,
                                            other_bracket_arr=[later_round_bias_tourney.entire_bracket])
        
        later_round_bias_tourney_dif_1 = Tournament(self.seeds, 
                                                        self.slots, 
                                                        later_round_bias_model_dif_1)
        # setup early round_bias dif 1
        later_round_bias_model_dif_2 = mmm.ModelPredictor(self.head_to_head_model, 
                                    self.scaler,
                                    self.predictor_dfs,
                                    self.year, 
                                    self.seeds,
                                    other_bracket_arr=[later_round_bias_tourney.entire_bracket,
                                                       later_round_bias_tourney_dif_1.entire_bracket])
        
        later_round_bias_tourney_dif_2 = Tournament(self.seeds, 
                                                        self.slots, 
                                                        later_round_bias_model_dif_2)
        
        # append to our array of tourneys
        self.tourney_arr = [unbiased_tourney, 
                            top_seed_dif_1_tourney, 
                            top_seed_dif_2_tourney, 
                            top_seed_dif_3_tourney,
                            early_round_bias_tourney,
                            early_round_bias_tourney_dif_1,
                            early_round_bias_tourney_dif_2,
                            later_round_bias_tourney,
                            later_round_bias_tourney_dif_1,
                            later_round_bias_tourney_dif_2]
    
       
    # simulate the tournament n times
    def score_ind_tournament(self, actual_results, index, print_res=False, scoring="ESPN"):
        tourney = self.tourney_arr[index]
        
        return tourney.score_tournament(actual_results, print_res=print_res, scoring=scoring)[0]
    
    # score the tournament
    def score_tournament(self, actual_results, print_res=False, scoring="ESPN"):
        scores = np.zeros(10)
        for i in range(10):
            scores[i] = self.score_ind_tournament(actual_results, i, print_res=print_res, scoring="ESPN")
        return scores
    
    # compute dif matrix
    def compute_dif_matrix(self, actual_results):
        self.dif_matrix = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                tourney_1 = self.tourney_arr[i]
                tourney_2 = self.tourney_arr[j]
                
                self.dif_matrix[i, j] = tourney_1.compare_to_dif_tournament(actual_results, tourney_2, print_res=False)[0]
        return self.dif_matrix
                
    def avg_game_dif(self):
        sum_of_dif_vect = np.sum(self.dif_matrix, axis=0)
        
        avg_vect = sum_of_dif_vect / 9.
        
        return avg_vect
    
    def compute_dif_vect(self, actual_results, other_results):
        self.dif_vect = np.zeros(10)
        for i in range(10):
            self.dif_vect[i], x, y = self.tourney_arr[i].compare_to_dif_tournament(actual_results, other_results, print_res=False)
            
        return self.dif_vect
            
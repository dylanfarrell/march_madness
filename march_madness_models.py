import pandas as pd
import numpy as np
import march_madness_games as mmg
import random as rand

################# MODELS FOR OUR TOURNAMENTS #####################

# predictor high seed
class BasicPredictor(object):
    # init function
    def __init__(self):
        return
    
    # head to head predicitons
    def predict(self, team_1, team_2):
        return team_1

# actual tournament results
class ActualTournament(object):
    # init function
    def __init__(self, data, include_scoring_dif=False):
        self.include_scoring_dif = include_scoring_dif                                            
        self.tourney = data
        return
    
    def predict(self, team_1, team_2):
        game_played_team_1_win = self.tourney[(self.tourney["WTeamID"] == int(team_1)) & (self.tourney["LTeamID"] == int(team_2))]
        game_played_team_2_win = self.tourney[(self.tourney["LTeamID"] == int(team_1)) & (self.tourney["WTeamID"] == int(team_2))]
        
        # extract winner and loser
        if game_played_team_1_win.shape[0] == 1:
            winning_team = team_1
            scoring_dif = game_played_team_1_win["WScore"] - game_played_team_1_win["LScore"]                                
        elif game_played_team_2_win.shape[0] == 1:
            winning_team = team_2
            scoring_dif = game_played_team_2_win["WScore"] - game_played_team_2_win["LScore"]       
        else:
            print("Error")
            return -1
        
        # return socre and scoring dif if we want                                       
        if self.include_scoring_dif:
            return (winning_team, scoring_dif.values[0])
        else:
            return winning_team                                   
                                                    
# predictor using markov chain stationary distribution
class MarkovPredictor(object):
    # init function
    def __init__(self, data):
        self.data = data
        return
    
    # head to head predicitons
    def predict(self, team_1, team_2):
        team_1 = int(team_1)
        team_2 = int(team_2)
        
        # lookup the pi values in the lookup table
        team_1_pi_i = self.data.loc[self.data["TeamID"] == team_1, "pi_i"].values[0]
        team_2_pi_i = self.data.loc[self.data["TeamID"] == team_2, "pi_i"].values[0]
        
        if team_1_pi_i > team_2_pi_i:
            return team_1
        else:
            return team_2

        
# MODEL PREDICTOR ------------------------------------------------------------------------

# however, you are able to do some biasing of the predictions
# higher_seed_bias=False      ----> if True, will predict higher seed (upset) with probability p + higher_seed_bias_delta
# higher_seed_bias_delta=.05  ----> tuned to how much bias we want towards upsets/top seed winning
        
# we are also able to do "cooling" of our model ----> cooling cooresponds to changing the bias depening on the round
# pass in a dict of the form {1:r1, 2:r2, 3:r3, 4:r4, 5:r5, 6:r6}
# when we update bias the probability, we do p + higher_seed_bias_delta * r_i depending on the round

# we are also able to pass in other brackets and induce bias based on the similiarity

# predictor using some model for predicting head to head games
class ModelPredictor(object):
    # init function
    def __init__(self, 
                 model, 
                 scaler, 
                 dfs_arr, 
                 year, 
                 seeds_df,
                 simulation=False,
                 higher_seed_bias=False,
                 higher_seed_bias_delta=.075,
                 cooling=None,
                 other_bracket_arr=[],
                 other_bracket_bias_delta = .1
                 ):
        
        self.model = model
        self.dfs_arr = dfs_arr
        self.year = year
        self.simulation = simulation
        self.scaler = scaler
        self.seeds_df = seeds_df
        self.higher_seed_bias = higher_seed_bias
        self.higher_seed_bias_delta = higher_seed_bias_delta
        self.cooling=cooling
        self.other_bracket_arr = other_bracket_arr
        self.other_bracket_bias_delta = other_bracket_bias_delta
        
        # used to check what round we are in
        self.game_count = 0
        return
    
    # head to head predicitons
    def predict(self, team_1, team_2):
        team_1 = int(team_1)
        team_2 = int(team_2)
        
        # min and max index
        min_index_team = min(team_1, team_2)
        max_index_team = max(team_1, team_2)
                
        # get the x values
        row = mmg.get_predictors_dif(min_index_team, max_index_team, self.year, self.dfs_arr)

        # predict probability team 1 win under model
        p_hat = self.model.predict_proba(self.scaler.transform(row.reshape(1,-1)))[0,1]
                        
        # get the seeds
        team_seeds = self.__get_seeds(min_index_team, max_index_team)
        min_index_seed, min_index_seed_str, max_index_seed, max_index_seed_str = team_seeds
        
        # get the current round of the game
        cur_round = self.__get_cur_round()

        # update the game count
        self.__update_game_count(min_index_seed_str, max_index_seed_str)

        # check if we want to induce upsets, update p_hat
        if self.higher_seed_bias:
            # check if cooling
            if self.cooling is None:
                bias_delta = self.higher_seed_bias_delta
            else:
                # update the bias
                cooling_factor = self.cooling.get(cur_round)
                bias_delta = self.higher_seed_bias_delta * cooling_factor
            
            # adjust our p_hat
            p_hat = self.__bias_p_hat_upset(p_hat, bias_delta, min_index_seed, max_index_seed)
              
        # check if we want to induce difference from other brackets, update p_hat
        if len(self.other_bracket_arr) != 0:
            # adjust our p_hat
            p_hat = self.__bias_p_hat_dif(p_hat, min_index_team, max_index_team)
        
        # make final prediction, determinisitcally or with biased coin flip
        return self.__make_prediction(p_hat, self.simulation, min_index_team, max_index_team)
          
    # gets seeds of team 1 and team 2
    def __get_seeds(self, team_1, team_2):
        # get the seeds to see which team is the underdog
        team_1_seed_str = self.seeds_df.loc[self.seeds_df["TeamID"] == team_1, "Seed"].values[0]
        team_2_seed_str = self.seeds_df.loc[self.seeds_df["TeamID"] == team_2, "Seed"].values[0]

        # convert the seeds to ints for comparieson
        team_1_seed = int(team_1_seed_str[1:3])
        team_2_seed = int(team_2_seed_str[1:3])  
        
        return team_1_seed, team_1_seed_str, team_2_seed, team_2_seed_str
    
    # checks if we have a play in game
    def __check_playin_game(self, team_1_seed_str, team_2_seed_str):
        # confirm not a play in game, iterate
        return len(team_1_seed_str) == 4 and len(team_2_seed_str) == 4
        
    # gets the current round of the game
    def __get_cur_round(self):
        # check which round we are in
        if self.game_count < 32:
            return 1
        elif self.game_count < 32 + 16:
            return 2
        elif self.game_count < 32 + 16 + 8:
            return 3
        elif self.game_count < 32 + 16 + 8 + 4:
            return 4
        elif self.game_count < 32 + 16 + 8 + 4 + 2:
            return 5
        elif self.game_count < 32 + 16 + 8 + 4 + 2 + 1:
            return 6
        else:
            print(self.game_count)
            print("issue with game count")
            return 
     
    # updates game count, if not a playin game
    def __update_game_count(self, min_index_seed_str, max_index_seed_str):
        # check if play in game, iterate game count if so
        if self.__check_playin_game(min_index_seed_str, max_index_seed_str):
            self.game_count = self.game_count
        else: 
            self.game_count = self.game_count + 1
        
     
    # biases the p_hat
    def __bias_p_hat_upset(self, p_hat, bias, min_index_seed, max_index_seed):
         # Update p_hat given the underdog status on one of the teams
        if min_index_seed < max_index_seed:
            # update p_hat to predict max_index more often
            return p_hat - bias

        # if max index team is the lower seed
        elif max_index_seed < min_index_seed:
            # update p_hat to predict min_index more often
            return p_hat + bias
        
        # otherwise just return phat
        else:
            return p_hat
      
    # biases the p_hat
    def __bias_p_hat_dif(self, p_hat, min_index_team, max_index_team):
        # if we care about differentiating from another bracket
        if len(self.other_bracket_arr) != 0:
            # buffers
            other_bracket_min_index_count = 0
            other_bracket_max_index_count = 0
                         
            # count similarities
            for other_bracket in self.other_bracket_arr:
                # predicted team by other bracket
                prediction = other_bracket.iloc[self.game_count - 1]["Prediction"]
                
                # iterate count of similiarity
                if int(prediction) == min_index_team:
                    other_bracket_min_index_count = other_bracket_min_index_count + 1
                elif int(prediction) == max_index_team:
                    other_bracket_max_index_count = other_bracket_max_index_count + 1
              
            # update bias if one of these teams was picked by other brackets
            if other_bracket_min_index_count + other_bracket_max_index_count != 0:
                # min index percent
                percent_min_index = float(other_bracket_min_index_count) / (other_bracket_min_index_count + other_bracket_max_index_count)
                
                # max index percent
                percent_max_index = float(other_bracket_max_index_count) / (other_bracket_min_index_count + other_bracket_max_index_count)
                    
                # if most brackets pick min index
                if percent_max_index < percent_min_index:
                    return p_hat - self.other_bracket_bias_delta
                # if most brackets pick max index, bias probability towards the min index
                else:
                    return p_hat + self.other_bracket_bias_delta  
              
            # otherwise, just use our model
            else:
                return p_hat
                             
        # dont update bias, if we are not checking other brackets
        else:
            return p_hat
                             
               
        
    # makes prediction
    def __make_prediction(self, p_hat, simulation, min_index_team, max_index_team):
        # if simulation, return min_index team with prob p_hat
        if simulation:
            random_unif = rand.uniform(0,1)
     
            # return min_index with probability p_hat
            if random_unif <= p_hat:
                return min_index_team
            else:
                return max_index_team
        
        # if not a simulation, return the prediction of the (possibly biased) model
        else:
            if p_hat > .5:
                return min_index_team
            else:
                return max_index_team    
    

# EXPECTED POINTS PREDICTOR ---------------------------------------------------------------------------------------------------
# predict based on expected points from the simulation   
# looks up the expected number of points 2 teams will score,
# predicts arg_max(E[points_1], E[points_2])

class ExpectedPointsPredictor(object):
    # pass in a dataframe with the expected points of each team from a simulation
    def __init__(self, points_df):
        self.points_df = points_df
        return
    
    # predict based on looking up expected points
    def predict(self, team_1, team_2):
     
        team_1_points = self.points_df.loc[self.points_df.index == int(team_1), "pred_points"].values[0]
        team_2_points = self.points_df.loc[self.points_df.index == int(team_2), "pred_points"].values[0]
        
        # predict max(points 1, points 2)
        if team_1_points > team_2_points:
            return team_1
        else:
            return team_2      
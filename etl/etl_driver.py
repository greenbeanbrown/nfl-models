import pandas as pd
import numpy as np
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, \
                               export_data, load_data, get_data, agg_raw_files, \
                               filter_sort_data, clean_player_data

import warnings
# Suppress the RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, '..')
from columns import PLAYER_INFO_COLS, TEAM_INFO_COLS, LEAGUE_INFO_COLS, GAME_INFO_COLS
from paths import PLAYER_DATA_PATHS, TEAM_DATA_PATHS, LEAGUE_DATA_PATHS, GAME_INFO_PATHS

import player_passing_etl
import player_rushing_etl 
import player_receiving_etl 
#import player_plays_etl
import player_positions_etl
#import player_snaps_etl
#import player_defense_etl
#import player_kicking_etl
 
import team_passing_etl
import team_rushing_etl
import team_receiving_etl
import team_plays_etl
import team_defense_etl

import league_passing_etl
import league_rushing_etl
import league_receiving_etl
import league_plays_etl

import game_info_etl



if __name__ == '__main__':

    stat_types = ['team-passing', 'team-rushing', 'team-plays','team-receiving', 'team-defense',
                  'player-passing','player-rushing','player-receiving','player-positions',
                  'league-passing','league-rushing','league-receiving','league-plays',
                  'game-info']

    # Aggregate raw data
    print('Aggregating raw Player data..')
    agg_raw_files(stat_type='player')
    print('Aggregating raw Team data..')
    agg_raw_files(stat_type='team')
    print('Aggregating raw Game data..')
    agg_raw_files(stat_type='game')

    # Get data for each stat type
    for stat_type in stat_types:
        print(stat_type)
        if 'player' in stat_type:
            data_dir = PLAYER_DATA_PATHS[stat_type]
            info_cols = PLAYER_INFO_COLS
            if stat_type == 'player-positions':
                raw_data = read_raw_data(stat_type=stat_type)
            else:
                # Read raw agg data
                raw_data = read_raw_data(stat_type=stat_type)
                # Get extra data needed
                team_data = get_data(stat_type=stat_type)
                # Merge the team-plays data with the player-level data
                merged_data = pd.merge(raw_data, 
                                       team_data, 
                                       on=TEAM_INFO_COLS)

                if len(merged_data) != max(len(raw_data), len(team_data)):
                    print('WARNING: MERGE MISMATCH FOUND!')
                    import ipdb; ipdb.set_trace()

                # Re-assign raw_data after all this to flow into rest of code
                raw_data = merged_data.copy()

        elif 'team' in stat_type:
            data_dir = TEAM_DATA_PATHS[stat_type]
            info_cols = TEAM_INFO_COLS
            if stat_type == 'team-receiving':
                #raw_data = get_data(stat_type = stat_type)
                raw_data = read_raw_data(stat_type='player-receiving')
                raw_data = rename_raw_cols(raw_data, stat_type=stat_type)
            else:
                raw_data = read_raw_data(stat_type=stat_type)
        elif 'league' in stat_type:
            data_dir = LEAGUE_DATA_PATHS[stat_type]
            info_cols = LEAGUE_INFO_COLS
            raw_data = get_data(stat_type = stat_type)
            raw_data = rename_raw_cols(raw_data, stat_type=stat_type)
        
        elif 'game' in stat_type:
            data_dir = GAME_INFO_PATHS[stat_type]
            info_cols = GAME_INFO_COLS
            raw_data = read_raw_data(stat_type=stat_type)
    
        # Filter data
        filtered_sorted_data = filter_sort_data(raw_data, stat_type=stat_type)

        # Add model features (stat_type specific)
        if stat_type == 'player-passing':
            features_data = player_passing_etl.add_model_features(filtered_sorted_data)
        elif stat_type == 'player-rushing':
            features_data = player_rushing_etl.add_model_features(filtered_sorted_data)
        elif stat_type == 'player-receiving':
            features_data = player_receiving_etl.add_model_features(filtered_sorted_data)
        elif stat_type == 'player-positions':
            features_data = filtered_sorted_data
        
        elif stat_type == 'team-passing':
            features_data = team_passing_etl.add_model_features(filtered_sorted_data)
        elif stat_type == 'team-rushing':
            features_data = team_rushing_etl.add_model_features(filtered_sorted_data)
        elif stat_type == 'team-receiving':
            features_data = team_receiving_etl.add_model_features(filtered_sorted_data)
        elif stat_type == 'team-defense':
            features_data = team_defense_etl.add_model_features(filtered_sorted_data)
        elif stat_type == 'team-plays':
            features_data = team_plays_etl.add_model_features(filtered_sorted_data)

        elif stat_type == 'league-passing':
            features_data = league_passing_etl.add_model_features(filtered_sorted_data)
        elif stat_type == 'league-rushing':
            features_data = league_rushing_etl.add_model_features(filtered_sorted_data)
        elif stat_type == 'league-receiving':
            features_data = league_receiving_etl.add_model_features(filtered_sorted_data)
        elif stat_type == 'league-plays':
            features_data = league_plays_etl.add_model_features(filtered_sorted_data)

        elif stat_type == 'game-info':
            # No features to add here
            features_data = filtered_sorted_data

        # Replace invalid values with NaN
        features_data = features_data.replace([np.inf, -np.inf], np.nan)

        # Split the current and lagged features into 2 tables
        if stat_type in ['player-positions','game-info']:
            player_stat_data = features_data
            lag_player_stat_data = None
        else:
            player_stat_data, lag_player_stat_data = split_lagged_data(features_data, info_cols=info_cols)

        # Export
        export_data(player_stat_data, lag_player_stat_data, data_dir, stat_type=stat_type)
        # Load into DB
        load_data(data_dir, stat_type=stat_type)

        # Begin projections ETL
        #projections_etl()
import pandas as pd
#import numpy as np
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, \
                               export_data, load_data, \
                               filter_sort_data, clean_player_data

sys.path.insert(0, '..')
from columns import PLAYER_DEFENSE_STAT_COLS, PLAYER_INFO_COLS
from paths import PLAYER_DATA_PATHS

def calc_stats(data):
    """
    Performs calculations to create
    certain metrics that aren't given
    """
    # Calculate some specific metrics
    return data


def add_specific_features(player_season_data):
    """
    Adding features that require weighted 
    average calculations 
    """
    data = player_season_data.copy()
    # Capture columns before adding new ones to track for later
    old_cols = data.columns
    # Add some features that require weighted averages
    # (most of these are found on Pro-Football-Reference game logs)
    # Season-long calcs
    # 3 Game calcs
    # 6 Game calcs 
    
    # Create lagged versions of the columns
    new_cols = data.columns
    weighted_cols = []
    lag_cols = [i for i in new_cols if i not in old_cols] + weighted_cols
    # Add each lagged copy of column
    lagged_columns = []
    # Add each lagged copy of column to the list
    for col in lag_cols:
        if 'Season' in col:
            lagged_columns.append(data.groupby('Season')[col].shift(1))
        else:
            lagged_columns.append(data[col].shift(1))

    # Concatenate all lagged columns to the DataFrame
    lagged_data = pd.concat(lagged_columns, axis=1)
    # Rename the new columns to include "Lag" prefix
    lagged_data.columns = ['Lag{}'.format(col) for col in lagged_data.columns]
    # Add the lagged data back into the regular data
    data = pd.concat([data, lagged_data], axis=1)

    return data

def add_model_features(sorted_filtered_data, stat_type):
    """
    Adds modeling features which include
    transformations of the stats columns 
    for each data set
    """
    data = sorted_filtered_data.copy()
    # Add some calculated stats
    data = calc_stats(data)
    # Add QBR and include it in rolling calculations
    stat_cols = PLAYER_DEFENSE_STAT_COLS
    # Isolate data for each player
    player_ids = data['PlayerId'].unique()

    agg_data_list = []

    for player_id in player_ids:
        player_data = data[data['PlayerId'] == player_id].copy()
        # Calculate different features and add to the data
        #player_data = add_specific_features(player_data)
        player_dict = add_rolling_features(player_data, stat_cols)
        # Append current player-season dict to the agg list
        agg_data_list.append(pd.DataFrame(player_dict))  # Create DataFrame from player_dict
    # Create an agg dataframe from the each individual player-season dict
    agg_data = pd.concat(agg_data_list, ignore_index=True)  # Concatenate all player DataFrames
    agg_data.reset_index(inplace=True, drop=True)

    return agg_data

if __name__ == '__main__':

    season = 2021
    stat_type = 'player-defense'
    data_dir = PLAYER_DATA_PATHS[stat_type]
    # Read raw excel file
    raw_data = read_raw_data(data_type='player', year=season)
    # Rename columns to something more manageable
    working_data = rename_raw_cols(raw_data, data_type='player')
    # Do some basic data prep before adding features
    clean_data = clean_player_data(working_data, year=season)
    # Filter data
    raw_player_stat_type_data = filter_sort_data(clean_data, stat_type=stat_type)
    # Add model features
    player_stat_type_data = add_model_features(raw_player_stat_type_data, stat_type=stat_type)
    # Split the current and lagged features into 2 tables
    player_stat_type, lag_player_stat_type = split_lagged_data(player_stat_type_data, info_cols=PLAYER_INFO_COLS)
    # Export
    export_data(player_stat_type, lag_player_stat_type, data_dir, stat_type=stat_type, year=season)
    # Load into DB
    load_data(data_dir, stat_type=stat_type)

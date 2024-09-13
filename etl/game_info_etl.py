import pandas as pd
#import numpy as np
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, agg_raw_files, \
                               export_data, load_data, \
                               filter_sort_data, clean_team_data

sys.path.insert(0, '..')
from columns import GAME_INFO_COLS
from paths import GAME_INFO_PATHS



if __name__ == '__main__':

    stat_type = 'game-info'
    data_dir = GAME_INFO_PATHS[stat_type]
    # Agg the raw data
    agg_raw_files(stat_type=stat_type)
    # Read raw aggregated data
    raw_data = read_raw_data(stat_type=stat_type)
    # Rename columns to something more manageable
    #working_data = rename_raw_cols(raw_data, data_type='team')
    # Do some basic data prep before adding features
    #clean_data = clean_team_data(working_data, year)
    # Filter data
    game_info_data = filter_sort_data(raw_data, stat_type=stat_type)
    # Add model features
    #team_stat_type_data = add_model_features(raw_team_stat_type_data)
    # Split the current and lagged features into 2 tables
    #team_stat_type, lag_team_stat_type = split_lagged_data(team_stat_type_data, info_cols=TEAM_INFO_COLS)
    # Export
    export_data(game_info_data, None, data_dir, stat_type=stat_type)
    # Load into DB
    load_data(data_dir, stat_type=stat_type)
 
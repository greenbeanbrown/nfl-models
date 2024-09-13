import pandas as pd
#import numpy as np
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, agg_raw_files, \
                                 export_data, load_data, \
                                 filter_sort_data, clean_player_data

sys.path.insert(0, '..')
from columns import PLAYER_POSITIONS_COLS
from paths import PLAYER_DATA_PATHS


if __name__ == '__main__':

    stat_type = 'player-positions'
    data_dir = PLAYER_DATA_PATHS[stat_type]
    # Agg the raw data
    agg_raw_files(stat_type=stat_type)
    # Read raw agg data
    raw_data = read_raw_data(stat_type=stat_type)
    # Rename columns to something more manageable
    #working_data = rename_raw_cols(raw_data, data_type='player')
    # Do some basic data prep before adding features
    #clean_data = clean_player_data(working_data)
    # Filter data
    player_stat_type = filter_sort_data(raw_data, stat_type=stat_type)
    # Add model features
    #player_stat_type_data = add_model_features(raw_player_stat_type_data, stat_type=stat_type)
    # Split the current and lagged features into 2 tables
    #player_stat_type, lag_player_stat_type = split_lagged_data(player_stat_type_data, info_cols=PLAYER_INFO_COLS)
    # Export
    export_data(player_stat_type, None, data_dir, stat_type=stat_type)
    # Load into DB
    load_data(data_dir, stat_type=stat_type)

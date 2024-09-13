import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

from datetime import datetime
import os 
import sqlite3
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, \
                               export_data, load_data, agg_raw_files, \
                               filter_sort_data, clean_team_data, get_data

sys.path.insert(0, '..')
from paths import DB_PATH, TEAM_DATA_PATHS
from columns import TEAM_INFO_COLS, TEAM_RECEIVING_COLS, TEAM_RECEIVING_STAT_COLS #, TEAM_RECEIVING_WEIGHTED_COLS

def nfl_connect():
    """
    Connect to the NFL database
    """
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    return conn

def calc_stats(data):
    """
    Performs calculations to create
    certain metrics that aren't given
    """
    # Calculate some specific metrics
    data['TeamYardsPerRec'] = data['TeamRecYards'] / data['TeamReceptions']
    data['TeamYardsPerRecTarget'] = data['TeamRecYards'] / data['TeamRecTargets']
    data['TeamCatchPct'] = data['TeamReceptions'] / data['TeamRecTargets']

    return data

def add_specific_features(league_season_data):
    """
    Adding features that require weighted 
    average calculations 
    """
    data = league_season_data.copy()
    # Capture columns before adding new ones to track for later
    old_cols = data.columns
    # Add some features that require weighted averages
    # (most of these are found on Pro-Football-Reference game logs)
    # Career-long calcs
    data.sort_values(['TeamAbbr','Season'], inplace=True)
    data['CareerTeamYardsPerRec'] = data.groupby('TeamAbbr')['TeamRecYards'].expanding().sum().values / data.groupby('TeamAbbr')['TeamReceptions'].expanding().sum().values
    data['CareerTeamYardsPerRecTarget'] = data.groupby('TeamAbbr')['TeamRecYards'].expanding().sum().values / data.groupby('TeamAbbr')['TeamRecTargets'].expanding().sum().values
    data['CareerTeamCatchPct'] = data.groupby('TeamAbbr')['TeamReceptions'].expanding().sum().values / data.groupby('TeamAbbr')['TeamRecTargets'].expanding().sum().values

    # 3 Game calcs
    data.sort_values(['TeamAbbr','Season'], inplace=True)
    data['3GameTeamYardsPerRec'] = data.groupby('TeamAbbr')['TeamRecYards'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamReceptions'].rolling(3, min_periods=0).sum().values
    data['3GameTeamYardsPerRecTarget'] = data.groupby('TeamAbbr')['TeamRecYards'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamRecTargets'].rolling(3, min_periods=0).sum().values
    data['3GameTeamCatchPct'] = data.groupby('TeamAbbr')['TeamReceptions'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamRecTargets'].rolling(3, min_periods=0).sum().values
    # 6 Game calcs 
    data['6GameTeamYardsPerRec'] = data.groupby('TeamAbbr')['TeamRecYards'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamReceptions'].rolling(6, min_periods=0).sum().values
    data['6GameTeamYardsPerRecTarget'] = data.groupby('TeamAbbr')['TeamRecYards'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamRecTargets'].rolling(6, min_periods=0).sum().values
    data['6GameTeamCatchPct'] = data.groupby('TeamAbbr')['TeamReceptions'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamRecTargets'].rolling(6, min_periods=0).sum().values

    # Season-long calcs
    data.sort_values(['Season','TeamAbbr'], inplace=True)
    data['SeasonTeamYardsPerRec'] = data.groupby(['Season','TeamAbbr'])['TeamRecYards'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['TeamReceptions'].expanding().sum().values
    data['SeasonTeamYardsPerRecTarget'] = data.groupby(['Season','TeamAbbr'])['TeamRecYards'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['TeamRecTargets'].expanding().sum().values
    data['SeasonTeamCatchPct']  = data.groupby(['Season','TeamAbbr'])['TeamReceptions'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['TeamRecTargets'].expanding().sum().values
    # Season 3 Game calcs
    data['Season3GameTeamYardsPerRec'] = data.groupby(['Season','TeamAbbr'])['TeamRecYards'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamReceptions'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamYardsPerRecTarget'] = data.groupby(['Season','TeamAbbr'])['TeamRecYards'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamRecTargets'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamCatchPct'] = data.groupby(['Season','TeamAbbr'])['TeamReceptions'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamRecTargets'].rolling(3, min_periods=0).sum().values
    # Season 6 Game calcs 
    data['Season6GameTeamYardsPerRec'] = data.groupby(['Season','TeamAbbr'])['TeamRecYards'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamReceptions'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamYardsPerRecTarget'] = data.groupby(['Season','TeamAbbr'])['TeamRecYards'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamRecTargets'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamCatchPct'] = data.groupby(['Season','TeamAbbr'])['TeamReceptions'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamRecTargets'].rolling(6, min_periods=0).sum().values

    # Create lagged versions of the columns
    new_cols = data.columns
    weighted_cols = ['TeamYardsPerRec','TeamYardsPerRecTarget','TeamCatchPct']
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

def add_model_features(sorted_filtered_data):
    """
    Adds modeling features which include
    transformations of the stats columns 
    for each data set
    """

    data = sorted_filtered_data.copy()
    stat_cols = TEAM_RECEIVING_STAT_COLS

    agg_data_list = []
    # Aggregate the player-level data into team-level
    # Get sums for each game date
    team_data = sorted_filtered_data.groupby(TEAM_INFO_COLS).sum()
    team_data.reset_index(inplace=True)
     # Calculate stats
    team_data = calc_stats(team_data)
    # Add season and weighted average features
    team_data = add_specific_features(team_data)
    # Add rolling features
    team_dict = add_rolling_features(team_data, stat_cols, league=False)
    team_data = pd.DataFrame(team_dict)

    return team_data

if __name__ == '__main__':

    stat_type = 'team-receiving'
    data_dir = TEAM_DATA_PATHS[stat_type]
    # Get team data 
    #raw_data = get_data(stat_type = stat_type)
    
    # Agg the raw data
    #agg_raw_files(stat_type=stat_type)
    
     # Read raw excel file
     # We trick the code here to use player-receiving because team-level data doesn't have receiving data
    raw_data = read_raw_data(stat_type='player-receiving')

    # Rename columns from Team to League
    working_data = rename_raw_cols(raw_data, stat_type=stat_type)
    # Filter data
    raw_team_stat_type_data = filter_sort_data(working_data, stat_type=stat_type)
    # Add model features
    team_stat_type_data = add_model_features(raw_team_stat_type_data)
    # Split the current and lagged features into 2 tables
    team_stat_type_data, lag_team_stat_type_data = split_lagged_data(team_stat_type_data, info_cols=TEAM_INFO_COLS)
    # Export
    export_data(team_stat_type_data, lag_team_stat_type_data, data_dir, stat_type=stat_type)
    # Load into DB
    load_data(data_dir, stat_type=stat_type)

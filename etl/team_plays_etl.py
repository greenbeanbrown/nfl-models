import pandas as pd
#import numpy as np
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, agg_raw_files, \
                               export_data, load_data, \
                               filter_sort_data, clean_team_data

sys.path.insert(0, '..')
from columns import TEAM_PLAYS_STAT_COLS, TEAM_PLAYS_COLS, TEAM_INFO_COLS, TEAM_PLAYS_EXTRA_COLS
from paths import TEAM_DATA_PATHS

def calc_stats(data):
    """
    Performs calculations to create
    certain metrics that aren't given
    """
    # Calculate some specific metrics
    data['TeamRushPlayPct'] = data['TeamRushAtt'] / data['TeamTotalPlays']
    data['TeamPassPlayPct'] = data['TeamPassAtt'] / data['TeamTotalPlays']
    data['Team3rdDownPct']  = data['Team3rdDownMade'] / data['Team3rdDownAtt']
    data['Team4thDownPct']  = data['Team4thDownMade'] / data['Team4thDownAtt']

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
    # Career-long calcs
    data.sort_values(['TeamAbbr','Season'], inplace=True)
    data['CareerTeamRushPlayPct'] = data.groupby('TeamAbbr')['TeamRushAtt'].expanding().sum().values / data.groupby('TeamAbbr')['TeamTotalPlays'].expanding().sum().values
    data['CareerTeamPassPlayPct'] = data.groupby('TeamAbbr')['TeamPassAtt'].expanding().sum().values / data.groupby('TeamAbbr')['TeamTotalPlays'].expanding().sum().values
    data['CareerTeam3rdDownPct'] = data.groupby('TeamAbbr')['Team3rdDownMade'].expanding().sum().values / data.groupby('TeamAbbr')['Team3rdDownAtt'].expanding().sum().values
    data['CareerTeam4thDownPct'] = data.groupby('TeamAbbr')['Team4thDownMade'].expanding().sum().values / data.groupby('TeamAbbr')['Team4thDownAtt'].expanding().sum().values
    # 3 Game calcs
    data.sort_values(['TeamAbbr','Season'], inplace=True)
    data['3GameTeamRushPlayPct'] = data.groupby('TeamAbbr')['TeamRushAtt'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['3GameTeamPassPlayPct'] = data.groupby('TeamAbbr')['TeamPassAtt'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['3GameTeam3rdDownPct'] = data.groupby('TeamAbbr')['Team3rdDownMade'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['Team3rdDownAtt'].rolling(3, min_periods=0).sum().values
    data['3GameTeam4thDownPct'] = data.groupby('TeamAbbr')['Team4thDownMade'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['Team4thDownAtt'].rolling(3, min_periods=0).sum().values
    # 6 Game calcs 
    data['6GameTeamRushPlayPct'] = data.groupby('TeamAbbr')['TeamRushAtt'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['6GameTeamPassPlayPct'] = data.groupby('TeamAbbr')['TeamPassAtt'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['6GameTeam3rdDownPct'] = data.groupby('TeamAbbr')['Team3rdDownMade'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['Team3rdDownAtt'].rolling(6, min_periods=0).sum().values
    data['6GameTeam4thDownPct'] = data.groupby('TeamAbbr')['Team4thDownMade'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['Team4thDownAtt'].rolling(6, min_periods=0).sum().values
    
    # Season-long calcs
    data.sort_values(['Season','TeamAbbr'], inplace=True)
    data['SeasonTeamRushPlayPct'] = data.groupby(['Season','TeamAbbr'])['TeamRushAtt'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['TeamTotalPlays'].expanding().sum().values
    data['SeasonTeamPassPlayPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['TeamTotalPlays'].expanding().sum().values
    data['SeasonTeam3rdDownPct'] = data.groupby(['Season','TeamAbbr'])['Team3rdDownMade'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['Team3rdDownAtt'].expanding().sum().values
    data['SeasonTeam4thDownPct'] = data.groupby(['Season','TeamAbbr'])['Team4thDownMade'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['Team4thDownAtt'].expanding().sum().values
    # Season 3 Game calcs
    data.sort_values(['TeamAbbr','Season'], inplace=True)
    data['Season3GameTeamRushPlayPct'] = data.groupby(['Season','TeamAbbr'])['TeamRushAtt'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamPassPlayPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeam3rdDownPct'] = data.groupby(['Season','TeamAbbr'])['Team3rdDownMade'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['Team3rdDownAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeam4thDownPct'] = data.groupby(['Season','TeamAbbr'])['Team4thDownMade'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['Team4thDownAtt'].rolling(3, min_periods=0).sum().values
    # Season 6 Game calcs 
    data['Season6GameTeamRushPlayPct'] = data.groupby(['Season','TeamAbbr'])['TeamRushAtt'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamPassPlayPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeam3rdDownPct'] = data.groupby(['Season','TeamAbbr'])['Team3rdDownMade'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['Team3rdDownAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeam4thDownPct'] = data.groupby(['Season','TeamAbbr'])['Team4thDownMade'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['Team4thDownAtt'].rolling(6, min_periods=0).sum().values


    # Create lagged versions of the columns
    new_cols = data.columns
    # Add lagged features for the other weighted columns too
    weighted_cols = ['TeamRushPlayPct','TeamPassPlayPct','Team3rdDownPct','Team4thDownPct']
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
    # Add some calculated stats
    data = calc_stats(data)
    # Flip opp stats to current team allowed
    #data = create_allowed_data(data)
    # Add QBR and include it in rolling calculations
    stat_cols = TEAM_PLAYS_STAT_COLS
    # Isolate data for each player
    teams = data['TeamAbbr'].unique()

    agg_data_list = []

    for team in teams:
        team_data = data[data['TeamAbbr'] == team].copy()
        # Calculate different features and add to the data
        team_data = add_specific_features(team_data)
        team_dict = add_rolling_features(team_data, stat_cols)
        # Append current player-season dict to the agg list
        agg_data_list.append(pd.DataFrame(team_dict))  # Create DataFrame from player_dict
    # Create an agg dataframe from the each individual player-season dict
    agg_data = pd.concat(agg_data_list, ignore_index=True)  # Concatenate all player DataFrames
    agg_data.reset_index(inplace=True, drop=True)

    # **** team-plays specific 
    # Remove extra columns that were needed for other calculations 
    for token in TEAM_PLAYS_EXTRA_COLS:
        extra_cols = [i for i in agg_data.columns if token in i]
        agg_data.drop(extra_cols, inplace=True, axis=1)

    return agg_data

if __name__ == '__main__':

    stat_type = 'team-plays'
    data_dir = TEAM_DATA_PATHS[stat_type]
    # Agg the raw data
    agg_raw_files(stat_type=stat_type)
    # Read raw excel file
    raw_data = read_raw_data(stat_type=stat_type)
    # Rename columns to something more manageable
    #working_data = rename_raw_cols(raw_data, data_type='team')
    # Do some basic data prep before adding features
    #clean_data = clean_team_data(working_data)
    # Filter data
    raw_team_stat_type_data = filter_sort_data(raw_data, stat_type=stat_type)
    # Add model features
    team_stat_type_data = add_model_features(raw_team_stat_type_data)
    # Split the current and lagged features into 2 tables
    team_stat_type, lag_team_stat_type = split_lagged_data(team_stat_type_data, info_cols=TEAM_INFO_COLS)
    # Export
    export_data(team_stat_type, lag_team_stat_type, data_dir, stat_type=stat_type)
    # Load into DB
    load_data(data_dir, stat_type=stat_type)

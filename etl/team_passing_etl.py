import pandas as pd
#import numpy as np
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, \
                               export_data, load_data, \
                               filter_sort_data, clean_team_data

sys.path.insert(0, '..')
from columns import TEAM_PASSING_STAT_COLS, TEAM_PASSING_WEIGHTED_COLS, TEAM_INFO_COLS
from paths import TEAM_DATA_PATHS

def calc_stats(data):
    """
    Performs calculations to create
    certain metrics that aren't given
    """
    # Calculate some specific metrics
    data['TeamPassCompPct'] = data['TeamPassComp'] / data['TeamPassAtt']
    data['TeamPassIntPct'] = data['TeamPassInt'] / data['TeamPassAtt']
    data['TeamPassTdPct'] = data['TeamPassTd'] / data['TeamPassAtt']
    data['TeamPassYardsPerComp'] = data['TeamPassYards'] / data['TeamPassComp']
    data['TeamPassYardsPerAtt'] = data['TeamPassYards'] / data['TeamPassAtt']
    data['TeamAdjPassYardsPerAtt'] = (data['TeamPassYards'] + 20 * data['TeamPassTd'] - 45 * data['TeamPassInt']) / data['TeamPassAtt']
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
    data['CareerTeamPassCompPct'] = data.groupby('TeamAbbr')['TeamPassComp'].expanding().sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].expanding().sum().values
    data['CareerTeamPassIntPct'] = data.groupby('TeamAbbr')['TeamPassInt'].expanding().sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].expanding().sum().values
    data['CareerTeamPassTdPct'] = data.groupby('TeamAbbr')['TeamPassTd'].expanding().sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].expanding().sum().values
    data['CareerTeamPassYardsPerComp'] = data.groupby('TeamAbbr')['TeamPassYards'].expanding().sum().values / data.groupby('TeamAbbr')['TeamPassComp'].expanding().sum().values
    data['CareerTeamPassYardsPerAtt'] = data.groupby('TeamAbbr')['TeamPassYards'].expanding().sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].expanding().sum().values
    data['CareerTeamAdjPassYardsPerAtt'] = (data.groupby('TeamAbbr')['TeamPassYards'].expanding().sum().values + 20 * data.groupby('TeamAbbr')['TeamPassTd'].expanding().sum().values - 45 * data.groupby('TeamAbbr')['TeamPassInt'].expanding().sum().values) / data.groupby('TeamAbbr')['TeamPassAtt'].expanding().sum().values
    # 3 Game calcs
    data.sort_values(['TeamAbbr','Season'], inplace=True)
    data['3GameTeamPassCompPct'] = data.groupby('TeamAbbr')['TeamPassComp'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    data['3GameTeamPassIntPct'] = data.groupby('TeamAbbr')['TeamPassInt'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    data['3GameTeamPassTdPct'] = data.groupby('TeamAbbr')['TeamPassTd'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    data['3GameTeamPassYardsPerComp'] = data.groupby('TeamAbbr')['TeamPassYards'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamPassComp'].rolling(3, min_periods=0).sum().values
    data['3GameTeamPassYardsPerAtt'] = data.groupby('TeamAbbr')['TeamPassYards'].rolling(3, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    data['3GameTeamAdjPassYardsPerAtt'] = (data.groupby('TeamAbbr')['TeamPassYards'].rolling(3, min_periods=0).sum().values + 20 * data.groupby('TeamAbbr')['TeamPassTd'].rolling(3, min_periods=0).sum().values - 45 * data.groupby('TeamAbbr')['TeamPassInt'].rolling(3, min_periods=0).sum().values) / data.groupby('TeamAbbr')['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    # 6 Game calcs 
    data['6GameTeamPassCompPct'] = data.groupby('TeamAbbr')['TeamPassComp'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].rolling(6, min_periods=0).sum().values
    data['6GameTeamPassIntPct'] = data.groupby('TeamAbbr')['TeamPassInt'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].rolling(6, min_periods=0).sum().values
    data['6GameTeamPassTdPct'] = data.groupby('TeamAbbr')['TeamPassTd'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].rolling(6, min_periods=0).sum().values
    data['6GameTeamPassYardsPerComp'] = data.groupby('TeamAbbr')['TeamPassYards'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamPassComp'].rolling(6, min_periods=0).sum().values
    data['6GameTeamPassYardsPerAtt'] = data.groupby('TeamAbbr')['TeamPassYards'].rolling(6, min_periods=0).sum().values / data.groupby('TeamAbbr')['TeamPassAtt'].rolling(6, min_periods=0).sum().values
    data['6GameTeamAdjPassYardsPerAtt'] = (data.groupby('TeamAbbr')['TeamPassYards'].rolling(6, min_periods=0).sum().values + 20 * data.groupby('TeamAbbr')['TeamPassTd'].rolling(6, min_periods=0).sum().values - 45 * data.groupby('TeamAbbr')['TeamPassInt'].rolling(6, min_periods=0).sum().values) / data.groupby('TeamAbbr')['TeamPassAtt'].rolling(6, min_periods=0).sum().values

    # Season-long calcs
    data.sort_values(['Season','TeamAbbr'], inplace=True)
    data['SeasonTeamPassCompPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassComp'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].expanding().sum().values
    data['SeasonTeamPassIntPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassInt'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].expanding().sum().values
    data['SeasonTeamPassTdPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassTd'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].expanding().sum().values
    data['SeasonTeamPassYardsPerComp'] = data.groupby(['Season','TeamAbbr'])['TeamPassYards'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassComp'].expanding().sum().values
    data['SeasonTeamPassYardsPerAtt'] = data.groupby(['Season','TeamAbbr'])['TeamPassYards'].expanding().sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].expanding().sum().values
    data['SeasonTeamAdjPassYardsPerAtt'] = (data.groupby(['Season','TeamAbbr'])['TeamPassYards'].expanding().sum().values + 20 * data.groupby(['Season','TeamAbbr'])['TeamPassTd'].expanding().sum().values - 45 * data.groupby(['Season','TeamAbbr'])['TeamPassInt'].expanding().sum().values) /  data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].expanding().sum().values
    # Season 3 Game calcs
    data.sort_values(['TeamAbbr','Season'], inplace=True)
    data['Season3GameTeamPassCompPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassComp'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamPassIntPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassInt'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamPassTdPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassTd'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamPassYardsPerComp'] = data.groupby(['Season','TeamAbbr'])['TeamPassYards'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassComp'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamPassYardsPerAtt'] = data.groupby(['Season','TeamAbbr'])['TeamPassYards'].rolling(3, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamAdjPassYardsPerAtt'] = (data.groupby(['Season','TeamAbbr'])['TeamPassYards'].rolling(3, min_periods=0).sum().values + 20 * data.groupby(['Season','TeamAbbr'])['TeamPassTd'].rolling(3, min_periods=0).sum().values - 45 * data.groupby(['Season','TeamAbbr'])['TeamPassInt'].rolling(3, min_periods=0).sum().values) / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    # Season 6 Game calcs 
    data['Season6GameTeamPassCompPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassComp'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamPassIntPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassInt'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamPassTdPct'] = data.groupby(['Season','TeamAbbr'])['TeamPassTd'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamPassYardsPerComp'] = data.groupby(['Season','TeamAbbr'])['TeamPassYards'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassComp'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamPassYardsPerAtt'] = data.groupby(['Season','TeamAbbr'])['TeamPassYards'].rolling(6, min_periods=0).sum().values / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamAdjPassYardsPerAtt'] = (data.groupby(['Season','TeamAbbr'])['TeamPassYards'].rolling(6, min_periods=0).sum().values + 20 * data.groupby(['Season','TeamAbbr'])['TeamPassTd'].rolling(6, min_periods=0).sum().values - 45 * data.groupby(['Season','TeamAbbr'])['TeamPassInt'].rolling(6, min_periods=0).sum().values) / data.groupby(['Season','TeamAbbr'])['TeamPassAtt'].rolling(6, min_periods=0).sum().values

    # Create lagged versions of the columns
    new_cols = data.columns
    weighted_cols = ['TeamPassCompPct','TeamPassIntPct','TeamPassTdPct','TeamPassYardsPerComp','TeamPassYardsPerAtt','TeamAdjPassYardsPerAtt']
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
    # Add QBR and include it in rolling calculations
    stat_cols = TEAM_PASSING_STAT_COLS
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

    return agg_data

if __name__ == '__main__':

    stat_type = 'team-passing'
    data_dir = TEAM_DATA_PATHS[stat_type]
    # Read raw excel file
    raw_data = read_raw_data(stat_type=stat_type)
    # Rename columns to something more manageable
    #working_data = rename_raw_cols(raw_data, data_type='team')
    # Do some basic data prep before adding features
    #clean_data = clean_team_data(working_data, year=season)
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

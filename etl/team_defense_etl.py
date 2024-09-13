import pandas as pd
#import numpy as np
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, \
                               export_data, load_data, \
                               filter_sort_data, clean_team_data

sys.path.insert(0, '..')
from columns import TEAM_DEFENSE_ALLOWED_STAT_COLS, TEAM_STAT_COLS, TEAM_INFO_COLS
from paths import TEAM_DATA_PATHS


def create_allowed_data(team_data):
    """
    Parses through the team data to 
    derive the opponents offensive data
    (AKA team defense allowed data)
    """
    # Capture just the info data
    info_data = team_data[TEAM_INFO_COLS].copy()
    # Sort team data to make sure the order is team, opp
    team_data.sort_values('GameId', inplace=True)
    team_data.reset_index(inplace=True, drop=True)
    calced_cols = ['TeamPassCompPct','TeamPassIntPct','TeamPassTdPct','TeamPassYardsPerComp','TeamPassYardsPerAtt',
                   'TeamAdjPassYardsPerAtt', 'TeamYardsPerRush','TeamRushTdPct']
    allowed_cols = TEAM_STAT_COLS + calced_cols
    allowed_data = team_data[allowed_cols].copy()
    
    # Flip team and opp stats
    dfs = []
    for idx in allowed_data.index:
        if idx % 2 == 0:
            df = pd.DataFrame(allowed_data.iloc[idx+1]).T
            dfs.append(df)
        else:
            df = pd.DataFrame(allowed_data.iloc[idx-1]).T
            dfs.append(df)
    
    # Combine all of the new dfs and reset index to allow easy joining
    allowed_data = pd.concat(dfs)
    allowed_data.reset_index(inplace=True, drop=True)
    # Update column names 
    allowed_data.columns = [i + 'Allowed' for i in allowed_data.columns]
    # Update the team_data with updated columns
    team_allowed_data = team_data.copy()
    team_allowed_data[allowed_data.columns] = allowed_data
    # Drop the old columns
    team_allowed_data.drop(TEAM_STAT_COLS + calced_cols, inplace=True, axis=1)
    # Sort before output
    team_allowed_data.sort_values(['TeamAbbr','GameDate'])
    
    return team_allowed_data

def calc_stats(data):
    """
    Performs calculations to create
    certain metrics that aren't given
    """
    # Calculate some specific metrics
    # PASS
    data['TeamPassCompPct'] = data['TeamPassComp'] / data['TeamPassAtt']
    data['TeamPassIntPct'] = data['TeamPassInt'] / data['TeamPassAtt']
    data['TeamPassTdPct'] = data['TeamPassTd'] / data['TeamPassAtt']
    data['TeamPassYardsPerComp'] = data['TeamPassYards'] / data['TeamPassComp']
    data['TeamPassYardsPerAtt'] = data['TeamPassYards'] / data['TeamPassAtt']
    data['TeamAdjPassYardsPerAtt'] = (data['TeamPassYards'] + 20 * data['TeamPassTd'] - 45 * data['TeamPassInt']) / data['TeamPassAtt']
    # RUSH
    data['TeamYardsPerRush'] = data['TeamRushYards'] / data['TeamRushAtt']
    data['TeamRushTdPct'] = data['TeamRushTd'] / data['TeamRushAtt']

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
    data['CareerTeamPassCompPctAllowed'] = data['TeamPassCompAllowed'].expanding().sum() / data['TeamPassAttAllowed'].expanding().sum()
    data['CareerTeamPassIntPctAllowed'] = data['TeamPassIntAllowed'].expanding().sum() / data['TeamPassAttAllowed'].expanding().sum()
    data['CareerTeamPassTdPctAllowed'] = data['TeamPassTdAllowed'].expanding().sum() / data['TeamPassAttAllowed'].expanding().sum()
    data['CareerTeamPassYardsPerCompAllowed'] = data['TeamPassYardsAllowed'].expanding().sum() / data['TeamPassCompAllowed'].expanding().sum()
    data['CareerTeamPassYardsPerAttAllowed'] = data['TeamPassYardsAllowed'].expanding().sum() / data['TeamPassAttAllowed'].expanding().sum()
    data['CareerTeamAdjPassYardsPerAttAllowed'] = (data['TeamPassYardsAllowed'].expanding().sum() + 20 * data['TeamPassTdAllowed'].expanding().sum() - 45 * data['TeamPassIntAllowed'].expanding().sum()) / data['TeamPassAttAllowed'].expanding().sum()
    data['CareerTeamYardsPerRushAllowed'] = data['TeamRushYardsAllowed'].expanding().sum() / data['TeamRushAttAllowed'].expanding().sum()
    data['CareerTeamRushTdPctAllowed'] = data['TeamRushTdAllowed'].expanding().sum() / data['TeamRushAttAllowed'].expanding().sum()
    # 3 Game calcs
    data['3GameTeamPassCompPctAllowed'] = data['TeamPassCompAllowed'].rolling(3, min_periods=0).sum() / data['TeamPassAttAllowed'].rolling(3, min_periods=0).sum()
    data['3GameTeamPassIntPctAllowed'] = data['TeamPassIntAllowed'].rolling(3, min_periods=0).sum() / data['TeamPassAttAllowed'].rolling(3, min_periods=0).sum()
    data['3GameTeamPassTdPctAllowed'] = data['TeamPassTdAllowed'].rolling(3, min_periods=0).sum() / data['TeamPassAttAllowed'].rolling(3, min_periods=0).sum()
    data['3GameTeamPassYardsPerCompAllowed'] = data['TeamPassYardsAllowed'].rolling(3, min_periods=0).sum() / data['TeamPassCompAllowed'].rolling(3, min_periods=0).sum()
    data['3GameTeamPassYardsPerAttAllowed'] = data['TeamPassYardsAllowed'].rolling(3, min_periods=0).sum() / data['TeamPassAttAllowed'].rolling(3, min_periods=0).sum()
    data['3GameTeamAdjPassYardsPerAttAllowed'] = (data['TeamPassYardsAllowed'].rolling(3, min_periods=0).sum() + 20 * data['TeamPassTdAllowed'].rolling(3, min_periods=0).sum() - 45 * data['TeamPassIntAllowed'].rolling(3, min_periods=0).sum()) / data['TeamPassAttAllowed'].rolling(3, min_periods=0).sum()
    data['3GameTeamYardsPerRushAllowed'] = data['TeamRushYardsAllowed'].rolling(3, min_periods=0).sum() / data['TeamRushAttAllowed'].rolling(3, min_periods=0).sum()
    data['3GameTeamRushTdPctAllowed'] = data['TeamRushTdAllowed'].rolling(3, min_periods=0).sum() / data['TeamRushAttAllowed'].rolling(3, min_periods=0).sum()
    # 6 Game calcs 
    data['6GameTeamPassCompPctAllowed'] = data['TeamPassCompAllowed'].rolling(6, min_periods=0).sum() / data['TeamPassAttAllowed'].rolling(6, min_periods=0).sum()
    data['6GameTeamPassIntPctAllowed'] = data['TeamPassIntAllowed'].rolling(6, min_periods=0).sum() / data['TeamPassAttAllowed'].rolling(6, min_periods=0).sum()
    data['6GameTeamPassTdPctAllowed'] = data['TeamPassTdAllowed'].rolling(6, min_periods=0).sum() / data['TeamPassAttAllowed'].rolling(6, min_periods=0).sum()
    data['6GameTeamPassYardsPerCompAllowed'] = data['TeamPassYardsAllowed'].rolling(6, min_periods=0).sum() / data['TeamPassCompAllowed'].rolling(6, min_periods=0).sum()
    data['6GameTeamPassYardsPerAttAllowed'] = data['TeamPassYardsAllowed'].rolling(6, min_periods=0).sum() / data['TeamPassAttAllowed'].rolling(6, min_periods=0).sum()
    data['6GameTeamAdjPassYardsPerAttAllowed'] = (data['TeamPassYardsAllowed'].rolling(6, min_periods=0).sum() + 20 * data['TeamPassTdAllowed'].rolling(3, min_periods=0).sum() - 45 * data['TeamPassIntAllowed'].rolling(6, min_periods=0).sum()) / data['TeamPassAttAllowed'].rolling(6, min_periods=0).sum()
    data['6GameTeamYardsPerRushAllowed'] = data['TeamRushYardsAllowed'].rolling(6, min_periods=0).sum() / data['TeamRushAttAllowed'].rolling(6, min_periods=0).sum()
    data['6GameTeamRushTdPctAllowed'] = data['TeamRushTdAllowed'].rolling(6, min_periods=0).sum() / data['TeamRushAttAllowed'].rolling(6, min_periods=0).sum()
    # Season-long calcs
    data['SeasonTeamPassCompPctAllowed'] = data.groupby('Season')['TeamPassCompAllowed'].expanding().sum().values / data.groupby('Season')['TeamPassAttAllowed'].expanding().sum().values
    data['SeasonTeamPassIntPctAllowed'] = data.groupby('Season')['TeamPassIntAllowed'].expanding().sum().values / data.groupby('Season')['TeamPassAttAllowed'].expanding().sum().values
    data['SeasonTeamPassTdPctAllowed'] = data.groupby('Season')['TeamPassTdAllowed'].expanding().sum().values / data.groupby('Season')['TeamPassAttAllowed'].expanding().sum().values
    data['SeasonTeamPassYardsPerCompAllowed'] = data.groupby('Season')['TeamPassYardsAllowed'].expanding().sum().values / data.groupby('Season')['TeamPassCompAllowed'].expanding().sum().values
    data['SeasonTeamPassYardsPerAttAllowed'] = data.groupby('Season')['TeamPassYardsAllowed'].expanding().sum().values / data.groupby('Season')['TeamPassAttAllowed'].expanding().sum().values
    data['SeasonTeamAdjPassYardsPerAttAllowed'] = (data.groupby('Season')['TeamPassYardsAllowed'].expanding().sum().values + 20 * data.groupby('Season')['TeamPassTdAllowed'].expanding().sum().values - 45 * data.groupby('Season')['TeamPassIntAllowed'].expanding().sum().values) / data.groupby('Season')['TeamPassAttAllowed'].expanding().sum().values
    data['SeasonTeamYardsPerRushAllowed'] = data.groupby('Season')['TeamRushYardsAllowed'].expanding().sum().values / data.groupby('Season')['TeamRushAttAllowed'].expanding().sum().values
    data['SeasonTeamRushTdPctAllowed'] = data.groupby('Season')['TeamRushTdAllowed'].expanding().sum().values / data.groupby('Season')['TeamRushAttAllowed'].expanding().sum().values
    # Season 3 Game calcs
    data['Season3GameTeamPassCompPctAllowed'] = data.groupby('Season')['TeamPassCompAllowed'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamPassAttAllowed'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamPassIntPctAllowed'] = data.groupby('Season')['TeamPassIntAllowed'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamPassAttAllowed'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamPassTdPctAllowed'] = data.groupby('Season')['TeamPassTdAllowed'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamPassAttAllowed'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamPassYardsPerCompAllowed'] = data.groupby('Season')['TeamPassYardsAllowed'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamPassCompAllowed'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamPassYardsPerAttAllowed'] = data.groupby('Season')['TeamPassYardsAllowed'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamPassAttAllowed'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamAdjPassYardsPerAttAllowed'] = (data.groupby('Season')['TeamPassYardsAllowed'].rolling(3, min_periods=0).sum().values + 20 * data.groupby('Season')['TeamPassTdAllowed'].rolling(3, min_periods=0).sum().values - 45 * data.groupby('Season')['TeamPassIntAllowed'].rolling(3, min_periods=0).sum().values) / data.groupby('Season')['TeamPassAttAllowed'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamYardsPerRushAllowed'] = data.groupby('Season')['TeamRushYardsAllowed'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamRushAttAllowed'].rolling(3, min_periods=0).sum().values
    data['Season3GameTeamRushTdPctAllowed'] = data.groupby('Season')['TeamRushTdAllowed'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamRushAttAllowed'].rolling(3, min_periods=0).sum().values
    # Season 6 Game calcs 
    data['Season6GameTeamPassCompPctAllowed'] = data.groupby('Season')['TeamPassCompAllowed'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamPassAttAllowed'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamPassIntPctAllowed'] = data.groupby('Season')['TeamPassIntAllowed'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamPassAttAllowed'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamPassTdPctAllowed'] = data.groupby('Season')['TeamPassTdAllowed'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamPassAttAllowed'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamPassYardsPerCompAllowed'] = data.groupby('Season')['TeamPassYardsAllowed'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamPassCompAllowed'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamPassYardsPerAttAllowed'] = data.groupby('Season')['TeamPassYardsAllowed'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamPassAttAllowed'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamAdjPassYardsPerAttAllowed'] = (data.groupby('Season')['TeamPassYardsAllowed'].rolling(6, min_periods=0).sum().values + 20 * data.groupby('Season')['TeamPassTdAllowed'].rolling(3, min_periods=0).sum().values - 45 * data.groupby('Season')['TeamPassIntAllowed'].rolling(6, min_periods=0).sum().values) / data.groupby('Season')['TeamPassAttAllowed'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamYardsPerRushAllowed'] = data.groupby('Season')['TeamRushYardsAllowed'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamRushAttAllowed'].rolling(6, min_periods=0).sum().values
    data['Season6GameTeamRushTdPctAllowed'] = data.groupby('Season')['TeamRushTdAllowed'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamRushAttAllowed'].rolling(6, min_periods=0).sum().values

    # Create lagged versions of the columns
    new_cols = data.columns
    # Add lagged features for the other weighted columns too
    weighted_cols = ['TeamPassCompPctAllowed','TeamPassIntPctAllowed','TeamPassTdPctAllowed','TeamPassYardsPerCompAllowed','TeamPassYardsPerAttAllowed',
                      'TeamAdjPassYardsPerAttAllowed', 'TeamYardsPerRushAllowed','TeamRushTdPctAllowed']
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
    data = create_allowed_data(data)
    #stat_cols = TEAM_DEFENSE_STAT_COLS
    stat_cols = TEAM_DEFENSE_ALLOWED_STAT_COLS
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

    stat_type = 'team-defense'
    data_dir = TEAM_DATA_PATHS[stat_type]
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

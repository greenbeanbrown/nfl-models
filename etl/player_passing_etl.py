import pandas as pd
import numpy as np
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, \
                               export_data, load_data, get_data, agg_raw_files, \
                               filter_sort_data, clean_player_data

sys.path.insert(0, '..')
from columns import PLAYER_PASSING_STAT_COLS, PLAYER_PASSING_EXTRA_COLS, PLAYER_INFO_COLS, TEAM_INFO_COLS, PLAYER_PASSING_TEAM_COLS
from paths import PLAYER_DATA_PATHS

def calc_stats(data):
    """
    Performs calculations to create
    certain metrics that aren't given
    """
    # Add some features that require weighted averages
    # (most of these are found on Pro-Football-Reference game logs)
    data['PassCompPct'] = data['PassComp'] / data['PassAtt']
    data['PassIntPct'] = data['PassInt'] / data['PassAtt']
    data['PassTdPct'] = data['PassTd'] / data['PassAtt']
    data['PassYardsPerComp'] = data['PassYards'] / data['PassComp']
    data['PassYardsPerAtt'] = data['PassYards'] / data['PassAtt']
    data['AdjPassYardsPerAtt'] = (data['PassYards'] + 20 * data['PassTd'] - 45 * data['PassInt']) / data['PassAtt']
    # QBR has a special calculation
    uncapped_a = (data['PassCompPct'] - 0.3) * 5
    a = [max(min(2.375, i), 0) for i in uncapped_a]
    uncapped_b = (data['PassYardsPerAtt'] - 3) * 0.25
    b = [max(min(2.375, i), 0) for i in uncapped_b]
    uncapped_c = data['PassTdPct'] * 20
    c = [max(min(2.375, i), 0) for i in uncapped_c]
    uncapped_d = 2.375 - (data['PassIntPct'] * 25)
    d = [max(min(2.375, i), 0) for i in uncapped_d]
    data['QBR'] = ((pd.Series(a) + pd.Series(b) + pd.Series(c) + pd.Series(d)) / 6 * 100).tolist()
    # Team-related calcs
    data['PassAttPerTeamPlay'] = data['PassAtt'] / data['TeamTotalPlays']
    data['PassCompPerTeamPlay'] = data['PassComp'] / data['TeamTotalPlays']
    data['PassYardsPerTeamPlay'] = data['PassYards'] /data['TeamTotalPlays']
    data['PassTdPerTeamPlay'] = data['PassTd'] / data['TeamTotalPlays']
    data['PassIntPerTeamPlay'] = data['PassInt'] / data['TeamTotalPlays']

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
    data['CareerPassCompPct'] = data['PassComp'].expanding().sum() / data['PassAtt'].expanding().sum()
    data['CareerPassIntPct'] = data['PassInt'].expanding().sum() / data['PassAtt'].expanding().sum()
    data['CareerPassTdPct'] = data['PassTd'].expanding().sum() / data['PassAtt'].expanding().sum()
    data['CareerPassYardsPerComp'] = data['PassYards'].expanding().sum() / data['PassComp'].expanding().sum()
    data['CareerPassYardsPerAtt'] = data['PassYards'].expanding().sum() / data['PassAtt'].expanding().sum()
    data['CareerAdjPassYardsPerAtt'] = (data['PassYards'].expanding().sum() + 20 * data['PassTd'].expanding().sum() - 45 * data['PassInt'].expanding().sum()) / data['PassAtt'].expanding().sum()
    data['CareerPassAttPerTeamPlay'] = data['PassAtt'].expanding().sum() / data['TeamTotalPlays'].expanding().sum()
    data['CareerPassCompPerTeamPlay'] = data['PassComp'].expanding().sum() / data['TeamTotalPlays'].expanding().sum()
    data['CareerPassYardsPerTeamPlay'] = data['PassYards'].expanding().sum() /data['TeamTotalPlays'].expanding().sum()
    data['CareerPassTdPerTeamPlay'] = data['PassTd'].expanding().sum() / data['TeamTotalPlays'].expanding().sum()
    data['CareerPassIntPerTeamPlay'] = data['PassInt'].expanding().sum() / data['TeamTotalPlays'].expanding().sum()
    # Career 3 Game calcs
    data['3GamePassCompPct'] = data['PassComp'].rolling(3, min_periods=0).sum() / data['PassAtt'].rolling(3, min_periods=0).sum()
    data['3GamePassIntPct'] = data['PassInt'].rolling(3, min_periods=0).sum() / data['PassAtt'].rolling(3, min_periods=0).sum()
    data['3GamePassTdPct'] = data['PassTd'].rolling(3, min_periods=0).sum() / data['PassAtt'].rolling(3, min_periods=0).sum()
    data['3GamePassYardsPerComp'] = data['PassYards'].rolling(3, min_periods=0).sum() / data['PassComp'].rolling(3, min_periods=0).sum()
    data['3GamePassYardsPerAtt'] = data['PassYards'].rolling(3, min_periods=0).sum() / data['PassAtt'].rolling(3, min_periods=0).sum()
    data['3GameAdjPassYardsPerAtt'] = (data['PassYards'].rolling(3, min_periods=0).sum() + 20 * data['PassTd'].rolling(3, min_periods=0).sum() - 45 * data['PassInt'].rolling(3, min_periods=0).sum()) / data['PassAtt'].rolling(3, min_periods=0).sum()
    data['3GamePassAttPerTeamPlay'] = data['PassAtt'].rolling(3, min_periods=0).sum() / data['TeamTotalPlays'].rolling(3, min_periods=0).sum()
    data['3GamePassCompPerTeamPlay'] = data['PassComp'].rolling(3, min_periods=0).sum() / data['TeamTotalPlays'].rolling(3, min_periods=0).sum()
    data['3GamePassYardsPerTeamPlay'] = data['PassYards'].rolling(3, min_periods=0).sum() /data['TeamTotalPlays'].rolling(3, min_periods=0).sum()
    data['3GamePassTdPerTeamPlay'] = data['PassTd'].rolling(3, min_periods=0).sum() / data['TeamTotalPlays'].rolling(3, min_periods=0).sum()
    data['3GamePassIntPerTeamPlay'] = data['PassInt'].rolling(3, min_periods=0).sum() / data['TeamTotalPlays'].rolling(3, min_periods=0).sum()
    # Career 6 Game calcs 
    data['6GamePassCompPct'] = data['PassComp'].rolling(6, min_periods=0).sum() / data['PassAtt'].rolling(6, min_periods=0).sum()
    data['6GamePassIntPct'] = data['PassInt'].rolling(6, min_periods=0).sum() / data['PassAtt'].rolling(6, min_periods=0).sum()
    data['6GamePassTdPct'] = data['PassTd'].rolling(6, min_periods=0).sum() / data['PassAtt'].rolling(6, min_periods=0).sum()
    data['6GamePassYardsPerComp'] = data['PassYards'].rolling(6, min_periods=0).sum() / data['PassComp'].rolling(6, min_periods=0).sum()
    data['6GamePassYardsPerAtt'] = data['PassYards'].rolling(6, min_periods=0).sum() / data['PassAtt'].rolling(6, min_periods=0).sum()
    data['6GameAdjPassYardsPerAtt'] = (data['PassYards'].rolling(6, min_periods=0).sum() + 20 * data['PassTd'].rolling(6, min_periods=0).sum() - 45 * data['PassInt'].rolling(6, min_periods=0).sum()) / data['PassAtt'].rolling(6, min_periods=0).sum()
    data['6GamePassAttPerTeamPlay'] = data['PassAtt'].rolling(6, min_periods=0).sum() / data['TeamTotalPlays'].rolling(6, min_periods=0).sum()
    data['6GamePassCompPerTeamPlay'] = data['PassComp'].rolling(6, min_periods=0).sum() / data['TeamTotalPlays'].rolling(6, min_periods=0).sum()
    data['6GamePassYardsPerTeamPlay'] = data['PassYards'].rolling(6, min_periods=0).sum() /data['TeamTotalPlays'].rolling(6, min_periods=0).sum()
    data['6GamePassTdPerTeamPlay'] = data['PassTd'].rolling(6, min_periods=0).sum() / data['TeamTotalPlays'].rolling(6, min_periods=0).sum()
    data['6GamePassIntPerTeamPlay'] = data['PassInt'].rolling(6, min_periods=0).sum() / data['TeamTotalPlays'].rolling(6, min_periods=0).sum()
    # Season-long calcs
    data['SeasonPassCompPct'] = data.groupby('Season')['PassComp'].expanding().sum().values / data.groupby('Season')['PassAtt'].expanding().sum().values
    data['SeasonPassIntPct'] = data.groupby('Season')['PassInt'].expanding().sum().values / data.groupby('Season')['PassAtt'].expanding().sum().values
    data['SeasonPassTdPct'] = data.groupby('Season')['PassTd'].expanding().sum().values / data.groupby('Season')['PassAtt'].expanding().sum().values
    data['SeasonPassYardsPerComp'] = data.groupby('Season')['PassYards'].expanding().sum().values / data.groupby('Season')['PassComp'].expanding().sum().values
    data['SeasonPassYardsPerAtt'] = data.groupby('Season')['PassYards'].expanding().sum().values / data.groupby('Season')['PassAtt'].expanding().sum().values
    data['SeasonAdjPassYardsPerAtt'] = (data.groupby('Season')['PassYards'].expanding().sum().values + 20 * data.groupby('Season')['PassTd'].expanding().sum().values - 45 * data.groupby('Season')['PassInt'].expanding().sum().values) / data.groupby('Season')['PassAtt'].expanding().sum().values
    data['SeasonPassAttPerTeamPlay'] = data.groupby('Season')['PassAtt'].expanding().sum().values / data.groupby('Season')['TeamTotalPlays'].expanding().sum().values
    data['SeasonPassCompPerTeamPlay'] = data.groupby('Season')['PassComp'].expanding().sum().values / data.groupby('Season')['TeamTotalPlays'].expanding().sum().values
    data['SeasonPassYardsPerTeamPlay'] = data.groupby('Season')['PassYards'].expanding().sum().values /data.groupby('Season')['TeamTotalPlays'].expanding().sum().values
    data['SeasonPassTdPerTeamPlay'] = data.groupby('Season')['PassTd'].expanding().sum().values / data.groupby('Season')['TeamTotalPlays'].expanding().sum().values
    data['SeasonPassIntPerTeamPlay'] = data.groupby('Season')['PassInt'].expanding().sum().values / data.groupby('Season')['TeamTotalPlays'].expanding().sum().values 
    # Season 3 Game calcs
    data['Season3GamePassCompPct'] = data.groupby('Season')['PassComp'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['PassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GamePassIntPct'] = data.groupby('Season')['PassInt'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['PassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GamePassTdPct'] = data.groupby('Season')['PassTd'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['PassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GamePassYardsPerComp'] = data.groupby('Season')['PassYards'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['PassComp'].rolling(3, min_periods=0).sum().values
    data['Season3GamePassYardsPerAtt'] = data.groupby('Season')['PassYards'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['PassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameAdjPassYardsPerAtt'] = (data.groupby('Season')['PassYards'].rolling(3, min_periods=0).sum().values + 20 * data.groupby('Season')['PassTd'].rolling(3, min_periods=0).sum().values - 45 * data.groupby('Season')['PassInt'].rolling(3, min_periods=0).sum().values) / data.groupby('Season')['PassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GamePassAttPerTeamPlay'] = data.groupby('Season')['PassAtt'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GamePassCompPerTeamPlay'] = data.groupby('Season')['PassComp'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GamePassYardsPerTeamPlay'] = data.groupby('Season')['PassYards'].rolling(3, min_periods=0).sum().values /data.groupby('Season')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GamePassTdPerTeamPlay'] = data.groupby('Season')['PassTd'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GamePassIntPerTeamPlay'] = data.groupby('Season')['PassInt'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    # Season 6 Game calcs 
    data['Season6GamePassCompPct'] = data.groupby('Season')['PassComp'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['PassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GamePassIntPct'] = data.groupby('Season')['PassInt'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['PassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GamePassTdPct'] = data.groupby('Season')['PassTd'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['PassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GamePassYardsPerComp'] = data.groupby('Season')['PassYards'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['PassComp'].rolling(6, min_periods=0).sum().values
    data['Season6GamePassYardsPerAtt'] = data.groupby('Season')['PassYards'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['PassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameAdjPassYardsPerAtt'] = (data.groupby('Season')['PassYards'].rolling(6, min_periods=0).sum().values + 20 * data.groupby('Season')['PassTd'].rolling(6, min_periods=0).sum().values - 45 * data.groupby('Season')['PassInt'].rolling(6, min_periods=0).sum().values) / data.groupby('Season')['PassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GamePassAttPerTeamPlay'] = data.groupby('Season')['PassAtt'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GamePassCompPerTeamPlay'] = data.groupby('Season')['PassComp'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GamePassYardsPerTeamPlay'] = data.groupby('Season')['PassYards'].rolling(6, min_periods=0).sum().values /data.groupby('Season')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GamePassTdPerTeamPlay'] = data.groupby('Season')['PassTd'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GamePassIntPerTeamPlay'] = data.groupby('Season')['PassInt'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values

    # Create lagged versions of the columns
    new_cols = data.columns
    weighted_cols = ['PassCompPct','PassIntPct','PassTdPct','PassYardsPerComp','PassYardsPerAtt','AdjPassYardsPerAtt','QBR',
                     'PassAttPerTeamPlay','PassCompPerTeamPlay', 'PassYardsPerTeamPlay', 'PassTdPerTeamPlay', 'PassIntPerTeamPlay'] 

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
    stat_cols = [i for i in PLAYER_PASSING_STAT_COLS if i not in PLAYER_PASSING_TEAM_COLS + PLAYER_PASSING_EXTRA_COLS] + ['QBR']
    # Isolate data for each player
    player_ids = data['PlayerId'].unique()

    agg_data_list = []

    for player_id in player_ids:
        player_data = data[data['PlayerId'] == player_id].copy()
        # Calculate different features and add to the data
        player_data = add_specific_features(player_data)
        player_dict = add_rolling_features(player_data, stat_cols)
        # Append current player-season dict to the agg list
        agg_data_list.append(pd.DataFrame(player_dict))  # Create DataFrame from player_dict
    # Create an agg dataframe from the each individual player-season dict
    agg_data = pd.concat(agg_data_list, ignore_index=True)  # Concatenate all player DataFrames
    agg_data.reset_index(inplace=True, drop=True)

    # Remove extra columns that were needed for other calculations 
    for col in PLAYER_PASSING_TEAM_COLS:
        agg_data.drop(col, inplace=True, axis=1)

    return agg_data

if __name__ == '__main__':

    stat_type = 'player-passing'
    data_dir = PLAYER_DATA_PATHS[stat_type]
    # Agg the raw data
    agg_raw_files(stat_type=stat_type)
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

    # Filter data
    raw_player_stat_type_data = filter_sort_data(merged_data, stat_type=stat_type)
    # Add model features
    player_stat_type_data = add_model_features(raw_player_stat_type_data)
    # Split the current and lagged features into 2 tables
    player_stat_type, lag_player_stat_type = split_lagged_data(player_stat_type_data, info_cols=PLAYER_INFO_COLS)
    # Export
    export_data(player_stat_type, lag_player_stat_type, data_dir, stat_type=stat_type)
    # Load into DB
    load_data(data_dir, stat_type=stat_type)

import pandas as pd
import numpy as np
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, \
                               export_data, load_data, get_data, \
                               filter_sort_data, clean_player_data

sys.path.insert(0, '..')
from columns import PLAYER_RECEIVING_STAT_COLS, PLAYER_RECEIVING_EXTRA_COLS, PLAYER_INFO_COLS, TEAM_INFO_COLS, PLAYER_RECEIVING_TEAM_COLS
from paths import PLAYER_DATA_PATHS

def calc_stats(data):
    """
    Performs calculations to create
    certain metrics that aren't given
    """
    # Calculate some specific metrics
    data['YardsPerRec'] = data['RecYards'] / data['Receptions']
    data['YardsPerRecTarget'] = data['RecYards'] / data['RecTargets']
    data['CatchPct'] = data['Receptions'] / data['RecTargets']
    # Calcs that use team data
    data['RecTargetsPerTeamPlay'] = data['RecTargets'] / data['TeamTotalPlays']
    data['RecPerTeamPlay'] = data['Receptions'] / data['TeamTotalPlays']
    data['RecYardsPerTeamPlay'] = data['RecYards'] / data['TeamTotalPlays']

    data['RecShare'] = data['Receptions'] / data['TeamPassComp']
    data['RecTargetShare'] = data['RecTargets'] / data['TeamPassAtt']
    data['RecYardsShare'] = data['RecYards'] / data['TeamPassYards']

    data['OffSnapShare'] = data['OffSnaps'] / data['TeamTotalPlays']

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
    data['CareerYardsPerRec'] = data['RecYards'].expanding().sum() / data['Receptions'].expanding().sum()
    data['CareerPerRecTarget'] = data['RecYards'].expanding().sum() / data['RecTargets'].expanding().sum()
    data['CareerCatchPct'] = data['Receptions'].expanding().sum() / data['RecTargets'].expanding().sum()
    data['CareerRecTargetsPerTeamPlay'] = data['RecTargets'].expanding().sum() / data['TeamTotalPlays'].expanding().sum()
    data['CareerRecPerTeamPlay'] = data['Receptions'].expanding().sum() / data['TeamTotalPlays'].expanding().sum()
    data['CareerRecYardsPerTeamPlay'] = data['RecYards'].expanding().sum() / data['TeamTotalPlays'].expanding().sum()
    data['CareerRecShare'] = data['Receptions'].expanding().sum() / data['TeamPassComp'].expanding().sum()
    data['CareerRecTargetShare'] = data['RecTargets'].expanding().sum() / data['TeamPassAtt'].expanding().sum()
    data['CareerRecYardsShare'] = data['RecYards'].expanding().sum() / data['TeamPassYards'].expanding().sum()
    data['CareerSnapShare'] = data['OffSnaps'].expanding().sum() / data['TeamTotalPlays'].expanding().sum()   
    # 3 Game calcs
    data['3GameYardsPerRush'] = data['RecYards'].rolling(3, min_periods=0).sum() / data['Receptions'].rolling(3, min_periods=0).sum()
    data['3GameYardsPerRecTarget'] = data['RecYards'].rolling(3, min_periods=0).sum() / data['RecTargets'].rolling(3, min_periods=0).sum()
    data['3GameCatchPct'] = data['Receptions'].rolling(3, min_periods=0).sum() / data['RecTargets'].rolling(3, min_periods=0).sum()
    data['3GameRecTargetsPerTeamPlay'] = data['RecTargets'].rolling(3, min_periods=0).sum() / data['TeamTotalPlays'].rolling(3, min_periods=0).sum()
    data['3GameRecPerTeamPlay'] = data['Receptions'].rolling(3, min_periods=0).sum() / data['TeamTotalPlays'].rolling(3, min_periods=0).sum()
    data['3GameRecYardsPerTeamPlay'] = data['RecYards'].rolling(3, min_periods=0).sum() / data['TeamTotalPlays'].rolling(3, min_periods=0).sum()
    data['3GameRecShare'] = data['Receptions'].rolling(3, min_periods=0).sum() / data['TeamPassComp'].rolling(3, min_periods=0).sum()
    data['3GameRecTargetShare'] = data['RecTargets'].rolling(3, min_periods=0).sum() / data['TeamPassAtt'].rolling(3, min_periods=0).sum()
    data['3GameRecYardsShare'] = data['RecYards'].rolling(3, min_periods=0).sum() / data['TeamPassYards'].rolling(3, min_periods=0).sum()
    data['3GameSnapShare'] = data['OffSnaps'].rolling(3, min_periods=0).sum() / data['TeamTotalPlays'].rolling(3, min_periods=0).sum()
    # 6 Game calcs 
    data['6GameYardsPerRush'] = data['RecYards'].rolling(6, min_periods=0).sum() / data['Receptions'].rolling(6, min_periods=0).sum()
    data['6GameYardsPerRecTarget'] = data['RecYards'].rolling(6, min_periods=0).sum() / data['RecTargets'].rolling(6, min_periods=0).sum()
    data['6GameCatchPct'] = data['Receptions'].rolling(6, min_periods=0).sum() / data['RecTargets'].rolling(6, min_periods=0).sum()
    data['6GameRecTargetsPerTeamPlay'] = data['RecTargets'].rolling(6, min_periods=0).sum() / data['TeamTotalPlays'].rolling(6, min_periods=0).sum()
    data['6GameRecPerTeamPlay'] = data['Receptions'].rolling(6, min_periods=0).sum() / data['TeamTotalPlays'].rolling(6, min_periods=0).sum()
    data['6GameRecYardsPerTeamPlay'] = data['RecYards'].rolling(6, min_periods=0).sum() / data['TeamTotalPlays'].rolling(6, min_periods=0).sum()
    data['6GameRecShare'] = data['Receptions'].rolling(6, min_periods=0).sum() / data['TeamPassComp'].rolling(6, min_periods=0).sum()
    data['6GameRecTargetShare'] = data['RecTargets'].rolling(6, min_periods=0).sum() / data['TeamPassAtt'].rolling(6, min_periods=0).sum()
    data['6GameRecYardsShare'] = data['RecYards'].rolling(6, min_periods=0).sum() / data['TeamPassYards'].rolling(6, min_periods=0).sum()
    data['6GameSnapShare'] = data['OffSnaps'].rolling(6, min_periods=0).sum() / data['TeamTotalPlays'].rolling(6, min_periods=0).sum()
    # Season-long calcs
    data['SeasonYardsPerRec'] = data.groupby('Season')['RecYards'].expanding().sum().values / data.groupby('Season')['Receptions'].expanding().sum().values
    data['SeasonPerRecTarget'] = data.groupby('Season')['RecYards'].expanding().sum().values / data.groupby('Season')['RecTargets'].expanding().sum().values
    data['SeasonCatchPct'] = data.groupby('Season')['Receptions'].expanding().sum().values / data.groupby('Season')['RecTargets'].expanding().sum().values
    data['SeasonRecTargetsPerTeamPlay'] = data.groupby('Season')['RecTargets'].expanding().sum().values / data.groupby('Season')['TeamTotalPlays'].expanding().sum().values
    data['SeasonRecPerTeamPlay'] = data.groupby('Season')['Receptions'].expanding().sum().values / data.groupby('Season')['TeamTotalPlays'].expanding().sum().values
    data['SeasonRecYardsPerTeamPlay'] = data.groupby('Season')['RecYards'].expanding().sum().values / data.groupby('Season')['TeamTotalPlays'].expanding().sum().values
    data['SeasonRecShare'] = data.groupby('Season')['Receptions'].expanding().sum().values / data.groupby('Season')['TeamPassComp'].expanding().sum().values
    data['SeasonRecTargetShare'] = data.groupby('Season')['RecTargets'].expanding().sum().values / data.groupby('Season')['TeamPassAtt'].expanding().sum().values
    data['SeasonRecYardsShare'] = data.groupby('Season')['RecYards'].expanding().sum().values / data.groupby('Season')['TeamPassYards'].expanding().sum().values
    data['SeasonSnapShare'] = data.groupby('Season')['OffSnaps'].expanding().sum().values / data.groupby('Season')['TeamTotalPlays'].expanding().sum().values
    # Season 3 Game calcs
    data['Season3GameYardsPerRush'] = data.groupby('Season')['RecYards'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['Receptions'].rolling(3, min_periods=0).sum().values
    data['Season3GameYardsPerRecTarget'] = data.groupby('Season')['RecYards'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['RecTargets'].rolling(3, min_periods=0).sum().values
    data['Season3GameCatchPct'] = data.groupby('Season')['Receptions'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['RecTargets'].rolling(3, min_periods=0).sum().values
    data['Season3GameRecTargetsPerTeamPlay'] = data.groupby('Season')['RecTargets'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GameRecPerTeamPlay'] = data.groupby('Season')['Receptions'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GameRecYardsPerTeamPlay'] = data.groupby('Season')['RecYards'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GameRecShare'] = data.groupby('Season')['Receptions'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamPassComp'].rolling(3, min_periods=0).sum().values
    data['Season3GameRecTargetShare'] = data.groupby('Season')['RecTargets'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamPassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameRecYardsShare'] = data.groupby('Season')['RecYards'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamPassYards'].rolling(3, min_periods=0).sum().values
    data['Season3GameSnapShare'] = data.groupby('Season')['OffSnaps'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(3, min_periods=0).sum().values
    # Season 6 Game calcs 
    data['Season6GameYardsPerRush'] = data.groupby('Season')['RecYards'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['Receptions'].rolling(6, min_periods=0).sum().values
    data['Season6GameYardsPerRecTarget'] = data.groupby('Season')['RecYards'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['RecTargets'].rolling(6, min_periods=0).sum().values
    data['Season6GameCatchPct'] = data.groupby('Season')['Receptions'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['RecTargets'].rolling(6, min_periods=0).sum().values
    data['Season6GameRecPerPlayerSnap'] = data.groupby('Season')['Receptions'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['OffSnaps'].rolling(6, min_periods=0).sum().values
    data['Season6GameRecYardsPerSnap'] = data.groupby('Season')['RecYards'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['OffSnaps'].rolling(6, min_periods=0).sum().values
    data['Season6GameRecTargetsPerTeamPlay'] = data.groupby('Season')['RecTargets'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GameRecPerTeamPlay'] = data.groupby('Season')['Receptions'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GameRecYardsPerTeamPlay'] = data.groupby('Season')['RecYards'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GameRecShare'] = data.groupby('Season')['Receptions'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamPassComp'].rolling(6, min_periods=0).sum().values
    data['Season6GameRecTargetShare'] = data.groupby('Season')['RecTargets'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamPassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameRecYardsShare'] = data.groupby('Season')['RecYards'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamPassYards'].rolling(6, min_periods=0).sum().values
    data['Season6GameSnapShare'] = data.groupby('Season')['OffSnaps'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['TeamTotalPlays'].rolling(6, min_periods=0).sum().values

    # Create lagged versions of the columns
    new_cols = data.columns
    weighted_cols = ['YardsPerRec','YardsPerRecTarget','CatchPct',
                     'RecTargetsPerTeamPlay','RecPerTeamPlay','RecYardsPerTeamPlay',
                     'RecShare','RecTargetShare','RecYardsShare','OffSnapShare']
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
    stat_cols = [i for i in PLAYER_RECEIVING_STAT_COLS if i not in PLAYER_RECEIVING_TEAM_COLS + PLAYER_RECEIVING_EXTRA_COLS] 
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
    for col in PLAYER_RECEIVING_TEAM_COLS:
        agg_data.drop(col, inplace=True, axis=1)


    return agg_data

if __name__ == '__main__':

    stat_type = 'player-receiving'
    data_dir = PLAYER_DATA_PATHS[stat_type]
    # Read raw excel file
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
    # Replace invalid values with NaN
    player_stat_type_data = player_stat_type_data.replace([np.inf, -np.inf], np.nan)
    # Split the current and lagged features into 2 tables
    player_stat_type, lag_player_stat_type = split_lagged_data(player_stat_type_data, info_cols=PLAYER_INFO_COLS)
    # Export
    export_data(player_stat_type, lag_player_stat_type, data_dir, stat_type=stat_type)
    # Load into DB
    load_data(data_dir, stat_type=stat_type)

import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

from datetime import datetime
import os 
import sqlite3
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, \
                               export_data, load_data, \
                               filter_sort_data, clean_team_data, get_data

sys.path.insert(0, '..')
from paths import DB_PATH, LEAGUE_DATA_PATHS
from columns import LEAGUE_INFO_COLS, LEAGUE_PASSING_COLS, LEAGUE_PASSING_STAT_COLS, LEAGUE_PASSING_WEIGHTED_COLS

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
    data['LeaguePassCompPct'] = data['LeaguePassComp'] / data['LeaguePassAtt']
    data['LeaguePassIntPct'] = data['LeaguePassInt'] / data['LeaguePassAtt']
    data['LeaguePassTdPct'] = data['LeaguePassTd'] / data['LeaguePassAtt']
    data['LeaguePassYardsPerComp'] = data['LeaguePassYards'] / data['LeaguePassComp']
    data['LeaguePassYardsPerAtt'] = data['LeaguePassYards'] / data['LeaguePassAtt']
    data['LeagueAdjPassYardsPerAtt'] = (data['LeaguePassYards'] + 20 * data['LeaguePassTd'] - 45 * data['LeaguePassInt']) / data['LeaguePassAtt']

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
    data['CareerLeaguePassCompPct'] = data['LeaguePassComp'].expanding().sum().values / data['LeaguePassAtt'].expanding().sum().values
    data['CareerLeaguePassIntPct'] = data['LeaguePassInt'].expanding().sum().values / data['LeaguePassAtt'].expanding().sum().values
    data['CareerLeaguePassTdPct'] = data['LeaguePassTd'].expanding().sum().values / data['LeaguePassAtt'].expanding().sum().values
    data['CareerLeaguePassYardsPerComp'] = data['LeaguePassYards'].expanding().sum().values / data['LeaguePassComp'].expanding().sum().values
    data['CareerLeaguePassYardsPerAtt'] = data['LeaguePassYards'].expanding().sum().values / data['LeaguePassAtt'].expanding().sum().values
    data['CareerLeagueAdjPassYardsPerAtt'] = (data['LeaguePassYards'].expanding().sum().values + 20 * data['LeaguePassTd'].expanding().sum().values - 45 * data['LeaguePassInt'].expanding().sum().values) / data['LeaguePassAtt'].expanding().sum().values
    # 3 Game calcs
    data['3GameLeaguePassCompPct'] = data['LeaguePassComp'].rolling(3, min_periods=0).sum().values / data['LeaguePassAtt'].rolling(3, min_periods=0).sum().values
    data['3GameLeaguePassIntPct'] = data['LeaguePassInt'].rolling(3, min_periods=0).sum().values / data['LeaguePassAtt'].rolling(3, min_periods=0).sum().values
    data['3GameLeaguePassTdPct'] = data['LeaguePassTd'].rolling(3, min_periods=0).sum().values / data['LeaguePassAtt'].rolling(3, min_periods=0).sum().values
    data['3GameLeaguePassYardsPerComp'] = data['LeaguePassYards'].rolling(3, min_periods=0).sum().values / data['LeaguePassComp'].rolling(3, min_periods=0).sum().values
    data['3GameLeaguePassYardsPerAtt'] = data['LeaguePassYards'].rolling(3, min_periods=0).sum().values / data['LeaguePassAtt'].rolling(3, min_periods=0).sum().values
    data['3GameLeagueAdjPassYardsPerAtt'] = (data['LeaguePassYards'].rolling(3, min_periods=0).sum().values + 20 * data['LeaguePassTd'].rolling(3, min_periods=0).sum().values - 45 * data['LeaguePassInt'].rolling(3, min_periods=0).sum().values) / data['LeaguePassAtt'].rolling(3, min_periods=0).sum().values
    # 6 Game calcs 
    data['6GameLeaguePassCompPct'] = data['LeaguePassComp'].rolling(6, min_periods=0).sum().values / data['LeaguePassAtt'].rolling(6, min_periods=0).sum().values
    data['6GameLeaguePassIntPct'] = data['LeaguePassInt'].rolling(6, min_periods=0).sum().values / data['LeaguePassAtt'].rolling(6, min_periods=0).sum().values
    data['6GameLeaguePassTdPct'] = data['LeaguePassTd'].rolling(6, min_periods=0).sum().values / data['LeaguePassAtt'].rolling(6, min_periods=0).sum().values
    data['6GameLeaguePassYardsPerComp'] = data['LeaguePassYards'].rolling(6, min_periods=0).sum().values / data['LeaguePassComp'].rolling(6, min_periods=0).sum().values
    data['6GameLeaguePassYardsPerAtt'] = data['LeaguePassYards'].rolling(6, min_periods=0).sum().values / data['LeaguePassAtt'].rolling(6, min_periods=0).sum().values
    data['6GameLeagueAdjPassYardsPerAtt'] = (data['LeaguePassYards'].rolling(6, min_periods=0).sum().values + 20 * data['LeaguePassTd'].rolling(6, min_periods=0).sum().values - 45 * data['LeaguePassInt'].rolling(6, min_periods=0).sum().values) / data['LeaguePassAtt'].rolling(6, min_periods=0).sum().values
    # Season-long calcs
    data['SeasonLeaguePassCompPct'] = data.groupby('Season')['LeaguePassComp'].expanding().sum().values / data.groupby('Season')['LeaguePassAtt'].expanding().sum().values
    data['SeasonLeaguePassIntPct'] = data.groupby('Season')['LeaguePassInt'].expanding().sum().values / data.groupby('Season')['LeaguePassAtt'].expanding().sum().values
    data['SeasonLeaguePassTdPct']  = data.groupby('Season')['LeaguePassTd'].expanding().sum().values / data.groupby('Season')['LeaguePassAtt'].expanding().sum().values
    data['SeasonLeaguePassYardsPerComp']  = data.groupby('Season')['LeaguePassYards'].expanding().sum().values / data.groupby('Season')['LeaguePassComp'].expanding().sum().values        
    data['SeasonLeaguePassYardsPerAtt']  = data.groupby('Season')['LeaguePassYards'].expanding().sum().values / data.groupby('Season')['LeaguePassAtt'].expanding().sum().values        
    data['SeasonLeagueAdjPassYardsPerAtt']  = (data.groupby('Season')['LeaguePassYards'].expanding().sum().values + 20 * data.groupby('Season')['LeaguePassTd'].expanding().sum().values - 45 * data.groupby('Season')['LeaguePassInt'].expanding().sum().values) / data.groupby('Season')['LeaguePassAtt'].expanding().sum().values        
    # 3 Game calcs
    data['Season3GameLeaguePassCompPct'] = data.groupby('Season')['LeaguePassComp'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['LeaguePassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameLeaguePassIntPct'] = data.groupby('Season')['LeaguePassInt'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['LeaguePassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameLeaguePassTdPct'] = data.groupby('Season')['LeaguePassTd'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['LeaguePassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameLeaguePassYardsPerComp'] = data.groupby('Season')['LeaguePassYards'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['LeaguePassComp'].rolling(3, min_periods=0).sum().values
    data['Season3GameLeaguePassYardsPerAtt'] = data.groupby('Season')['LeaguePassYards'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['LeaguePassAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameLeagueAdjPassYardsPerAtt'] = (data.groupby('Season')['LeaguePassYards'].rolling(3, min_periods=0).sum().values + 20 * data.groupby('Season')['LeaguePassTd'].rolling(3, min_periods=0).sum().values - 45 * data.groupby('Season')['LeaguePassInt'].rolling(3, min_periods=0).sum().values) / data.groupby('Season')['LeaguePassAtt'].rolling(3, min_periods=0).sum().values
    # 6 Game calcs 
    data['Season6GameLeaguePassCompPct'] = data.groupby('Season')['LeaguePassComp'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['LeaguePassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameLeaguePassIntPct'] = data.groupby('Season')['LeaguePassInt'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['LeaguePassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameLeaguePassTdPct'] = data.groupby('Season')['LeaguePassTd'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['LeaguePassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameLeaguePassYardsPerComp'] = data.groupby('Season')['LeaguePassYards'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['LeaguePassComp'].rolling(6, min_periods=0).sum().values
    data['Season6GameLeaguePassYardsPerAtt'] = data.groupby('Season')['LeaguePassYards'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['LeaguePassAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameLeagueAdjPassYardsPerAtt'] = (data.groupby('Season')['LeaguePassYards'].rolling(6, min_periods=0).sum().values + 20 * data.groupby('Season')['LeaguePassTd'].rolling(6, min_periods=0).sum().values - 45 * data.groupby('Season')['LeaguePassInt'].rolling(6, min_periods=0).sum().values) / data.groupby('Season')['LeaguePassAtt'].rolling(6, min_periods=0).sum().values

    # Create lagged versions of the columns
    new_cols = data.columns
    weighted_cols = ['LeaguePassCompPct','LeaguePassIntPct','LeaguePassTdPct','LeaguePassYardsPerComp','LeaguePassYardsPerAtt','LeagueAdjPassYardsPerAtt']
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
    stat_cols = LEAGUE_PASSING_STAT_COLS

    agg_data_list = []

    for season in data['Season'].unique():
        # Create a dataframe for only current season
        season_data = data[data['Season'] == season]
        # Get number of games for each day
        count_data = pd.DataFrame(season_data.groupby('GameDate')['GameDate'].count() / 2)
        count_data.rename(columns={'GameDate':'NumGames'}, inplace=True)
        count_data.reset_index(inplace=True)
        # Get sums for each game date
        sum_data = season_data.groupby(LEAGUE_INFO_COLS).sum()
        sum_data.reset_index(inplace=True)
        # Merge the grouped data together into current season league data
        league_data = pd.merge(count_data,
                               sum_data,
                               on = 'GameDate')

        # Calculate passing stats
        league_data = calc_stats(league_data)
        # Add season and weighted average features
        league_data = add_specific_features(league_data)
        # Add rolling features
        league_dict = add_rolling_features(league_data, stat_cols, league=True)
        # Append current player-season dict to the agg list
        agg_data_list.append(pd.DataFrame(league_dict))

    # Create an agg dataframe from the each individual player-season dict
    agg_data = pd.concat(agg_data_list, ignore_index=True)  # Concatenate all player DataFrames
    agg_data.reset_index(inplace=True, drop=True)

    return agg_data

if __name__ == '__main__':

    stat_type = 'league-passing'
    data_dir = LEAGUE_DATA_PATHS[stat_type]
    # Get team data 
    raw_data = get_data(stat_type = stat_type)
    # Rename columns from Team to League
    working_data = rename_raw_cols(raw_data, stat_type=stat_type)
    # Filter data
    raw_league_stat_type_data = filter_sort_data(working_data, stat_type=stat_type)
    # Add model features
    league_stat_type_data = add_model_features(raw_league_stat_type_data)
    # Split the current and lagged features into 2 tables
    league_stat_type_data, lag_league_stat_type_data = split_lagged_data(league_stat_type_data, info_cols=LEAGUE_INFO_COLS)
    # Export
    export_data(league_stat_type_data, lag_league_stat_type_data, data_dir, stat_type=stat_type)
    # Load into DB
    load_data(data_dir, stat_type=stat_type)

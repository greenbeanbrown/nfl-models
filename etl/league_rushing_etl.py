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
from columns import LEAGUE_INFO_COLS, LEAGUE_RUSHING_COLS, LEAGUE_RUSHING_STAT_COLS, LEAGUE_RUSHING_WEIGHTED_COLS

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
    data['LeagueYardsPerRush'] = data['LeagueRushYards'] / data['LeagueRushAtt']
    data['LeagueRushTdPct'] = data['LeagueRushTd'] / data['LeagueRushAtt']

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
    data['CareerLeagueYardsPerRush'] = data['LeagueRushYards'].expanding().sum() / data['LeagueRushAtt'].expanding().sum()
    data['CareerLeagueRushTdPct'] = data['LeagueRushTd'].expanding().sum() / data['LeagueRushAtt'].expanding().sum()
    # 3 Game calcs
    data['3GameLeagueYardsPerRush'] = data['LeagueRushYards'].rolling(3, min_periods=0).sum() / data['LeagueRushAtt'].rolling(3, min_periods=0).sum()
    data['3GameLeagueRushTdPct'] = data['LeagueRushTd'].rolling(3, min_periods=0).sum() / data['LeagueRushAtt'].rolling(3, min_periods=0).sum()
    # 6 Game calcs 
    data['6GameLeagueYardsPerRush'] = data['LeagueRushYards'].rolling(6, min_periods=0).sum() / data['LeagueRushAtt'].rolling(6, min_periods=0).sum()
    data['6GameLeagueRushTdPct'] = data['LeagueRushTd'].rolling(6, min_periods=0).sum() / data['LeagueRushAtt'].rolling(6, min_periods=0).sum()
    # Season-long calcs
    data['SeasonLeagueYardsPerRush'] = data.groupby('Season')['LeagueRushYards'].expanding().sum().values / data.groupby('Season')['LeagueRushAtt'].expanding().sum().values
    data['SeasonLeagueRushTdPct'] = data.groupby('Season')['LeagueRushTd'].expanding().sum().values / data.groupby('Season')['LeagueRushAtt'].expanding().sum().values
    # Season 3 Game
    data['Season3GameLeagueYardsPerRush'] = data.groupby('Season')['LeagueRushYards'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['LeagueRushAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameLeagueRushTdPct'] = data.groupby('Season')['LeagueRushTd'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['LeagueRushAtt'].rolling(3, min_periods=0).sum().values
    # Season 6 Game
    data['Season6GameLeagueYardsPerRush'] = data.groupby('Season')['LeagueRushYards'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['LeagueRushAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameLeagueRushTdPct'] = data.groupby('Season')['LeagueRushTd'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['LeagueRushAtt'].rolling(6, min_periods=0).sum().values


    # Create lagged versions of the columns
    new_cols = data.columns
    weighted_cols = ['LeagueYardsPerRush','LeagueRushTdPct']
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
    stat_cols = LEAGUE_RUSHING_STAT_COLS

    agg_data_list = []

    # Create a dataframe for entire dataset (i.e. Career)
    # Get number of games for each day
    count_data = pd.DataFrame(sorted_filtered_data.groupby('GameDate')['GameDate'].count() / 2)
    count_data.rename(columns={'GameDate':'NumGames'}, inplace=True)
    count_data.reset_index(inplace=True)
    # Get sums for each game date
    sum_data = sorted_filtered_data.groupby(LEAGUE_INFO_COLS).sum()
    sum_data.reset_index(inplace=True)
    # Merge the grouped data together into current season league data
    league_data = pd.merge(count_data,
                           sum_data,
                           on = 'GameDate')
    # Calculate stats
    league_data = calc_stats(league_data)
    # Add season and weighted average features
    league_data = add_specific_features(league_data)
    # Add rolling features
    league_dict = add_rolling_features(league_data, stat_cols, league=True)
    league_data = pd.DataFrame(league_dict)

    return league_data

if __name__ == '__main__':

    #season = 2022
    stat_type = 'league-rushing'
    data_dir = LEAGUE_DATA_PATHS[stat_type]
    # Get team data 
    raw_data = get_data(stat_type = stat_type)
    # Rename columns from Team to League
    working_data = rename_raw_cols(raw_data, stat_type = stat_type)
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

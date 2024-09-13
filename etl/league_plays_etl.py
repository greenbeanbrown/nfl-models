import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

from datetime import datetime
import os 
import sqlite3
import sys

from shared_etl_functions import read_raw_data, rename_raw_cols, split_lagged_data, add_rolling_features, \
                               export_data, load_data, nfl_connect, \
                               filter_sort_data, clean_team_data, get_data

sys.path.insert(0, '..')
from paths import DB_PATH, LEAGUE_DATA_PATHS
from columns import LEAGUE_INFO_COLS, LEAGUE_PLAYS_COLS, LEAGUE_PLAYS_STAT_COLS, LEAGUE_PLAYS_WEIGHTED_COLS, LEAGUE_PLAYS_EXTRA_COLS

def calc_stats(data):
    """
    Performs calculations to create
    certain metrics that aren't given
    """
    # Calculate some specific metrics
    data['LeagueRushPlayPct'] = data['LeagueRushAtt'] / data['LeagueTotalPlays']
    data['LeaguePassPlayPct'] = data['LeaguePassAtt'] / data['LeagueTotalPlays']
    data['League3rdDownPct']  = data['League3rdDownMade'] / data['League3rdDownAtt']
    data['League4thDownPct']  = data['League4thDownMade'] / data['League4thDownAtt']

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
    data['CareerLeagueRushPlayPct'] = data['LeagueRushAtt'].expanding().sum() / data['LeagueTotalPlays'].expanding().sum()
    data['CareerLeaguePassPlayPct'] = data['LeaguePassAtt'].expanding().sum() / data['LeagueTotalPlays'].expanding().sum()
    data['CareerLeague3rdDownPct']  = data['League3rdDownMade'].expanding().sum() / data['League3rdDownAtt'].expanding().sum()
    data['CareerLeague4thDownPct']  = data['League4thDownMade'].expanding().sum() / data['League4thDownAtt'].expanding().sum()
    # 3 Game calcs
    data['3GameLeagueRushPlayPct'] = data['LeagueRushAtt'].rolling(3, min_periods=0).sum() / data['LeagueTotalPlays'].rolling(3, min_periods=0).sum()
    data['3GameLeaguePassPlayPct'] = data['LeaguePassAtt'].rolling(3, min_periods=0).sum() / data['LeagueTotalPlays'].rolling(3, min_periods=0).sum()
    data['3GameLeague3rdDownPct'] = data['League3rdDownMade'].rolling(3, min_periods=0).sum() / data['League3rdDownAtt'].rolling(3, min_periods=0).sum()
    data['3GameLeague4thDownPct'] = data['League4thDownMade'].rolling(3, min_periods=0).sum() / data['League4thDownAtt'].rolling(3, min_periods=0).sum()
    # 6 Game calcs 
    data['6GameLeagueRushPlayPct'] = data['LeagueRushAtt'].rolling(6, min_periods=0).sum() / data['LeagueTotalPlays'].rolling(6, min_periods=0).sum()
    data['6GameLeaguePassPlayPct'] = data['LeaguePassAtt'].rolling(6, min_periods=0).sum() / data['LeagueTotalPlays'].rolling(6, min_periods=0).sum()
    data['6GameLeague3rdDownPct'] = data['League3rdDownMade'].rolling(6, min_periods=0).sum() / data['League3rdDownAtt'].rolling(6, min_periods=0).sum()
    data['6GameLeague4thDownPct'] = data['League4thDownMade'].rolling(6, min_periods=0).sum() / data['League4thDownAtt'].rolling(6, min_periods=0).sum()
    
    # Season-long calcs
    data['SeasonLeagueRushPlayPct'] = data.groupby('Season')['LeagueRushAtt'].expanding().sum().values / data.groupby('Season')['LeagueTotalPlays'].expanding().sum().values
    data['SeasonLeaguePassPlayPct'] = data.groupby('Season')['LeaguePassAtt'].expanding().sum().values / data.groupby('Season')['LeagueTotalPlays'].expanding().sum().values
    data['SeasonLeague3rdDownPct']  = data.groupby('Season')['League3rdDownMade'].expanding().sum().values / data.groupby('Season')['League3rdDownAtt'].expanding().sum().values
    data['SeasonLeague4thDownPct']  = data.groupby('Season')['League4thDownMade'].expanding().sum().values / data.groupby('Season')['League4thDownAtt'].expanding().sum().values    
    # Season 3 Game calcs
    data['Season3GameLeagueRushPlayPct'] = data.groupby('Season')['LeagueRushAtt'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['LeagueTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GameLeaguePassPlayPct'] = data.groupby('Season')['LeaguePassAtt'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['LeagueTotalPlays'].rolling(3, min_periods=0).sum().values
    data['Season3GameLeague3rdDownPct'] = data.groupby('Season')['League3rdDownMade'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['League3rdDownAtt'].rolling(3, min_periods=0).sum().values
    data['Season3GameLeague4thDownPct'] = data.groupby('Season')['League4thDownMade'].rolling(3, min_periods=0).sum().values / data.groupby('Season')['League4thDownAtt'].rolling(3, min_periods=0).sum().values
    # Season 6 Game calcs 
    data['Season6GameLeagueRushPlayPct'] = data.groupby('Season')['LeagueRushAtt'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['LeagueTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GameLeaguePassPlayPct'] = data.groupby('Season')['LeaguePassAtt'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['LeagueTotalPlays'].rolling(6, min_periods=0).sum().values
    data['Season6GameLeague3rdDownPct'] = data.groupby('Season')['League3rdDownMade'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['League3rdDownAtt'].rolling(6, min_periods=0).sum().values
    data['Season6GameLeague4thDownPct'] = data.groupby('Season')['League4thDownMade'].rolling(6, min_periods=0).sum().values / data.groupby('Season')['League4thDownAtt'].rolling(6, min_periods=0).sum().values
    
    # Create lagged versions of the columns
    new_cols = data.columns
    weighted_cols = ['LeagueRushPlayPct', 'LeaguePassPlayPct', 'League3rdDownPct', 'League4thDownPct']
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
    stat_cols = LEAGUE_PLAYS_STAT_COLS

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
        # Calculate stats
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

    # **** league-plays specific 
    # Remove extra columns that were needed for other calculations 
    for token in LEAGUE_PLAYS_EXTRA_COLS:
        extra_cols = [i for i in agg_data.columns if token in i]
        agg_data.drop(extra_cols, inplace=True, axis=1)

    return agg_data

if __name__ == '__main__':

    stat_type = 'league-plays'
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

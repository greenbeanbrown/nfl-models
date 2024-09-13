import pandas as pd
import numpy as np
import os
from datetime import datetime
import sqlite3
import sys
#import mysql.connector
#from mysql.connector import Error
import sqlalchemy as db
import json
import pickle
import logging
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, TIMESTAMP, create_engine, ForeignKey, Index, case, func
from sqlalchemy.orm import sessionmaker, relationship
from xgboost import XGBRegressor
from requests import Session as APISession
import re
from scipy.stats import norm

sys.path.insert(0, '..')
from paths import DB_PATH, PLAYER_DATA_PATHS, TEAM_DATA_PATHS,  GAME_INFO_PATHS
from columns import RAW_PLAYER_STATS_COLS, PLAYER_PASSING_COLS, PLAYER_PASSING_BASE_STAT_COLS, \
                    PLAYER_RUSHING_COLS, PLAYER_RUSHING_BASE_STAT_COLS, \
                    PLAYER_RECEIVING_COLS, PLAYER_RECEIVING_BASE_STAT_COLS, \
                    PLAYER_DEFENSE_COLS, PLAYER_DEFENSE_STAT_COLS, \
                    PLAYER_KICKING_COLS, PLAYER_KICKING_STAT_COLS, \
                    PLAYER_SNAPS_COLS, PLAYER_SNAPS_STAT_COLS, \
                    PLAYER_POSITIONS_COLS, \
                    RAW_TEAM_STATS_COLS, TEAM_PASSING_STAT_COLS, TEAM_PASSING_COLS, \
                    TEAM_RUSHING_COLS, TEAM_RUSHING_STAT_COLS, \
                    TEAM_DEFENSE_COLS, TEAM_DEFENSE_STAT_COLS, \
                    TEAM_PLAYS_COLS, TEAM_PLAYS_STAT_COLS, \
                    TEAM_RECEIVING_COLS, TEAM_RECEIVING_STAT_COLS, \
                    LEAGUE_PASSING_COLS, LEAGUE_PASSING_STAT_COLS, \
                    LEAGUE_RUSHING_COLS, LEAGUE_RUSHING_STAT_COLS, \
                    LEAGUE_RECEIVING_COLS, LEAGUE_RECEIVING_STAT_COLS, \
                    LEAGUE_PLAYS_COLS, LEAGUE_PLAYS_STAT_COLS, \
                    TEAM_INFO_COLS, \
                    GAME_INFO_COLS
                    
from teams import TEAM_ABBRS, BDB_TEAM_ABBRS 
from name_exceptions import names, player_ids

sys.path.insert(0, '../../bet-app/code/')
from data_models import Price, Bet, Game


def read_model(model_dir, model_name):
    """
    """
    model_path = model_dir + model_name + '.pkl'
    features_path = model_dir + 'features/' + model_name + '.features'

    # Open the pickle file in binary read mode
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Open the .txt file in read mode and read the lines
    features = []
    with open(features_path, 'r') as file:
        for line in file:
            # Remove leading and trailing whitespace and append the line to the features list
            features.append(line.strip())

    return model, features

def odds_connect():
    """
    """
    logging.info('Establishing database connections..')
    # Stats
    #stats_conn = sqlite3.connect('../data/nfl.db')
    sqlite3.connect(DB_PATH)
    # Odds
    db_url = "mysql+pymysql://admin:chicken12@odds-rds-instance.cpogijhs7bep.us-west-2.rds.amazonaws.com/odds?ssl=us-west-2-bundle.pem"
    engine = create_engine(db_url, connect_args={'ssl': {'ca': './us-west-2-bundle.pem'}})
    Session = sessionmaker(bind=engine)
    db_session = Session
    odds_session = db_session()
    return  odds_session, engine

def nfl_connect():
    """
    Connect to the NFL database
    """
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    return conn

def get_opp_defense_data(conn, team_abbrs):
    """
    """
    sql_path = '../sql/latest_team_defense.sql'
    with open(sql_path, 'r') as sql_file:
        query = sql_file.read()    
    # Format the query to include placeholders for the team abbreviations
    placeholders = ','.join(['?'] * len(team_abbrs))
    query = query.replace('%s', placeholders)

    # Execute the query with the team abbreviations as parameters
    return pd.read_sql_query(query, conn, params=team_abbrs)

def get_latest_totals(session, engine):
    """
    """
    # Get the current UTC time
    current_utc_time = datetime.utcnow()
    # Query to retrieve the most recent Moneyline prices for each team from Pinnacle
    query = (
        session.query(
            Game.id.label('game_id'),  # Select game_id
            func.max(Bet.line).label('PointTotal')  # Use MAX to select one line per game
        )
        .join(Game, Bet.game_id == Game.id)  # Correct the join to reference the Game table only once
        .join(Price, Price.bet_id == Bet.id)  # Join with the Price table
        .filter(
            Bet.market_id == "Total Points",  # Filter for Total Points market
            Bet.sportsbook_id == "Pinnacle",
            Price.current_flag.is_(True),
            Price.is_main == 1,  # Additional filter for is_main = 1
            Game.league_id == "NFL",  # Filter for Game league
            Game.start_time >= current_utc_time  # Filter to remove results where Game.start_time is before current UTC time
        )
        .group_by(Game.id)  # Group by game_id to ensure unique records per game
    )
    # Convert query result to a DataFrame
    #df = pd.DataFrame(engine.connect().execute(query.statement))
    df = pd.read_sql_query(query.statement, engine)
    # Close the session
    session.close()    
    return df


def get_latest_moneylines(session, engine):
    """
    """
    # Get the current UTC time
    current_utc_time = datetime.utcnow()
    # Query to retrieve the most recent Moneyline prices for each team from Pinnacle
    query = (
        session.query(
            case(
                (Bet.bet == Game.home_team, Game.home_team_abbr),  # Calculate team_abbr
                else_=Game.away_team_abbr
            ).label('TeamAbbr'),
            Game.id.label('game_id'),
            Price.decimal_price.label('Moneyline'),  # Include decimal_price
            case(
                (Bet.bet == Game.home_team, 1),  # Calculate home_flag
                else_=0
            ).label('HomeFlag')
        )
        .join(Bet, Price.bet_id == Bet.id)
        .join(Game, Bet.game_id == Game.id)  # Join with the Game table
        .filter(
            Bet.market_id == "Moneyline",
            Bet.sportsbook_id == "Pinnacle",
            Price.current_flag.is_(True),
            Game.league_id == "NFL",  # Filter for Game league
            Game.start_time >= current_utc_time  # Filter to remove results where Game.start_time is before current UTC time
        )
    )
    # Convert query result to a DataFrame
    #df = pd.DataFrame(engine.connect().execute(query.statement))
    df = pd.read_sql_query(query.statement, engine)
    # Close the session
    session.close()    
    return df

def get_modeling_data(conn, target, lag):
    """
    Retrieve data with lagged features 
    for modeling projections
    """
    logging.info('Retrieving modeling data for stat: %s', target)
    if lag == True:
        sql_path = '../sql/{}/{}_backtest.sql'.format(target, target.lower(), target.lower())
    else:
        sql_path = '../sql/{}/{}_prod.sql'.format(target, target.lower(), target.lower())
        
    with open(sql_path, 'r') as sql_file:
        query = sql_file.read()

    modeling_data = pd.read_sql_query(query, conn)
    # Shouldn't impact anything on non-lagged data
    modeling_data.columns = [i.replace('Lag','') for i in modeling_data.columns]

    logging.info('Retrieved %d records of modeling data', len(modeling_data))

    return modeling_data

def agg_raw_files(stat_type):
    """
    Loops through each raw Excel data file
    and aggregates into a single CSV file
    """
    # Create an empty list to store dataframes from individual files
    dfs = []
    
    if 'player' in stat_type:
        data_dir = '../data/player-stats/raw/'
    elif ('team' in stat_type) or ('game' in stat_type):
        data_dir = '../data/team-stats/raw/'
    # List all files in the data directory
    files = [f for f in os.listdir(data_dir) if os.path.isfile(data_dir + f)]
    # Loop through files in the directory
    for file in files:
        # Check if the file is an Excel file
        if 'Agg' not in file:
            file_path = os.path.join(data_dir, file)
            # Read each Excel file into a dataframe and append to the list
            #df = pd.read_excel(file_path, header=2)
            raw_data = pd.read_csv(file_path, low_memory=False)
            # Do some basic data prep
            if 'player' in stat_type:
                # Rename and standardize raw columns
                renamed_data = rename_raw_cols(raw_data, stat_type=stat_type)
                clean_data = clean_player_data(renamed_data)
                # Derive season year and add it real quick
                season =  int(file.split('-')[1])
                output_filename = 'NFL-Agg-Player-BoxScore-Dataset.csv'
            elif ('team' in stat_type):
                # Rename and standardize raw columns
                renamed_data = rename_raw_cols(raw_data, stat_type=stat_type)
                # Clean up the raw renamed data
                clean_data = clean_team_data(renamed_data)
                # Derive season year and add it real quick
                season =  int(file.split('-')[0])
                output_filename = 'NFL-Agg-Team-BoxScore-Dataset.csv'
            elif ('game' in stat_type):
                # Rename and standardize raw columns
                renamed_data = rename_raw_cols(raw_data, stat_type=stat_type)
                # Clean up the raw renamed data
                clean_data = clean_game_data(renamed_data)
                # Derive season year and add it real quick
                season =  int(file.split('-')[0])                
                output_filename = 'NFL-Agg-Team-BoxScore-Dataset.csv'

            clean_data['Season'] = season
            dfs.append(clean_data)

    # Concatenate all dataframes into a single dataframe
    aggregated_df = pd.concat(dfs, ignore_index=True)
    # Create the output file path
    output_path = os.path.join(data_dir.replace('excel/',''), output_filename)
    # Persist the aggregated dataframe to an Excel file
    aggregated_df.to_csv(output_path, index=False)

    return

def read_raw_data(stat_type):
    """ 
    Constructs a data file path given 
    a couple parameters and then reads in 
    the raw excel file appropriately
    """
    # Setup file paths
    if 'player' in stat_type:
        data_dir = PLAYER_DATA_PATHS['raw-player-data-dir']
        file_name = 'NFL-Agg-Player-BoxScore-Dataset.csv'
        # Read raw excel file
        file_path = data_dir + file_name
        df = pd.read_csv(file_path, low_memory=False)
        # Drop duplicates - I've found some in there for whatever reason
        df = df.drop_duplicates()

    elif ('team' in stat_type) or ('game' in stat_type):
        data_dir = TEAM_DATA_PATHS['raw-team-data-dir']
        file_name = 'NFL-Agg-Team-BoxScore-Dataset.csv'        
        # Read raw excel file
        file_path = data_dir + file_name
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()

    return df


def get_data(stat_type):
    """
    Query the team tables to read in
    appropriate team data
    """
    conn = nfl_connect()
    # Get team stats
    if stat_type == 'league-passing':
        query = 'SELECT * FROM team_passing'
    elif stat_type == 'league-rushing':
        query = 'SELECT * FROM team_rushing'
    elif stat_type == 'league-receiving':
        query = 'SELECT * FROM team_receiving'        
    elif stat_type == 'league-plays':
        query = """
                SELECT tp.*, tr.TeamRushAtt, tpas.TeamPassAtt
                FROM team_plays AS tp
                JOIN team_rushing AS tr ON tp.GameId = tr.GameId
                JOIN team_passing AS tpas ON tp.GameId = tpas.GameId
                """
    # player-plays needs to be run AFTER team-plays
    elif stat_type == 'player-rushing':
        query = """
                SELECT 
                    tr.GameId,
                    tr.GameId2,
                    tr.GameDate,
                    tr.Season,
                    tr.TeamAbbr,
                    tr.TeamRushAtt,
                    tp.TeamTotalPlays
                FROM team_rushing AS tr
                JOIN team_plays AS tp
                    ON tr.GameDate = tp.GameDate
                    AND tr.GameId = tp.GameId
                    AND tr.GameId2 = tp.GameId2
                    AND tr.TeamAbbr = tp.TeamAbbr;
                """ 
    elif stat_type == 'player-receiving':
        query = """
                SELECT 
                    trec.GameId,
                    trec.GameId2,
                    trec.GameDate,
                    trec.Season,
                    trec.TeamAbbr,
                    tpas.TeamPassYards,
                    tpas.TeamPassComp,
                    tpas.TeamPassAtt,
                    tp.TeamTotalPlays
                FROM team_receiving AS trec
                JOIN team_plays AS tp
                    ON trec.GameDate = tp.GameDate
                    AND trec.GameId = tp.GameId
                    AND trec.GameId2 = tp.GameId2
                    AND trec.TeamAbbr = tp.TeamAbbr
                JOIN team_passing AS tpas
                    ON trec.GameDate = tpas.GameDate
                    AND trec.GameId = tpas.GameId
                    AND trec.GameId2 = tpas.GameId2
                    AND trec.TeamAbbr = tpas.TeamAbbr
                ;
                """ 
    elif stat_type == 'player-passing':
        query = """
                SELECT 
                    tpas.GameId,
                    tpas.GameId2,
                    tpas.GameDate,
                    tpas.Season,
                    tpas.TeamAbbr,
                    tpas.TeamPassYards,
                    tpas.TeamPassComp,
                    tpas.TeamPassAtt,
                    tp.TeamTotalPlays
                FROM team_passing AS tpas
                JOIN team_plays AS tp
                    ON tpas.GameDate = tp.GameDate
                    AND tpas.GameId = tp.GameId
                    AND tpas.GameId2 = tp.GameId2
                    AND tpas.TeamAbbr = tp.TeamAbbr
                ;
                """ 
    # We use a query for team-receiving because the Raw Team data doesn't
    # have any info about receiving data really
    #elif stat_type == 'team-receiving':
    #    query = 'SELECT * FROM player_receiving'        

    # Execute the sql query
    data = pd.read_sql_query(query, conn)

    return data

def rename_raw_cols(data, stat_type):
    """
    Re-name raw Excel file columns
    into something more manageable
    """
    # Re-name cols
    if 'player' in stat_type:
        df = data.rename(columns=RAW_PLAYER_STATS_COLS)
    elif 'team' in stat_type:
        if stat_type == 'team-receiving':
            #df = data.drop(['Player','PlayerId'], axis=1)
            df = data.copy()
            df.columns = ['Team' + i if i not in TEAM_INFO_COLS else i for i in df.columns]
        else:
            df = data.rename(columns=RAW_TEAM_STATS_COLS)
    elif 'league' in stat_type:
        data.columns = [i.replace('Team','League') for i in data.columns]
        df = data.copy()
    elif 'game' in stat_type:
        df = data.rename(columns=RAW_TEAM_STATS_COLS)

    return df

def split_lagged_data(combined_data, info_cols):
    """
    Splits the combined data into 
    current and lagged versions
    to make modeling easier
    """
    # Identify columns with and without lag
    non_lag_cols = info_cols + [i for i in combined_data.columns.difference(info_cols) if 'Lag' not in i]
    lag_cols = info_cols + [i for i in combined_data.columns.difference(info_cols) if 'Lag' in i]

    data = combined_data[non_lag_cols].copy()
    lag_data = combined_data[lag_cols].copy()

    return data, lag_data

def add_rolling_features(data, stat_cols, league=False):
    """
    Performs simple calculations on 
    stats to create rolling and lagged
    modeling features
    """
    rolling6_data = data[stat_cols].rolling(6, min_periods=0)
    rolling3_data = data[stat_cols].rolling(3, min_periods=0)
    
    player_dict = {}  # Dictionary to store column data for this player

    # Populate existing values into the dictionary    
    for col in data.columns:
        player_dict[col] = data[col].values
    
    for stat_col in stat_cols:
        if league == False:
            # Add a regular lag column
            player_dict[f'Lag{stat_col}'] = data[stat_col].shift(1)
            # Season Avg
            player_dict[f'SeasonAvg{stat_col}'] = np.array(data.groupby('Season')[stat_col].expanding().mean().reset_index(level=[0, 1], drop=True))
            player_dict[f'LagSeasonAvg{stat_col}'] = np.array(pd.Series(player_dict[f'SeasonAvg{stat_col}']).shift(1))
            # Season Median
            player_dict[f'SeasonMedian{stat_col}'] = np.array(data.groupby('Season')[stat_col].expanding().median().reset_index(level=[0, 1], drop=True))
            player_dict[f'LagSeasonMedian{stat_col}'] = np.array(pd.Series(player_dict[f'SeasonMedian{stat_col}']).shift(1))
            # Season Std
            player_dict[f'SeasonStd{stat_col}'] = np.array(data.groupby('Season')[stat_col].expanding().std().reset_index(level=[0, 1], drop=True))
            player_dict[f'LagSeasonStd{stat_col}'] = np.array(pd.Series(player_dict[f'SeasonStd{stat_col}']).shift(1))            
            # Career Avg
            player_dict[f'CareerAvg{stat_col}'] = data[stat_col].expanding().mean()
            player_dict[f'LagCareerAvg{stat_col}'] = player_dict[f'CareerAvg{stat_col}'].shift(1)
            # Career Median
            player_dict[f'CareerMedian{stat_col}'] = data[stat_col].expanding().median()
            player_dict[f'LagCareerMedian{stat_col}'] = player_dict[f'CareerMedian{stat_col}'].shift(1)
            # Career Std
            player_dict[f'CareerStd{stat_col}'] = data[stat_col].expanding().std()
            player_dict[f'LagCareerStd{stat_col}'] = player_dict[f'CareerStd{stat_col}'].shift(1)
            # 6 Game Avg
            player_dict[f'6GameAvg{stat_col}'] = rolling6_data[stat_col].mean()
            player_dict[f'Lag6GameAvg{stat_col}'] = player_dict[f'6GameAvg{stat_col}'].shift(1)
            # 6 Game Median
            player_dict[f'6GameMedian{stat_col}'] = rolling6_data[stat_col].median()
            player_dict[f'Lag6GameMedian{stat_col}'] = player_dict[f'6GameMedian{stat_col}'].shift(1)
            # 6 Game Std
            player_dict[f'6GameStd{stat_col}'] = rolling6_data[stat_col].std()
            player_dict[f'Lag6GameStd{stat_col}'] = player_dict[f'6GameStd{stat_col}'].shift(1)
            # 3 Game Avg
            player_dict[f'3GameAvg{stat_col}'] = rolling3_data[stat_col].mean()
            player_dict[f'Lag3GameAvg{stat_col}'] = player_dict[f'3GameAvg{stat_col}'].shift(1)
            # 3 Game Median
            player_dict[f'3GameMedian{stat_col}'] = rolling3_data[stat_col].median()
            player_dict[f'Lag3GameMedian{stat_col}'] = player_dict[f'3GameMedian{stat_col}'].shift(1)
            # 3 Game Std
            player_dict[f'3GameStd{stat_col}'] = rolling3_data[stat_col].std()
            player_dict[f'Lag3GameStd{stat_col}'] = player_dict[f'3GameStd{stat_col}'].shift(1)
        else:
            # Add a regular lag column
            player_dict[f'Lag{stat_col}'] = data[stat_col].shift(1)
            # Career Avg
            player_dict[f'CareerAvg{stat_col}'] = data[stat_col].expanding().sum() / (data['NumGames']*2).expanding().sum()
            player_dict[f'LagCareerAvg{stat_col}'] = player_dict[f'CareerAvg{stat_col}'].shift(1)
            # Career 3 Game Avg
            player_dict[f'3GameAvg{stat_col}'] = data[stat_col].expanding().sum() / (data['NumGames']*2).expanding().sum()
            player_dict[f'Lag3GameAvg{stat_col}'] = player_dict[f'3GameAvg{stat_col}'].shift(1)
            # Career 6 Game Avg
            player_dict[f'6GameAvg{stat_col}'] = data[stat_col].expanding().sum() / (data['NumGames']*2).expanding().sum()
            player_dict[f'Lag6GameAvg{stat_col}'] = player_dict[f'6GameAvg{stat_col}'] 
            # Season Avg
            player_dict[f'SeasonAvg{stat_col}'] = pd.Series(data.groupby('Season')[stat_col].expanding().sum().values / (data.groupby('Season')['NumGames'].expanding().sum().values*2))
            player_dict[f'LagSeasonAvg{stat_col}'] = player_dict[f'SeasonAvg{stat_col}'].shift(1)
            # Season 3 Game Avg
            player_dict[f'Season3GameAvg{stat_col}'] = pd.Series(data.groupby('Season')[stat_col].rolling(3, min_periods=0).sum().values / (data.groupby('Season')['NumGames'].rolling(3, min_periods=0).sum().values*2))
            player_dict[f'LagSeason3GameAvg{stat_col}'] = player_dict[f'Season3GameAvg{stat_col}'].shift(1)
            # Season 6 Game Avg
            player_dict[f'Season6GameAvg{stat_col}'] = pd.Series(data.groupby('Season')[stat_col].rolling(6, min_periods=0).sum().values / (data.groupby('Season')['NumGames'].rolling(6, min_periods=0).sum().values*2))
            player_dict[f'LagSeason6GameAvg{stat_col}'] = player_dict[f'Season6GameAvg{stat_col}'] 

    return player_dict

def agg_data(data_dir):
    """
    Aggregates prepped individual season
    data files into a single dataframe
    """
    # Get all eligible filenames
    filenames = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and ('lag' not in f)]
    lag_filenames = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and ('lag' in f)]
    # This dataframe will hold everything
    agg_df = pd.DataFrame()
    lag_agg_df = pd.DataFrame()
    # Loop through each season's data file and concat into a master
    for filename in filenames:
        file_path = data_dir + filename
        df = pd.read_csv(file_path)
        agg_df = pd.concat([agg_df, df])
        agg_df.reset_index(inplace=True, drop=True)

    for lag_filename in lag_filenames:
        lag_file_path = data_dir + lag_filename
        lag_df = pd.read_csv(lag_file_path)
        lag_agg_df = pd.concat([lag_agg_df, lag_df])        
        lag_agg_df.reset_index(inplace=True, drop=True)

    return agg_df, lag_agg_df

def export_data(data, lag_data, data_dir, stat_type):
    """
    Persist lagged and non-lagged
    data
    """

    file_name = '{}_data.csv'.format(stat_type)
    file_path = data_dir + file_name
    data.to_csv(file_path, index=False)

    if type(lag_data) == pd.DataFrame:
        lag_file_name = 'lag_' + file_name
        lag_file_path = data_dir + lag_file_name
        lag_data.to_csv(lag_file_path, index=False)

    return

def load_data(data_dir, stat_type):
    """
    Aggregates prepped individual season
    data files and then loads into nfl.db
    """
    # Connect to database
    #db_path = DB_PATH
    #conn = sqlite3.connect(db_path)
    engine = nfl_connect()
    # Aggregate data
    data, lag_data = agg_data(data_dir)
    # Push data via UPDATE
    if stat_type == 'player-passing':
        data.to_sql('player_passing', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_player_passing', engine, if_exists='replace', index=False)
    elif stat_type == 'player-rushing':
        data.to_sql('player_rushing', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_player_rushing', engine, if_exists='replace', index=False)
    elif stat_type == 'player-receiving':
        data.to_sql('player_receiving', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_player_receiving', engine, if_exists='replace', index=False)
    elif stat_type == 'player-defense':
        data.to_sql('player_defense', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_player_defense', engine, if_exists='replace', index=False)    
    elif stat_type == 'player-kicking':
        data.to_sql('player_kicking', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_player_kicking', engine, if_exists='replace', index=False)    
    elif stat_type == 'player-rushing-plays':
        data.to_sql('player_rushing_plays', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_player_rushing_plays', engine, if_exists='replace', index=False)
    elif stat_type == 'player-positions':
        data.to_sql('player_positions', engine, if_exists='replace', index=False)

    elif stat_type == 'team-passing':
        data.to_sql('team_passing', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_team_passing', engine, if_exists='replace', index=False)    
    elif stat_type == 'team-rushing':
        data.to_sql('team_rushing', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_team_rushing', engine, if_exists='replace', index=False)   
    elif stat_type == 'team-receiving':
        data.to_sql('team_receiving', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_team_receiving', engine, if_exists='replace', index=False)           
    elif stat_type == 'team-defense':
        data.to_sql('team_defense', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_team_defense', engine, if_exists='replace', index=False)   
    elif stat_type == 'team-plays':
        data.to_sql('team_plays', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_team_plays', engine, if_exists='replace', index=False)   

    elif stat_type == 'league-passing':
        data.to_sql('league_passing', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_league_passing', engine, if_exists='replace', index=False)
    elif stat_type == 'league-rushing':
        data.to_sql('league_rushing', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_league_rushing', engine, if_exists='replace', index=False)
    elif stat_type == 'league-receiving':
        data.to_sql('league_receiving', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_league_receiving', engine, if_exists='replace', index=False)
    elif stat_type == 'league-plays':
        data.to_sql('league_plays', engine, if_exists='replace', index=False)
        lag_data.to_sql('lag_league_plays', engine, if_exists='replace', index=False)

    elif stat_type == 'game-info':
        data.to_sql('game_info', engine, if_exists='replace', index=False)

    return

def filter_sort_data(player_data, stat_type='player-passing'):
    """
    Filters the data down to columns
    relevant to the given stat_type and
    then sorts it chronologically by player
    """
    data = player_data.copy()
    # Filter columns
    if stat_type == 'player-passing':
        cols = PLAYER_PASSING_COLS
        stat_cols = PLAYER_PASSING_BASE_STAT_COLS
    elif stat_type == 'player-rushing':
        cols = PLAYER_RUSHING_COLS
        stat_cols = PLAYER_RUSHING_BASE_STAT_COLS
    elif stat_type == 'player-receiving':
        cols = PLAYER_RECEIVING_COLS
        stat_cols = PLAYER_RECEIVING_BASE_STAT_COLS
    elif stat_type == 'player-defense':
        cols = PLAYER_DEFENSE_COLS
        stat_cols = PLAYER_DEFENSE_STAT_COLS        
    elif stat_type == 'player-kicking':
        cols = PLAYER_KICKING_COLS
        stat_cols = PLAYER_KICKING_STAT_COLS        
    elif stat_type == 'player-snaps':
        cols = PLAYER_SNAPS_COLS
        stat_cols = PLAYER_SNAPS_STAT_COLS   
    elif stat_type == 'player-positions':
        cols = PLAYER_POSITIONS_COLS
        stat_cols = PLAYER_POSITIONS_COLS

    elif stat_type == 'team-passing':
        cols = TEAM_PASSING_COLS
        stat_cols = TEAM_PASSING_STAT_COLS
    elif stat_type == 'team-rushing':
        cols = TEAM_RUSHING_COLS
        stat_cols = TEAM_RUSHING_STAT_COLS
    elif stat_type == 'team-receiving':
        cols = TEAM_RECEIVING_COLS
        stat_cols = TEAM_RECEIVING_STAT_COLS        
    elif stat_type == 'team-defense':
        cols = TEAM_DEFENSE_COLS
        stat_cols = TEAM_DEFENSE_STAT_COLS
    elif stat_type == 'team-plays':
        cols = TEAM_PLAYS_COLS
        stat_cols = TEAM_PLAYS_STAT_COLS

    elif stat_type == 'league-passing':
        cols = LEAGUE_PASSING_COLS
        stat_cols = LEAGUE_PASSING_STAT_COLS
    elif stat_type == 'league-rushing':
        cols = LEAGUE_RUSHING_COLS
        stat_cols = LEAGUE_RUSHING_STAT_COLS
    elif stat_type == 'league-receiving':
        cols = LEAGUE_RECEIVING_COLS
        stat_cols = LEAGUE_RECEIVING_STAT_COLS        
    elif stat_type == 'league-plays':
        cols = LEAGUE_PLAYS_COLS
        stat_cols = LEAGUE_PLAYS_STAT_COLS

    elif stat_type == 'game-info':
        cols = GAME_INFO_COLS
        stat_cols = GAME_INFO_COLS

    # Trim down to certain cols
    try:
        trimmed_data = data[cols]
    except:
        import ipdb; ipdb.set_trace()

    # Remove records where everything is 0
    if stat_type == 'player-positions':
        sorted_filtered_data = trimmed_data.sort_values(['Player','GameDate'])
        sorted_filtered_data.reset_index(inplace=True, drop=True)
    elif stat_type == 'game-info':
        sorted_filtered_data = trimmed_data.sort_values(['GameDate'])
        sorted_filtered_data.reset_index(inplace=True, drop=True)
    else:
        filtered_data = trimmed_data[trimmed_data[stat_cols].sum(axis=1) != 0]
        # Sort the filtered data
        if 'player' in stat_type:
            sorted_filtered_data = filtered_data.sort_values(['Player','GameDate'])
        elif 'team' in stat_type:
            sorted_filtered_data = filtered_data.sort_values(['TeamAbbr','GameDate'])
        elif 'league' in stat_type:
            sorted_filtered_data = filtered_data.sort_values(['GameDate'])

        sorted_filtered_data.reset_index(inplace=True, drop=True)

    return sorted_filtered_data

def clean_player_data(raw_data):
    """
    Perform some basic data prep
    to the raw Excel data
    """
    data = raw_data.copy()
    # Add Team Abbreviations
    data['TeamAbbr'] = [TEAM_ABBRS.get(i) for i in data['Team']]
    ## Add Home/Away teams
    def add_home_away_abbrs(game_id):
        """
        Used for replacing the BDB team abbr format
        with a standardized one
        """
        matchup = game_id.split('-')[1]
        bdb_away_abbr, bdb_home_abbr = matchup.split('@')
        away_abbr = BDB_TEAM_ABBRS.get(bdb_away_abbr)
        home_abbr = BDB_TEAM_ABBRS.get(bdb_home_abbr)
        return away_abbr, home_abbr
    
    abbrs_list = data['GameId'].apply(lambda x: add_home_away_abbrs(x)).tolist()
    #away_abbrs, home_abbrs = pd.DataFrame(abbrs_list, index=data.index)

    away_abbrs = [item[0] for item in abbrs_list]
    home_abbrs = [item[1] for item in abbrs_list]

    # Standardize dates
    game_dates_dt = [datetime.strptime(i, '%m/%d/%Y') for i in data['GameDate']]
    game_dates_str = [datetime.strftime(i, '%Y-%m-%d') for i in game_dates_dt]
    data['GameDate'] = game_dates_str
    # GameId2
    data['GameId2'] = [date + '-' + away_abbr + '-' + home_abbr 
               for date, away_abbr, home_abbr 
               in zip(game_dates_str, away_abbrs, home_abbrs)]

    def map_player_ids(df):
        """
        Reads in the player ID lookup
        file (manually generated) and 
        uses it to map the older players
        PlayerIds to valid PlayerIds which
        match the current/future convention
        """
        # Read the lookup CSV
        lookup_df = pd.read_csv('old_playerid_lookup.csv')
        # Create a mapping from OriginalId to PlayerId
        orig_id_mapping = dict(zip(lookup_df['OriginalId'], lookup_df['PlayerId']))
        # Map the PlayerId column in the lookup to OriginalId 
        # and update it with PlayerId values
        player_ids = df['PlayerId'].replace(orig_id_mapping)
        return player_ids
    
    # Handle odd cases where OffSnaps is NaN, but a player has a 1+ PassAtt or RushAtt
    data['OffSnaps'] = data['OffSnaps'].replace([np.inf, -np.inf], 0)
    data['DefSnaps'] = data['DefSnaps'].replace([np.inf, -np.inf], 0)

    # Handle PlayerId exceptions (only applies to older PlayerIds)
    data['PlayerId'] = map_player_ids(data)
    # Handle known player name exceptions
    data['Player'] = data['Player'].replace(names)
    data['PlayerId'] = data['PlayerId'].replace(player_ids)

    # Handle known data exceptions
    # 2023 - both Michael Carter's pulling in as the RB
    data.loc[(data['PlayerId'] == 'michael-carter-3') & (data['DefSnaps'] > 0), 'Position'] = 'CB'
    data.loc[(data['PlayerId'] == 'michael-carter-3') & (data['DefSnaps'] > 0), 'PlayerId'] = 'michael-carter-ii'

    # Drop unneeded columns
    if 'Unnamed: 75' in data.columns:
        data.drop('Unnamed: 75', axis=1, inplace=True)

    return data

def clean_team_data(raw_data):
    """
    Perform some basic data prep
    to the raw Excel data
    """
    data = raw_data.copy()
    # GameType flag (Reg Season or Playoffs)
    data['GameType'] = ['R' if 'Regular' in i else 'P' for i in data['DatasetName']]
    # Add Team Abbreviations
    data['TeamAbbr'] = [TEAM_ABBRS.get(i) for i in data['Team']]
    # Derive opponent
    opps = []
    for idx in data['TeamAbbr'].index:
        if idx % 2 == 0:
            opp = data['TeamAbbr'].iloc[idx+1]
            opps.append(opp)
        else:
            opp = data['TeamAbbr'].iloc[idx-1]
            opps.append(opp)
    data['OppAbbr'] = opps

    # Add season year as a column
    #data['Season'] = year

    # Home and Away Flags
    data['HomeFlag'] = [1 if i == 'Home' else 0 for i in data['Venue']]
    data['RoadFlag'] = [1 if i == 'Road' else 0 for i in data['Venue']]
    data['NeutralFlag'] = [1 if i == 'Neutral' else 0 for i in data['Venue']]

    # Add Home/Away teams
    def add_home_away_abbrs(game_id):
        """
        Used for replacing the BDB team abbr format
        with a standardized one
        """
        matchup = game_id.split('-')[1]
        bdb_away_abbr, bdb_home_abbr = matchup.split('@')
        away_abbr = BDB_TEAM_ABBRS.get(bdb_away_abbr)
        home_abbr = BDB_TEAM_ABBRS.get(bdb_home_abbr)
        return away_abbr, home_abbr
    
    abbrs_list = data['GameId'].apply(lambda x: add_home_away_abbrs(x)).tolist()
    #away_abbrs, home_abbrs = pd.DataFrame(abbrs_list, index=data.index)
    away_abbrs = [item[0] for item in abbrs_list]
    home_abbrs = [item[1] for item in abbrs_list]

    # Standardize dates
    game_dates_dt = [datetime.strptime(i, '%m/%d/%Y') for i in data['GameDate']]
    game_dates_str = [datetime.strftime(i, '%Y-%m-%d') for i in game_dates_dt]
    data['GameDate'] = game_dates_str
    # GameId2
    data['GameId2'] = [date + '-' + away_abbr + '-' + home_abbr 
               for date, away_abbr, home_abbr 
               in zip(game_dates_str, away_abbrs, home_abbrs)]
    
    if 'Unnamed: 62' in data.columns:
        data.drop('Unnamed: 62', axis=1, inplace=True)

    return data

def clean_game_data(raw_data):
    """
    Perform some basic data prep
    to the raw Excel data
    """
    data = raw_data.copy()
    # GameType flag (Reg Season or Playoffs)
    data['GameType'] = ['R' if 'Regular' in i else 'P' for i in data['DatasetName']]
    # Add Team Abbreviations
    data['TeamAbbr'] = [TEAM_ABBRS.get(i) for i in data['Team']]
    # Derive opponent
    opps = []
    for idx in data['TeamAbbr'].index:
        if idx % 2 == 0:
            opp = data['TeamAbbr'].iloc[idx+1]
            opps.append(opp)
        else:
            opp = data['TeamAbbr'].iloc[idx-1]
            opps.append(opp)
    data['OppAbbr'] = opps

    # Home and Away Flags
    data['HomeFlag'] = [1 if i == 'Home' else 0 for i in data['Venue']]
    data['RoadFlag'] = [1 if i == 'Road' else 0 for i in data['Venue']]
    data['NeutralFlag'] = [1 if i == 'Neutral' else 0 for i in data['Venue']]

    # Add Home/Away teams
    def add_home_away_abbrs(game_id):
        """
        Used for replacing the BDB team abbr format
        with a standardized one
        """
        matchup = game_id.split('-')[1]
        bdb_away_abbr, bdb_home_abbr = matchup.split('@')
        away_abbr = BDB_TEAM_ABBRS.get(bdb_away_abbr)
        home_abbr = BDB_TEAM_ABBRS.get(bdb_home_abbr)
        return away_abbr, home_abbr
    
    abbrs_list = data['GameId'].apply(lambda x: add_home_away_abbrs(x)).tolist()
    data[['AwayAbbr', 'HomeAbbr']] = pd.DataFrame(abbrs_list, index=data.index)
    # Standardize dates
    game_dates_dt = [datetime.strptime(i, '%m/%d/%Y') for i in data['GameDate']]
    game_dates_str = [datetime.strftime(i, '%Y-%m-%d') for i in game_dates_dt]
    data['GameDate'] = game_dates_str
    # GameId2
    data['GameId2'] = [date + '-' + away_abbr + '-' + home_abbr 
                   for date, away_abbr, home_abbr 
                   in zip(game_dates_str, data.AwayAbbr, data.HomeAbbr)]
    

    # Manually correct known invalid data
    data.loc[data.GameId == '43723-CHI@DEN', 'OpeningTotal'] = 41.5

    data.loc[(data.GameId == '43723-PHI@ATL') & (data.TeamAbbr == 'ATL'), 'OpeningSpread'] = -1
    data.loc[(data.GameId == '43723-PHI@ATL') & (data.TeamAbbr == 'PHI'), 'OpeningSpread'] = 1
    data.loc[(data.GameId == '43723-PHI@ATL'), 'OpeningTotal'] = 51
    data.loc[(data.GameId == '43723-PHI@ATL'), 'OpeningTotal'] = 51
    
    data.loc[(data.GameId == '43723-BUF@NYG') & (data.TeamAbbr == 'BUF'), 'OpeningSpread'] = 2.5
    data.loc[(data.GameId == '43723-BUF@NYG') & (data.TeamAbbr == 'NYG'), 'OpeningSpread'] = -2.5   
    data.loc[(data.GameId == '43723-BUF@NYG'), 'OpeningTotal'] = 42.5
    data.loc[(data.GameId == '43723-BUF@NYG'), 'OpeningTotal'] = 42.5

    data.loc[(data.GameId == '43804-DAL@CHI'), 'OpeningTotal'] = 43
    data.loc[(data.GameId == '43804-DAL@CHI'), 'OpeningTotal'] = 43

    data['ClosingMoneyline'] = [int(str(i).replace('even', '100')) for i in data['ClosingMoneyline']]

    # Drop unnecessary column if necessary
    if 'Unnamed: 62' in data.columns:
        data.drop('Unnamed: 62', axis=1, inplace=True)
    if 'TeamFirstDowns' in data.columns:
        data.drop('TeamFirstDowns', axis=1, inplace=True)
    if 'StartTime' in data.columns:
        data.drop('StartTime', axis=1, inplace=True)

    return data

def find_nearest_neighbors(df, k, projection_col, std_estimation, lookback_years=None):
    """
    Finds the n closest players projections
    for each player in the dataframe
    """
    if std_estimation == 'k-nearest-season-projection':
        # Need to filter Seasons here differently for backtesting vs prod runs
        df = df[df['Season'].isin(lookback_years)]

    cohort_column = f"{projection_col}Cohort"
    cohort_proj_column = f"{projection_col}CohortProj"
    df[cohort_column] = None
    df[cohort_proj_column] = None

    game_dates = pd.to_datetime(df['GameDate'])
    for idx in df.index:
        player_id = df.at[idx, 'PlayerId']
        player_stat_value = df.at[idx, projection_col]
        position = df.at[idx, 'Position']
        # Calculate the date X days before the game_date
        game_date = game_dates[idx]
        start_date = game_date - pd.Timedelta(days=5)
        # Filter down to current position
        working_df = df[df['Position'] == position].copy()
        # Filter to current weeks games based on input
        if std_estimation == 'k-nearest-current-projection':
            working_df = working_df[(game_dates >= start_date) & (game_dates <= game_date)].copy()
            
        # Calculate the absolute differences between the current player's stat value and all other players' stat values
        working_df['diff'] = (working_df[projection_col] - player_stat_value) ** 2
        # Sort by the difference and exclude the current player
        closest_players = working_df[working_df['PlayerId'] != player_id].sort_values(by='diff').drop_duplicates(subset='PlayerId').head(k)['PlayerId'].tolist()
        closest_projections = working_df[working_df['PlayerId'] != player_id].sort_values(by='diff').drop_duplicates(subset='PlayerId').head(k)[projection_col].tolist()

        cohort = [*closest_players, *[player_id]]
        # Assign the closest players' PlayerIds to the new column
        df.at[idx, cohort_column] = cohort
        df.at[idx, cohort_proj_column] = closest_projections

    # Drop the temporary 'diff' column
    #df.drop(columns=['diff'], inplace=True)

    return df

def estimate_std(df, target, std_estimation, k, lookback_years=None):
    """
    """
    def pull_historical_stats(target, player_ids, table, game_date_before):
        """
        """
        #db_path = 'D:/nfl-models/data/nfl.db'
        db_path = '../data/nfl.db'
        conn = sqlite3.connect(db_path)
        # Convert list of player_ids to a comma-separated string
        player_ids_str = ','.join(f"'{player_id}'" for player_id in player_ids)
        # SQL query with an additional WHERE clause to filter records prior to the given game date
        query = f"""
        SELECT {target} 
        FROM {table} 
        WHERE PlayerId IN ({player_ids_str}) 
        AND GameDate < '{game_date_before}'
        """
        return pd.read_sql_query(query, conn)

    def create_cohort_samples(df, table, k):
        # List of columns to calculate the sample for
        cohort_column = f"x{target}Cohort"
        sample_column = f"x{target}Sample"
        df[sample_column] = None

        for idx in df.index:
            game_date = df.at[idx, 'GameDate']
            player_ids = df.at[idx, cohort_column]
            # Call pull_historical_stats to get the historical stats for the 5 closest players
            historical_stats = pull_historical_stats(target, player_ids, table, game_date)
            # Convert the historical stats to a list and assign to the new column
            df.at[idx, sample_column] = historical_stats[target].tolist()

        # Calculate the standard deviation and number of observations for each sample
        #df = calc_sample_metrics(df)

        return df
    
    table = 'player_passing' if 'Pass' in target else 'player_rushing' if 'Rush' in target else 'player_receiving'

    sample_col = 'x{}Sample'.format(target)
    logging.info('Creating historical sample of %s', target)
    # Apply the function and create the new dataframe
    if std_estimation == 'player':
        df[sample_col] = df.apply(lambda row: pull_historical_stats(target, [row['PlayerId']], table, row['GameDate'])[target].tolist(), axis=1)

    elif std_estimation in ['k-nearest-season-projection']:
        df = find_nearest_neighbors(df, k, 'x{}'.format(target), std_estimation, lookback_years)
        df = create_cohort_samples(df.copy(), table, k)

    elif std_estimation == ['k-nearest-historical-projection', 'k-nearest-current-projection']:
        df = find_nearest_neighbors(df, k, 'x{}'.format(target), std_estimation)
        df = create_cohort_samples(df, table, k)

    # Extract the relevant columns
    #filtered_data = df[['PlayerId', 'GameDate', stat_col, sample_col]].copy()
    #filtered_data = df[['PlayerId', 'Player', 'GameDate', stat_col, sample_col]].copy()
    #filtered_data = sorted_data[['PlayerId', 'GameDate', stat_col]].copy()
    
    # Calculate final param
    df.loc[:, 'NumObs'] = df[sample_col].apply(len)
    df.loc[:, '{}Std'.format(target)] = df[sample_col].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)

    return df

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
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, TIMESTAMP, create_engine, ForeignKey, Index
from sqlalchemy.orm import sessionmaker, relationship
from xgboost import XGBRegressor
from requests import Session as APISession
import re
from scipy.stats import norm

#### BACKTEST FUNCTIONS ####

def convert_to_decimal(price):
    # Convert American Odds to Decimal Odds
    if price > 0:
        decimal_price = 1 + (price / 100)
    else:
        decimal_price = 1 - (100 / price)
    return decimal_price

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def create_player_id(name):
    # Lowercase all characters
    name = name.lower()
    # Replace all spaces and non-letter characters with a single dash
    name = re.sub(r'[^a-z]+', '-', name)
    # Remove consecutive dashes
    name = re.sub(r'-+', '-', name)
    # Strip any leading or trailing dashes
    name = name.strip('-')
    return name

def setup_db_connection():
    """
    """
    logging.info('Establishing database connections..')
    # Stats
    stats_conn = sqlite3.connect('../data/nfl.db')
    # Odds
    db_url = "mysql+pymysql://admin:chicken12@odds-rds-instance.cpogijhs7bep.us-west-2.rds.amazonaws.com/odds?ssl=us-west-2-bundle.pem"
    engine = create_engine(db_url, connect_args={'ssl': {'ca': '../us-west-2-bundle.pem'}})
    Session = sessionmaker(bind=engine)
    db_session = Session
    odds_session = db_session()
    return stats_conn, odds_session

def get_modeling_data(conn, target):
    """
    Retrieve data with lagged features 
    for modeling projections
    """
    logging.info('Retrieving modeling data for stat: %s', target)
    
    sql_path = '../sql/{}/{}_backtest.sql'.format(target, target.lower(), target.lower())
    with open(sql_path, 'r') as sql_file:
        query = sql_file.read()

    modeling_data = pd.read_sql_query(query, conn)
    modeling_data.columns = [i.replace('Lag','') for i in modeling_data.columns]

    logging.info('Retrieved %d records of modeling data', len(modeling_data))

    return modeling_data

def find_nearest_neighbors(df, k, projection_col, std_estimation):
    """
    Finds the n closest players projections
    for each player in the dataframe
    """
    if std_estimation == 'k-nearest-season-projection':
        df = df[df['Season'] == 2023]

    cohort_column = f"{projection_col}Cohort"
    cohort_proj_column = f"{projection_col}CohortProj"
    df[cohort_column] = None
    df[cohort_proj_column] = None

    game_dates = pd.to_datetime(df['GameDate'])
    for idx in df.index:
        player_id = df.at[idx, 'PlayerId']
        player_stat_value = df.at[idx, projection_col]
        # Calculate the date X days before the game_date
        game_date = game_dates[idx]
        start_date = game_date - pd.Timedelta(days=5)
        
        if std_estimation == 'k-nearest-week-projection':
            working_df = df[(game_dates >= start_date) & (game_dates <= game_date)].copy()
        else:
            working_df = df.copy()
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

def estimate_std(df, target, std_estimation, k):
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

    def calc_sample_metrics(df):
        std_col = f"{target}Std"
        obs_col = f"{target}Obs"
        # Calculate the standard deviation and number of observations
        df[std_col] = df[sample_col].apply(lambda x: np.std(x) if isinstance(x, list) else np.nan)
        df[obs_col] = df[sample_col].apply(lambda x: len(x) if isinstance(x, list) else 0)

        return df

    # Function to create a list of historical stat up to each GameDate
    def create_player_sample(group):
        historical_data = []
        for index, row in group.iterrows():
            # Include all assists from rows with an earlier GameDate
            historical = [x for x in group[group['GameDate'] < row['GameDate']][target].tolist() if pd.notna(x)]
            historical_data.append(historical)
        return pd.DataFrame({sample_col: historical_data}, index=group.index)


    def create_cohort_samples(df, k):
        # List of columns to calculate the sample for
        cohort_column = f"x{target}Cohort"
        sample_column = f"x{target}Sample"
        df[sample_column] = None
        # Determine which table to pull from
        table = 'player_passing' if 'Pass' in target else 'player_rushing' if 'Rush' in target else 'player_receiving'

        for idx in df.index:
            game_date = df.at[idx, 'GameDate']
            player_ids = df.at[idx, cohort_column]
            # Call pull_historical_stats to get the historical stats for the 5 closest players
            historical_stats = pull_historical_stats(target, player_ids, table, game_date)
            # Convert the historical stats to a list and assign to the new column
            df.at[idx, sample_column] = historical_stats[target].tolist()

        # Calculate the standard deviation and number of observations for each sample
        df = calc_sample_metrics(df)

        return df

    sample_col = 'x{}Sample'.format(target)
    logging.info('Creating historical sample of %s', target)
    # Apply the function and create the new dataframe
    if std_estimation == 'player':
        df[sample_col] = df.groupby(['PlayerId', 'Player']).apply(create_player_sample).reset_index(level=[0,1], drop=True)
        #df[sample_col] = df.groupby('PlayerId').apply(create_player_sample).reset_index(level=[0,1], drop=True)
    elif std_estimation in ['k-nearest-season-projection','k-nearest-historical-projection','k-nearest-week-projection']:
        df = find_nearest_neighbors(df, k, 'x{}'.format(target), std_estimation)
        df = create_cohort_samples(df, k)

    # Extract the relevant columns
    #filtered_data = df[['PlayerId', 'GameDate', stat_col, sample_col]].copy()
    #filtered_data = df[['PlayerId', 'Player', 'GameDate', stat_col, sample_col]].copy()
    #filtered_data = sorted_data[['PlayerId', 'GameDate', stat_col]].copy()
    
    # Calculate final param
    df.loc[:, 'NumObs'] = df[sample_col].apply(len)
    df.loc[:, 'SampleStd'] = df[sample_col].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)

    return df


def get_results_data(conn, target, backtest_year):
    """
    """
    logging.info('Retrieving historical data for stat type: %s', target)

    current_season = backtest_year + 1

    if target == 'PassYards':
        table_name = 'player_passing'
    elif target == 'RushYards':
        table_name = 'player_rushing'
    else:
        import ipdb; ipdb.set_trace()

    query = "SELECT GameDate, PlayerId, Player, {} FROM {} WHERE Season NOT IN ({})".format('{}'.format(target), table_name, current_season)
    raw_historical_data = pd.read_sql_query(query, conn)
    logging.info('Retrieved %d records of historical data', len(raw_historical_data))

    # Remove Lag from column name
    historical_data = raw_historical_data.copy()
    historical_data.columns = [i.replace('Lag','') for i in historical_data.columns]

    # Sort the data by PlayerId, Player, and GameDate
    sorted_data = historical_data.sort_values(['PlayerId', 'Player', 'GameDate'])

    return sorted_data

def get_oj_game_ids(game_ids):
    """
    Map OJ game_ids to game start dates
    """
    endpoint = 'https://api-external.oddsjam.com/api/v2/scores'
    params = {'key':'6e9259fe-cfa5-4dab-8f0e-fefaf98f696d',
              'league':'NFL',
              'game_id': game_ids}
    
    api_session = APISession()

    response = api_session.get(endpoint, params=params)
    raw_data = pd.DataFrame(response.json()['data'])
    
    cols = ['game_id','start_date']
    data = raw_data[cols]
    return data

def prep_odds(raw_df, target, sportsbooks):
    """
    """
    # Determine market name
    if target == 'PassYards':
        market = 'Player Passing Yards'
    elif target == 'RushYards':
        market = 'Player Rushing Yards'
    else:
        import ipdb; ipdb.set_trace()

    # Filter rows based on 'market'
    df = raw_df[raw_df['market'] == market].copy()

    if type(sportsbooks) == str:
        df = df[df['sportsbook'] == sportsbooks]
    else:
        df = df[df['sportsbook'].isin(sportsbooks)]

    # Extract name, over_under, and line from the 'bet' column
    #df['side'] = ['Over' if ' Over ' in i else ' Under ' for i in df['bet']]
    df[['name', 'side', 'line']] = df['bet'].str.extract(r'(.+)\s(Over|Under)\s(.+)')
    
    # Drop rows with NaN in 'name'
    df = df.dropna(subset=['name'])
    if df.empty:
        return df

    # Define a function to extract opening and closing prices
    def extract_prices(values):
        opening_price = values[0]['price']
        opening_timestamp = values[0]['timestamp']
        closing_price = values[-1]['price']
        closing_timestamp = values[-1]['timestamp']
        return pd.Series([opening_timestamp, opening_price, closing_timestamp, closing_price,
                          convert_to_decimal(opening_price) - convert_to_decimal(closing_price)])

    # Apply the function to the 'values' column
    df[['open_timestamp','opening','close_timestamp', 'closing', 'line_move']] = df['values'].copy().apply(extract_prices)

    # Clean-up and convert types as needed
    df['name'] = df['name'].str.strip()
    df['line'] = df['line'].astype(float)
    # Get start dates
    game_ids = df['game_id'].unique()
    game_id_df = get_oj_game_ids(game_ids)
    # Merge in game dates
    merged_df = pd.merge(df, game_id_df, on='game_id')
    merged_df['game_date'] = merged_df['start_date'].str[:10] # 2023-11-03
    # Drop values column now
    merged_df.drop('values', axis=1, inplace=True)

    return merged_df

def get_historical_odds(target, sportsbooks):
    """
    """
    logging.info('Retrieving historical odds data')
    # Path to .pickle files
    historical_odds_dir = '../data/historical-odds/'
    pickle_dir = historical_odds_dir + '/pickles/'
    agg_df = pd.DataFrame()
    # Loop through each pickle file and prep it
    logging.info('Prepping pickle odds')
    for filename in os.listdir(pickle_dir):
        # LOG: print(filename)
        # Read in raw pickle data
        file_path = os.path.join(pickle_dir, filename)
        #logging.info('Prepping pickle odds: %s', file_path)
        raw_pickle_df = pd.read_pickle(file_path)
        raw_pickle_df.reset_index(inplace=True, drop=True)
        # Prepare data
        historical_odds_df = prep_odds(raw_pickle_df, target=target, sportsbooks=sportsbooks)

        if not historical_odds_df.empty:
            # Get last record for each unique bet (to ensure accurate closing odds, otherwise it gets weird)
            cols = ['name','game_date','bet','sportsbook']
            idx = historical_odds_df.groupby(cols)['close_timestamp'].idxmax()
            final_df = historical_odds_df.loc[idx].reset_index(drop=True)   
            agg_df = pd.concat([agg_df, final_df.drop_duplicates()])
    
    # Drop duplicates again
    agg_df.drop_duplicates(inplace=True)
    # Try to filter out odds which aren't valid at close (it throws off results heavily)
    final_df = filter_closing_odds(agg_df)

    return final_df

def filter_closing_odds(df):
    """
    Filter the dataframe to keep records where the close timestamp is within 30 minutes prior to the start date.
    """
    # Convert 'start_date' to a datetime object with timezone awareness (UTC)
    df['start_datetime'] = pd.to_datetime(df['start_date']).dt.tz_convert('UTC')
    # Convert 'close_timestamp' to datetime format with timezone awareness (UTC)
    df['close_datetime'] = pd.to_datetime(df['close_timestamp'], unit='ms', utc=True)
    
    # Filter the dataframe to keep records with close_timestamp within 30 minutes before start_date
    filtered_odds_data = df[
        (df['close_datetime'] >= (df['start_datetime'] - pd.Timedelta(minutes=30))) &
        (df['close_datetime'] <= df['start_datetime'])
    ]
    
    # Drop the temporary columns 'close_datetime' and 'start_datetime' if no longer needed
    filtered_odds_data = filtered_odds_data.drop(columns=['close_datetime', 'start_datetime'])

    return filtered_odds_data

# Create a new column 'WinProb' in probability_data
def calc_win_prob(row, target):
    mean = row['x{}'.format(target)]
    std = row['SampleStd']
    x = row['line']
    cdf_value = norm.cdf(x, loc=mean, scale=std)
    if row['side'] == 'Under':
        return cdf_value
    elif row['side'] == 'Over':
        return 1 - cdf_value

def compile_backtest_results(probability_data, target, features, sportsbook=None):
    """
    """
    target_col = 'x{}'.format(target)
    
    roi_df = probability_data.groupby('bet_flag')['profit'].sum() / probability_data.groupby('bet_flag')['profit'].count()
    profit_df = probability_data.groupby('bet_flag')['profit'].sum()
    line_move_df = probability_data.groupby('bet_flag')['line_move'].sum()
    #game_date_df = probability_data.groupby(['game_date','bet_flag'])['profit'].sum()
    
    # Calculate RMSE for model and line predictions
    total_model_rmse = np.sqrt(np.sum((probability_data[target_col] - probability_data[target])**2) / len(probability_data))
    total_line_rmse = np.sqrt(np.sum((probability_data['line'] - probability_data[target])**2) / len(probability_data))

    bets_model_rmse = np.sqrt(np.sum((probability_data[probability_data['bet_flag'] == 1][target_col] - probability_data[probability_data['bet_flag'] == 1][target])**2) / len(probability_data[probability_data['bet_flag'] == 1]))
    bets_line_rmse = np.sqrt(np.sum((probability_data[probability_data['bet_flag'] == 1]['line'] - probability_data[probability_data['bet_flag'] == 1][target])**2) / len(probability_data[probability_data['bet_flag'] == 1]))

    # Calculate Mean Absolute Error (MAE) for model and line predictions
    total_model_mae = np.mean(np.abs(probability_data[target_col] - probability_data[target]))
    total_line_mae = np.mean(np.abs(probability_data['line'] - probability_data[target]))

    # Calculate Mean Absolute Error (MAE) for model and line predictions for bets
    bets_model_mae = np.mean(np.abs(probability_data[probability_data['bet_flag'] == 1][target_col] - probability_data[probability_data['bet_flag'] == 1][target]))
    bets_line_mae = np.mean(np.abs(probability_data[probability_data['bet_flag'] == 1]['line'] - probability_data[probability_data['bet_flag'] == 1][target]))

    # Log the summary data
    logging.info("------------------------------------------------------------------")
    if sportsbook == None:
        logging.info("Agg Backtest Summary:")
    else:
        logging.info("{} Backtest Summary:".format(sportsbook))

    logging.info("Bet Count: %d out of %d total potential bets", probability_data['bet_flag'].sum(), len(probability_data))
    logging.info("")
    logging.info("Return on Investment (ROI) - Model Bets: %s vs Passes: %s", roi_df[1], roi_df[0])
    logging.info("Total Profit - Model Bets: %s vs Passes: %s", profit_df[1], profit_df[0])
    logging.info("")
    logging.info("Total Line Move - Model Bets: %s vs Passes: %s", line_move_df[1], line_move_df[0])
    logging.info("")
    logging.info("Root Mean Square Error (RMSE) - Model: %s vs Line: %s", total_model_rmse, total_line_rmse)
    logging.info("Flagged Bets - Root Mean Square Error (RMSE) - Model: %s vs Line: %s", bets_model_rmse, bets_line_rmse)
    logging.info("")
    logging.info("Mean Absolute Error (MAE) - Model: %s vs Line: %s", total_model_mae, total_line_mae)
    logging.info("Flagged Bets - Mean Absolute Error (MAE) - Model: %s vs Line: %s", bets_model_mae, bets_line_mae)
    logging.info("")

    # Calculate log loss for model probabilities
    total_model_log_loss = -np.mean(probability_data[probability_data['bet_flag'] == 1].apply(
        lambda row: np.log(row['WinProb']) if row['bet_result'] == 'Win' else np.log(1 - row['WinProb']), axis=1))

    # Calculate log loss for line probabilities
    total_line_log_loss = -np.mean(probability_data[probability_data['bet_flag'] == 1].apply(
        lambda row: np.log(1 / row['closing_decimal']) if row['bet_result'] == 'Win' else np.log(1 - 1 / row['closing_decimal']), axis=1))

    # Calculate log loss for model probabilities for bets
    bets_model_log_loss = -np.mean(probability_data[probability_data['bet_flag'] == 1].apply(
        lambda row: np.log(row['WinProb']) if row['bet_result'] == 'Win' else np.log(1 - row['WinProb']), axis=1))

    # Calculate log loss for line probabilities for bets
    bets_line_log_loss = -np.mean(probability_data[probability_data['bet_flag'] == 1].apply(
        lambda row: np.log(1 / row['closing_decimal']) if row['bet_result'] == 'Win' else np.log(1 - 1 / row['closing_decimal']), axis=1))
        

    logging.info("Log Loss Error - Model: %s vs Line: %s", total_model_log_loss, total_line_log_loss)
    logging.info("Flagged Bets - Log Loss Error - Model: %s vs Line: %s", bets_model_log_loss, bets_line_log_loss)    


    if len(probability_data['Position'].unique()) > 1:
        positional_profit = probability_data.groupby(['Position','bet_flag'])['profit'].sum()
        positional_roi = probability_data.groupby(['Position','bet_flag'])['profit'].sum() / probability_data.groupby(['Position','bet_flag'])['profit'].count()
        logging.info("Positional ROI - Model Bets: \n {}".format(positional_roi))
        logging.info("Positional Profit - Model Bets: \n {}".format(positional_profit))

    #logging.info("Features: %s", features)


    return 

def create_final_report(probability_data, target, sportsbooks, features):
    """
    """

    # Analysis per sportsbook
    for sportsbook in sportsbooks:
        sportsbook_data = probability_data[probability_data['sportsbook'] == sportsbook]
        compile_backtest_results(sportsbook_data, target, features=features, sportsbook = sportsbook)

    # Aggregate report
    compile_backtest_results(probability_data, target, features=features)

    return
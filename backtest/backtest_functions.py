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
    std = row['{}Std'.format(target)]
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
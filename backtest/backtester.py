import pandas as pd
pd.options.mode.chained_assignment = None

import sys

import numpy as np
import re

import os

import sqlite3
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, TIMESTAMP, create_engine, ForeignKey, Index
from sqlalchemy.orm import sessionmaker, relationship

from xgboost import XGBRegressor

import logging

from requests import Session as APISession

from datetime import datetime, timedelta  
import pytz

import json
import pickle

from scipy.stats import norm

from backtest_functions import read_config, get_results_data, get_historical_odds, create_player_id, convert_to_decimal, calc_win_prob, create_final_report, calc_prob_params
import sys
sys.path.insert(0, '../etl/')
from shared_etl_functions import read_model, get_modeling_data, estimate_std, get_modeling_data, nfl_connect, odds_connect, calc_prob_params

# Process parameters
if len(sys.argv) < 2:
    user_model = False
    model_name = None
elif len(sys.argv) == 2:   
    user_model = True
    model_name = sys.argv[1]
elif len(sys.argv) == 3:
    user_model = True
    model_name = sys.argv[1]
    position = sys.argv[2]
else:
    print('Invalid params..')
    raise

# Ensure the logs directory exists
os.makedirs('./logs', exist_ok=True)

# Get the current timestamp
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# Configure logging
log_filename = f'./logs/{model_name}-{timestamp}.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])

if __name__ == '__main__':
    """
    It is assumed that all models are 
    trained outside of this script
    (in a jupyter notebook for example)
    """
    # Read in parameters
    config = read_config('config.json')
    
    model_name = config.get('model_name')
    position = config.get('position')
    user_model = True if model_name != "" else False

    odds_type = config.get('odds_type')
    backtest_year = config.get('backtest_year')
    target = config.get('target')
    sportsbooks = config.get('sportsbooks')
    target = config.get('target')

    lower_odds_bound = config.get('lower_odds_bound')
    upper_odds_bound = config.get('upper_odds_bound')

    distribution = config.get('distribution')

    diff_criteria = config.get('diff_criteria')
    prob_criteria = config.get('prob_criteria')

    min_obs = config.get('min_obs')
    # player, k-nearest-season-projection, k-nearest-historical-projection, k-nearest-week-projection
    std_estimation = None if config.get('std_estimation') == '' else config.get('std_estimation')
    k = None if config.get('k') == '' else config.get('k')

    logging.info("User model: {}".format(model_name))
    logging.info("Odds Type: {}".format(odds_type))
    logging.info("Distribution Type: {}".format(distribution))
    logging.info("Diff Thresholds: %s to %s", diff_criteria['min_threshold'], diff_criteria['max_threshold'])
    logging.info("Probability Thresholds: %s to %s", prob_criteria['min_threshold'], prob_criteria['max_threshold'])
    logging.info("Min Observations Allowed in backtest: %s", min_obs)
    logging.info("Target Stat: {}".format(target))
    logging.info("Sportsbooks: {}".format(sportsbooks))
    logging.info("STD Estimation technique: {}, using a k parameter of: {}".format(std_estimation, k))
    logging.info("---------------------------------------------")

    # If no user model is provided, we just backtest using simple features
    if user_model:
        model_dir = '../models/{}/'.format(target)
        model, features = read_model(model_dir, model_name)
    else:
        model = None
        model_name = 'SeasonAvg{}'.format(target)
        features = ['SeasonAvg{}'.format(target)]
        logging.info("No model provided, using only this feature to predict: {}".format(model_name))

    # Setup data connections
    #stats_db_conn, odds_db_conn = setup_db_connection()
    stats_db_conn = nfl_connect()
    #odds_db_session, engine = odds_connect()

    # Get data
    modeling_data = get_modeling_data(stats_db_conn, target, lag=True, position=position)
    results_data = get_results_data(stats_db_conn, target, backtest_year)
    odds_data = get_historical_odds(target, sportsbooks=sportsbooks, odds_type=odds_type)
    # Merge historical and modeling data
    #model_historical_data = pd.merge(modeling_data, historical_data, on=['PlayerId','GameDate'])
    model_historical_data = pd.merge(modeling_data, results_data, on=['PlayerId','Player','GameDate'])

    # Add features as necessary
    if (target == 'RecYards') and (model_name is not None):
        receptions_model_dir = '../models/Receptions/'
        model_name = 'xgb_xreceptions_{}_backtest'.format(position)
        receptions_model, receptions_features = read_model(receptions_model_dir, model_name)
        model_historical_data['xReceptions'] = receptions_model.predict(model_historical_data[receptions_features])   
    elif (target == 'RushYards') and (model_name is not None):
        receptions_model_dir = '../models/RushAtt/'
        model_name = 'xgb_xrushatt_{}_backtest'.format(position)
        rushatt_model, rushatt_features = read_model(receptions_model_dir, model_name)
        model_historical_data['xRushAtt'] = rushatt_model.predict(model_historical_data[rushatt_features])   

    # Drop invalid target values
    # NOTE: this line of code DRAMATICALLY changes results (removing it degrades prediction quality from what I've seen)
    model_historical_data = model_historical_data.dropna(subset=features)

    # Prep model data
    X = model_historical_data[features]

    # Make predictions
    if model is not None:
        # Use the model if available
        model_historical_data['x{}'.format(target)] = model.predict(X)
    else:
        # Use the single feature directly
        model_historical_data['x{}'.format(target)] = X[features[0]]

    # Create samples and estimate standard deviation
    predictions_data = estimate_std(model_historical_data, target, std_estimation, k, lookback_years=[backtest_year])

    # Prep odds data
    odds_data['player_id'] = odds_data['name'].apply(create_player_id)    
    odds_data['game_date_est_dt'] = [
        datetime.strptime(i, '%Y-%m-%dT%H:%M:%S%z').astimezone(pytz.timezone('US/Eastern'))
        for i in odds_data['start_date']
    ]

    odds_data['game_date_est'] = odds_data['game_date_est_dt'].dt.strftime('%Y-%m-%d')
    odds_data['price'] = odds_data[odds_type].apply(lambda x: convert_to_decimal(x))
    #odds_data['price'] = odds_data['closing'].apply(lambda x: convert_to_decimal(x))
    #odds_data['opening_decimal'] = odds_data['opening'].apply(lambda x: convert_to_decimal(x))


    # Filter out wild odds (throws off results a lot in small samples)
    odds_data = odds_data[(odds_data['price'] <= convert_to_decimal(lower_odds_bound)) & (odds_data['price'] >= convert_to_decimal(upper_odds_bound))]
    logging.info("Filtering odds data to only include odds between {} and {}".format(lower_odds_bound, upper_odds_bound))

    # Merge odds and predictions
    if std_estimation != 'player':
        cols = ['PlayerId','GameDate','Position',target,'x{}'.format(target),'x{}Cohort'.format(target), 'x{}CohortProj'.format(target), '{}Std'.format(target),'x{}Sample'.format(target)]
    else:
        cols = ['PlayerId','GameDate','Position',target,'x{}'.format(target), '{}Std'.format(target),'x{}Sample'.format(target)]

    predictions_data = predictions_data[cols]
    merged_data = pd.merge(predictions_data, odds_data, left_on=['PlayerId','GameDate'], right_on=['player_id','game_date_est'])

    # Prep data for calculating probabilities
    merged_data['BookProb'] = 1 / merged_data['price']  
    # Drop records where the length of the target + 'Sample' is less than min_obs
    initial_count = len(merged_data)
    probability_data = merged_data[merged_data['x{}Sample'.format(target)].apply(lambda x: len(x)) >= min_obs]
    filtered_count = len(probability_data)
    logging.info(f"Filtered out {initial_count - filtered_count} records for having less than {min_obs} observations in Sample. {filtered_count} records remain.")

    # Determine optimal distribution fit and calculate gamma parameters
    probability_data[['BestFit', 'GammaShape', 'GammaScale']] = probability_data.apply(lambda row: calc_prob_params(row, target=target), axis=1)

    # Calculate probabilities using a PDF
    if distribution == 'dynamic':
        probability_data['WinProb'] = probability_data.apply(lambda row: calc_win_prob(row, distribution=row['BestFit'], target=target,), axis=1)
    else:
        probability_data['WinProb'] = probability_data.apply(lambda row: calc_win_prob(row, distribution=distribution, target=target,), axis=1)

    # Calculate edges
    probability_data['Diff'] = probability_data.apply(
        lambda row: row['x{}'.format(target)] - row['line'],
        axis=1
    )
    # Calculate absolute diff edge
    probability_data['DiffEdge'] = probability_data.apply(
            lambda row: row['x{}'.format(target)] - row['line'] if row['side'] == 'Over' else row['line'] - row['x{}'.format(target)],
            axis=1
        )
    
    # Calculate probability edge
    probability_data['ProbEdge'] = probability_data['WinProb'] - probability_data['BookProb']
    
    # Grade all bets
    probability_data['bet_result'] = np.where(
        (probability_data['side'] == 'Over') & (probability_data[target] - probability_data['line'] > 0),
        'Win',
        np.where(
            (probability_data['side'] == 'Under') & (probability_data['line'] - probability_data[target] > 0),
            'Win',
            'Loss'
        )
    )
    # Determine bets to make based on threshold
    probability_data['bet_flag'] = np.where(
        (probability_data['ProbEdge'] >= prob_criteria['min_threshold']) & 
        (probability_data['ProbEdge'] <= prob_criteria['max_threshold']) & 

        (probability_data['DiffEdge'] >= diff_criteria['min_threshold']) & 
        (probability_data['DiffEdge'] <= diff_criteria['max_threshold']), 
        1, 
        0
    )
    probability_data['profit'] = np.where(probability_data['bet_result'] == 'Win', probability_data['price'] - 1, -1)

    # Reporting summary    
    create_final_report(probability_data, target, sportsbooks, features)

    probability_data.to_csv('./debug.csv', index=False)
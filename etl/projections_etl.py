import logging
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, TIMESTAMP, create_engine, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, relationship
import pandas as pd

import json

from shared_etl_functions import read_model, nfl_connect, odds_connect, get_modeling_data, get_opp_defense_data, get_latest_moneylines, get_latest_totals, estimate_std
from schedule_data_etl import create_schedule_data

import sys
sys.path.insert(0, '../../bet-app/code/')
from data_models import Game, Player, NFLProjection
from functions import create_utc_timestamp, insert_new_records


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Base = declarative_base()

def projections_etl(model_dir):
    """
    """
    logging.info("Starting projections ETL process")

    # Connect to DBs
    logging.info("Connecting to NFL stats database")
    stats_conn = nfl_connect()
    
    logging.info("Connecting to odds database")
    odds_db_session, engine = odds_connect()
    # Check if table exists and create if not
    NFLProjection.__table__.create(bind=engine, checkfirst=True)

    # Define current active markets
    targets = ['PassYards','RushYards']

    # Initialize an empty DataFrame with the base columns   
    agg_projections_data = pd.DataFrame(columns=['GameId2', 'PlayerId', 'Player', 'TeamAbbr', 'GameDate','Position'])


    for target in targets:
        logging.info(f"Processing: {target}")
        
        # Read model and features
        model_name = f'xgb_x{target.lower()}_2024'
        model, features = read_model(model_dir, model_name)
        
        # Read in model parameters
        with open('../models/prod_params.json', 'r') as file:
            params = json.load(file)

        # Pull data
        historic_modeling_data = get_modeling_data(stats_conn, target, lag=True)
        current_modeling_data = get_modeling_data(stats_conn, target, lag=False)
        schedule_data = create_schedule_data()
        schedule_data['GameDate'] = schedule_data['GameId2'].str[:10]
        logging.info(f"Current modeling data size before merging schedule data: {len(current_modeling_data)}")
        # Merge schedule data in to get OppAbbr and updated GameDate, GameId2
        drop_cols = ['GameId','GameId2','GameDate']
        current_modeling_data.drop(drop_cols, axis=1, inplace=True)
        current_modeling_data = pd.merge(current_modeling_data, schedule_data, on='TeamAbbr')
        logging.info(f"Current modeling data size after merging schedule data: {len(current_modeling_data)}")

        # Get opp data using OppAbbr
        team_abbrs = current_modeling_data['TeamAbbr'].unique()
        current_opp_data = get_opp_defense_data(stats_conn, team_abbrs)

        # Merge opponent team data into modeling dataset
        info_cols = ['GameId','GameId2','Season','GameDate']
        logging.info(f"Current modeling data size before merging opponent data: {len(current_modeling_data)}")
        current_modeling_data = pd.merge(
            current_modeling_data, 
            current_opp_data.drop(columns=info_cols), 
            left_on=['OppAbbr'], 
            right_on=['TeamAbbr'], 
            suffixes=('', '_y')
        )
        current_modeling_data = current_modeling_data.drop(columns=['TeamAbbr_y'])
        logging.info(f"Current modeling data size after merging opponent data: {len(current_modeling_data)}")

        # Get latest game odds data (home_flag is calc'ed )
        latest_moneylines = get_latest_moneylines(odds_db_session, engine)
        latest_totals = get_latest_totals(odds_db_session, engine)
        game_odds_data = pd.merge(latest_moneylines, latest_totals, on='game_id')

        logging.info(f"Current modeling data size before merging game odds data: {len(current_modeling_data)}")
        current_modeling_data = pd.merge(current_modeling_data, game_odds_data, on='TeamAbbr')
        logging.info(f"Current modeling data size after merging game odds data: {len(current_modeling_data)}")

        # Drop invalid target values (this has a large impact on results actually)
        initial_historic_count = len(historic_modeling_data)
        historic_modeling_data = historic_modeling_data.dropna(subset=features)
        dropped_historic_count = initial_historic_count - len(historic_modeling_data)
        remaining_historic_count = len(historic_modeling_data)
        logging.info(f"Dropped {dropped_historic_count} invalid records from historical data, {remaining_historic_count} records remain")

        initial_current_count = len(current_modeling_data)
        current_modeling_data = current_modeling_data.dropna(subset=features)
        dropped_current_count = initial_current_count - len(current_modeling_data)
        remaining_current_count = len(current_modeling_data)
        logging.info(f"Dropped {dropped_current_count} invalid records from current data, {remaining_current_count} records remain")

        # Make predictions
        logging.info(f"Making predictions for target: {target}")
        X_current = current_modeling_data[features]
        X_historic = historic_modeling_data[features]
        current_modeling_data['x{}'.format(target)] = model.predict(X_current)
        historic_modeling_data['x{}'.format(target)] = model.predict(X_historic)

        # Combine historic and current week projections before estimating STD
        combined_cols = ['PlayerId','Player','Position','TeamAbbr','Season','GameId2', 'GameDate', 'x{}'.format(target)]
        combined_modeling_data = pd.concat([current_modeling_data, historic_modeling_data])[combined_cols]
        initial_combined_count = len(combined_modeling_data)
        logging.info(f"Initial combined modeling data size: {initial_combined_count}")
        # Estimate STD
        logging.info(f"Estimating standard deviation for target: {target}")
        std_estimation = params[target]['std_estimation']
        k = params[target]['k']
        # Calculte STD off of player's career historical sample
        if std_estimation == 'player':
            combined_modeling_data = estimate_std(combined_modeling_data, target, std_estimation, k, lookback_years=[2023,2024])
        elif std_estimation == 'k-nearest-season-projection':
            # FLAG - might want to expand to 2023 for early weeks of season to increase sample size
            combined_modeling_data = estimate_std(combined_modeling_data, target, std_estimation, k, lookback_years=[2024])
        
        else:
            import ipdb; ipdb.set_trace()
        
        # Remove historical records after estimating STD
        projections_data = combined_modeling_data.dropna(subset='GameId2')   # Historical modeling data doesn't have GameId2 field

        # Log the length of the dataframe after dropping records
        final_combined_count = len(projections_data)
        dropped_combined_count = initial_combined_count - final_combined_count
        logging.info(f"Final combined modeling data size: {final_combined_count}")
        logging.info(f"Dropped {dropped_combined_count} records from combined modeling data for projections data")

        # Filter resultset down to relevant columns
        cols = ['GameId2','PlayerId','Player','TeamAbbr','GameDate','Position','x{}'.format(target),'{}Std'.format(target)]
        projections_data = projections_data[cols]

        # Add current target stat's projections to the agg dataframe
        agg_projections_data = pd.merge(agg_projections_data, 
                                        projections_data, 
                                        on=['GameId2', 'PlayerId', 'Player', 'TeamAbbr', 'GameDate','Position'], 
                                        how='outer')
    

    # Map IDs at some point
    logging.info(f"Mapping GameId2 to internal ID")
    game_ids = odds_db_session.query(Game.game_id2, Game.id).all()
    game_ids_df = pd.DataFrame(game_ids, columns=['GameId2', 'game_id'])
    # Merge internal game IDs to working data
    agg_projections_data = pd.merge(agg_projections_data, 
                          game_ids_df, 
                          on = 'GameId2',
                          how = 'left')

    # Add projection source (always internal for now)
    agg_projections_data['source'] = 'internal'

    # Add timestamp
    agg_projections_data['timestamp'] = create_utc_timestamp()

    # Convert NaN to None for MySQL
    agg_projections_data = agg_projections_data.replace({pd.NA: None, pd.NaT: None, float('nan'): None})


    # Filter and rename columns
    agg_projections_data = agg_projections_data.rename(columns={'PlayerId':'player_id2',
                                                                'Position':'position',
                                                                'xPassYards':'pass_yards',
                                                                'PassYardsStd':'pass_yards_std',
                                                                'xRushYards':'rush_yards',
                                                                'RushYardsStd':'rush_yards_std'})
    
    final_cols = ['game_id','player_id2','position','source','pass_yards','pass_yards_std','rush_yards','rush_yards_std', 'timestamp']     
    new_records_df = agg_projections_data[final_cols].copy()

    # Insert and replace records (for now)
    logging.info(f"Inserting projection records")

    # Insert new, unique records
    if not new_records_df.empty:
        # FLAG: improve this at some point
        odds_db_session.query(NFLProjection).delete()
        insert_new_records(odds_db_session, new_records_df, NFLProjection)

    logging.info("Projections ETL process completed")
    return

if __name__ == '__main__':
    model_dir = '../models/prod/'
projections_etl(model_dir)
import logging
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, TIMESTAMP, create_engine, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, relationship
import pandas as pd

import json

from shared_etl_functions import read_model, nfl_connect, odds_connect, get_modeling_data, get_opp_defense_data, get_latest_moneylines, get_latest_totals, get_latest_spreads, estimate_std, calc_prob_params
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
    targets = ['RushAtt','RushYards','PassYards','Receptions','RecYards']

    # Initialize an empty DataFrame with the base columns   
    agg_projections_data = pd.DataFrame(columns=['GameId2', 'PlayerId', 'Player', 'TeamAbbr', 'GameDate','Position'])

    for target in targets:
        logging.info(f"Processing: {target}")
        
        # Read model and features
        if target in ['RushAtt','RushYards']:
            model_name_qb =f'xgb_x{target.lower()}_qb_prod'
            model_name_rb =f'xgb_x{target.lower()}_rb_prod'
            qb_model, qb_features = read_model(model_dir, model_name_qb)
            rb_model, rb_features = read_model(model_dir, model_name_rb)
            features = list(set([*qb_features, *rb_features]))
        elif target in ['Receptions','RecYards']:
            model_name_rb =f'xgb_x{target.lower()}_rb_prod'
            model_name_wr =f'xgb_x{target.lower()}_wr_prod'
            model_name_te =f'xgb_x{target.lower()}_te_prod'
            rb_model, rb_features = read_model(model_dir, model_name_rb)
            wr_model, wr_features = read_model(model_dir, model_name_wr)
            te_model, te_features = read_model(model_dir, model_name_te)
            features = list(set([*rb_features, *wr_features, *te_features]))
        else:
            model_name = f'xgb_x{target.lower()}_qb_prod'
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
        latest_spreads = get_latest_spreads(odds_db_session, engine)
        game_odds_data = pd.merge(latest_moneylines, latest_totals, on='game_id')
        game_odds_data = pd.merge(game_odds_data, latest_spreads, on='TeamAbbr')

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
        combined_cols = ['PlayerId','Player','Position','TeamAbbr','Season','GameId2', 'GameDate', 'x{}'.format(target)]
        if target in ['RushAtt','RushYards']:
            # Predict QB
            qb_current_modeling_data = current_modeling_data[current_modeling_data['Position'] == 'QB'].copy()
            qb_historic_modeling_data = historic_modeling_data[historic_modeling_data['Position'] == 'QB'].copy()
            X_qb_current = qb_current_modeling_data[qb_features].copy()
            X_qb_historic = qb_historic_modeling_data[qb_historic_modeling_data['Position'] == 'QB'][qb_features].copy()
            qb_current_modeling_data['x{}'.format(target)] = qb_model.predict(X_qb_current)
            qb_historic_modeling_data['x{}'.format(target)] = qb_model.predict(X_qb_historic)
            # Estimate QB STD
            logging.info(f"Estimating standard deviation for QB target: {target}")
            qb_std_estimation = params[target]['QB']['std_estimation']
            qb_k = params[target]['QB']['k']            
            qb_distribution = params[target]['QB']['distribution']            
            qb_combined_modeling_data = pd.concat([qb_current_modeling_data, qb_historic_modeling_data])[combined_cols]
            qb_combined_modeling_data = estimate_std(qb_combined_modeling_data, target, qb_std_estimation, qb_k, lookback_years=[2024])
            # Determine distribution
            if qb_distribution == 'normal':
                qb_combined_modeling_data['{}Dist'.format(target)] = 'normal'
            elif qb_distribution == 'gamma':
                qb_combined_modeling_data['{}Dist'.format(target)] = 'gamma'
            else:
                qb_combined_modeling_data[['{}Dist'.format(target), 'GammaShape', 'GammaScale']] = qb_combined_modeling_data.apply(lambda row: calc_prob_params(row, target=target), axis=1)

            # Predict RB
            rb_current_modeling_data = current_modeling_data[current_modeling_data['Position'] == 'RB'].copy()
            rb_historic_modeling_data = historic_modeling_data[historic_modeling_data['Position'] == 'RB'].copy()
            X_rb_current = rb_current_modeling_data[rb_features].copy()
            X_rb_historic = rb_historic_modeling_data[rb_historic_modeling_data['Position'] == 'RB'][rb_features].copy()
            rb_current_modeling_data['x{}'.format(target)] = rb_model.predict(X_rb_current)
            rb_historic_modeling_data['x{}'.format(target)] = rb_model.predict(X_rb_historic)
            # Estimate RB STD
            logging.info(f"Estimating standard deviation for RB target: {target}")
            rb_std_estimation = params[target]['RB']['std_estimation']
            rb_k = params[target]['RB']['k']            
            rb_distribution = params[target]['RB']['distribution']            
            rb_combined_modeling_data = pd.concat([rb_current_modeling_data, rb_historic_modeling_data])[combined_cols]
            rb_combined_modeling_data = estimate_std(rb_combined_modeling_data, target, rb_std_estimation, rb_k, lookback_years=[2024])
            # Determine distribution
            if rb_distribution == 'normal':
                rb_combined_modeling_data['{}Dist'.format(target)] = 'normal'
            elif rb_distribution == 'gamma':
                rb_combined_modeling_data['{}Dist'.format(target)] = 'gamma'
            else:
                rb_combined_modeling_data[['{}Dist'.format(target), 'GammaShape', 'GammaScale']] = rb_combined_modeling_data.apply(lambda row: calc_prob_params(row, target=target), axis=1)


            # Concatenate them together
            combined_modeling_data = pd.concat([qb_combined_modeling_data, rb_combined_modeling_data])
            
            initial_combined_count = len(combined_modeling_data)
            logging.info(f"Initial combined modeling data size: {initial_combined_count}")       

        elif target in ['Receptions','RecYards']:
            # Predict RB
            rb_current_modeling_data = current_modeling_data[current_modeling_data['Position'] == 'RB'].copy()
            rb_historic_modeling_data = historic_modeling_data[historic_modeling_data['Position'] == 'RB'].copy()
            X_rb_current = rb_current_modeling_data[rb_features].copy()
            X_rb_historic = rb_historic_modeling_data[rb_historic_modeling_data['Position'] == 'RB'][rb_features].copy()
            rb_current_modeling_data['x{}'.format(target)] = rb_model.predict(X_rb_current)
            rb_historic_modeling_data['x{}'.format(target)] = rb_model.predict(X_rb_historic)
            # Estimate RB STD
            logging.info(f"Estimating standard deviation for RB target: {target}")
            rb_std_estimation = params[target]['RB']['std_estimation']
            rb_k = params[target]['RB']['k']            
            rb_distribution = params[target]['RB']['distribution']            
            rb_combined_modeling_data = pd.concat([rb_current_modeling_data, rb_historic_modeling_data])[combined_cols]
            rb_combined_modeling_data = estimate_std(rb_combined_modeling_data, target, rb_std_estimation, rb_k, lookback_years=[2024])
            # Determine distribution
            if rb_distribution == 'normal':
                rb_combined_modeling_data['{}Dist'.format(target)] = 'normal'
            elif rb_distribution == 'gamma':
                rb_combined_modeling_data['{}Dist'.format(target)] = 'gamma'
            else:
                rb_combined_modeling_data[['{}Dist'.format(target), 'GammaShape', 'GammaScale']] = rb_combined_modeling_data.apply(lambda row: calc_prob_params(row, target=target), axis=1)

            # Predict WR
            wr_current_modeling_data = current_modeling_data[current_modeling_data['Position'] == 'WR'].copy()
            wr_historic_modeling_data = historic_modeling_data[historic_modeling_data['Position'] == 'WR'].copy()
            X_wr_current = wr_current_modeling_data[wr_features].copy()
            X_wr_historic = wr_historic_modeling_data[wr_historic_modeling_data['Position'] == 'WR'][wr_features].copy()
            wr_current_modeling_data['x{}'.format(target)] = wr_model.predict(X_wr_current)
            wr_historic_modeling_data['x{}'.format(target)] = wr_model.predict(X_wr_historic)
            # Estimate WR STD
            logging.info(f"Estimating standard deviation for WR target: {target}")
            wr_std_estimation = params[target]['WR']['std_estimation']
            wr_k = params[target]['WR']['k']            
            wr_distribution = params[target]['WR']['distribution']            
            wr_combined_modeling_data = pd.concat([wr_current_modeling_data, wr_historic_modeling_data])[combined_cols]
            wr_combined_modeling_data = estimate_std(wr_combined_modeling_data, target, wr_std_estimation, wr_k, lookback_years=[2024])
            # Determine distribution
            if wr_distribution == 'normal':
                wr_combined_modeling_data['{}Dist'.format(target)] = 'normal'
            elif wr_distribution == 'gamma':
                wr_combined_modeling_data['{}Dist'.format(target)] = 'gamma'
            else:
                wr_combined_modeling_data[['{}Dist'.format(target), 'GammaShape', 'GammaScale']] = wr_combined_modeling_data.apply(lambda row: calc_prob_params(row, target=target), axis=1)

            # Predict TE
            te_current_modeling_data = current_modeling_data[current_modeling_data['Position'] == 'TE'].copy()
            te_historic_modeling_data = historic_modeling_data[historic_modeling_data['Position'] == 'TE'].copy()
            X_te_current = te_current_modeling_data[te_features].copy()
            X_te_historic = te_historic_modeling_data[te_historic_modeling_data['Position'] == 'TE'][te_features].copy()
            te_current_modeling_data['x{}'.format(target)] = te_model.predict(X_te_current)
            te_historic_modeling_data['x{}'.format(target)] = te_model.predict(X_te_historic)
            # Estimate TE STD
            logging.info(f"Estimating standard deviation for TE target: {target}")
            te_std_estimation = params[target]['TE']['std_estimation']
            te_distribution = params[target]['TE']['distribution']            
            te_k = params[target]['TE']['k']            
            te_combined_modeling_data = pd.concat([te_current_modeling_data, te_historic_modeling_data])[combined_cols]
            te_combined_modeling_data = estimate_std(te_combined_modeling_data, target, te_std_estimation, te_k, lookback_years=[2024])
            # Determine distribution
            if te_distribution == 'normal':
                te_combined_modeling_data['{}Dist'.format(target)] = 'normal'
            elif te_distribution == 'gamma':
                te_combined_modeling_data['{}Dist'.format(target)] = 'gamma'
            else:
                te_combined_modeling_data[['{}Dist'.format(target), 'GammaShape', 'GammaScale']] = te_combined_modeling_data.apply(lambda row: calc_prob_params(row, target=target), axis=1)

            # Concatenate them together
            combined_modeling_data = pd.concat([rb_combined_modeling_data, wr_combined_modeling_data, te_combined_modeling_data])
            initial_combined_count = len(combined_modeling_data)
            logging.info(f"Initial combined modeling data size: {initial_combined_count}")    

        elif target == 'PassYards':
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
            std_estimation = params[target]['QB']['std_estimation']
            k = params[target]['QB']['k']
            distribution = params[target]['QB']['distribution']            
            # Calculte STD off of player's career historical sample
            combined_modeling_data = estimate_std(combined_modeling_data, target, std_estimation, k, lookback_years=[2024])
            # Determine distribution
            if distribution == 'normal':
                combined_modeling_data['{}Dist'.format(target)] = 'normal'
            elif distribution == 'gamma':
                combined_modeling_data['{}Dist'.format(target)] = 'gamma'
            else:
                combined_modeling_data[['{}Dist'.format(target), 'GammaShape', 'GammaScale']] = combined_modeling_data.apply(lambda row: calc_prob_params(row, target=target), axis=1)
        
        # Remove historical records after estimating STD
        projections_data = combined_modeling_data.dropna(subset='GameId2')   # Historical modeling data doesn't have GameId2 field

        # Log the length of the dataframe after dropping records
        final_combined_count = len(projections_data)
        dropped_combined_count = initial_combined_count - final_combined_count
        logging.info(f"Final combined modeling data size: {final_combined_count}")
        logging.info(f"Dropped {dropped_combined_count} records from combined modeling data for projections data")

        # Filter resultset down to relevant columns
        cols = ['GameId2','PlayerId','Player','TeamAbbr','GameDate','Position','x{}'.format(target),'{}Std'.format(target),'{}Dist'.format(target)]
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
    cols ={'game_id' : 'game_id',
           'PlayerId':'player_id2',
           'Position':'position',
           'source'  : 'source',
           'xPassYards':'pass_yards',
           'PassYardsStd':'pass_yards_std',
           'PassYardsDist':'pass_yards_dist',
           'xRushAtt':'rush_att',
           'RushAttStd':'rush_att_std',
           'RushAttDist':'rush_att_dist',
           'xRushYards':'rush_yards',
           'RushYardsStd':'rush_yards_std',
           'RushYardsDist':'rush_yards_dist',
           'xReceptions':'receptions',
           'ReceptionsStd':'receptions_std',
           'ReceptionsDist':'receptions_dist',
           'xRecYards':'rec_yards',
           'RecYardsStd':'rec_yards_std',
           'RecYardsDist':'rec_yards_dist',
           'timestamp':'timestamp'}
    
    agg_projections_data = agg_projections_data.rename(columns=cols)
    
    final_cols = cols.values()     
    new_records_df = agg_projections_data[final_cols].copy()

    # Insert and replace records (for now)
    logging.info(f"Inserting projection records")
    # Insert new, unique records
    if not new_records_df.empty:
        # FLAG: improve this at some point
        #odds_db_session.query(NFLProjection).delete()
        insert_new_records(odds_db_session, new_records_df, NFLProjection)

    logging.info("Projections ETL process completed")
    return

if __name__ == '__main__':
    model_dir = '../models/prod/'
projections_etl(model_dir)
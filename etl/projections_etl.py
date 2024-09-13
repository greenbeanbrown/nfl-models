import logging
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, TIMESTAMP, create_engine, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, relationship
import pandas as pd

from shared_etl_functions import read_model, nfl_connect, odds_connect, get_modeling_data, get_opp_defense_data, get_latest_moneylines, get_latest_totals
from schedule_data_etl import create_schedule_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Base = declarative_base()

class NFLProjection(Base):
    """
    SQLAlchemy schema for wagers table
    """
    __tablename__ = 'projections'   # Use if tying to a specific table
    # PK
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    # FK
    player_id = Column(Integer, ForeignKey('players.oj_id'), nullable=False)  
    game_id = Column(Integer, ForeignKey('games.id'), nullable=False)  

    source = Column(String(255), nullable=False)

    # PROJECTION FIELDS
    #pass_att = Column(Float, nullable=False) # Nulls should be 0s (makes sense for projections)
    #pass_att_std = Column(Float, nullable=True) 

    #pass_comp = Column(Float, nullable=False) # Nulls should be 0s (makes sense for projections)
    #pass_comp_std = Column(Float, nullable=True)     

    pass_yards = Column(Float, nullable=False) # Nulls should be 0s (makes sense for projections)
    pass_yards_std = Column(Float, nullable=True)  

    #pass_td = Column(Float, nullable=False) # Nulls should be 0s (makes sense for projections)
    #pass_td_std = Column(Float, nullable=True)  

    #pass_int = Column(Float, nullable=False) # Nulls should be 0s (makes sense for projections)
    #pass_int_std = Column(Float, nullable=True)  

    rush_yards = Column(Float, nullable=False) # Nulls should be 0s (makes sense for projections)
    rush_yards_std = Column(Float, nullable=True)  

    #rush_td = Column(Float, nullable=False) # Nulls should be 0s (makes sense for projections)
    #rush_td_std = Column(Float, nullable=True)  

    #rec_catches = Column(Float, nullable=False) # Nulls should be 0s (makes sense for projections)
    #rec_catches_std = Column(Float, nullable=True)                      

    #rec_yards = Column(Float, nullable=False) # Nulls should be 0s (makes sense for projections)
    #rec_yards_std = Column(Float, nullable=True)                      

    #rec_td = Column(Float, nullable=False) # Nulls should be 0s (makes sense for projections)
    #rec_td_std = Column(Float, nullable=True)                

    timestamp = Column(TIMESTAMP, nullable=False)          

    game = relationship("Game")
    player = relationship("Player")

def projections_etl(model_dir):
    """
    """
    logging.info("Starting projections ETL process")
    
    # Connect to DBs
    logging.info("Connecting to NFL stats database")
    stats_conn = nfl_connect()
    
    logging.info("Connecting to odds database")
    odds_db_session, engine = odds_connect()

    # Define current active markets
    targets = ['PassYards','RushYards']

    for target in targets:
        logging.info(f"Processing: {target}")
        
        # Read model and features
        model_name = f'xgb_x{target.lower()}_2024'
        model, features = read_model(model_dir, model_name)
        
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

        # Estimate STD
        logging.info(f"Estimating standard deviation for target: {target}")
        std_estimation = 'player'
        k = None
        import ipdb; ipdb.set_trace()

        #estimate_std(current_modeling_data, historic_modeling_data)
        import ipdb; ipdb.set_trace()

        # Map IDs at some point
        logging.info(f"Mapping IDs for target: {target}")

        # Insert records
        logging.info(f"Inserting records for target: {target}")

    logging.info("Projections ETL process completed")
    return

if __name__ == '__main__':
    model_dir = '../models/prod/'
    projections_etl(model_dir)
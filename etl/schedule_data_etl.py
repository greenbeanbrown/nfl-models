import datetime
import pytz
import pandas as pd
import requests

def get_data(endpoint):
    """
    Call OddsJam API to get specific odds
    for a given market in a given game
    """
    try:
        # Make API call
        response = requests.get(endpoint)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON data from the response
            data = response.json()
        else:
            # Handle the case when the request was not successful
            print("Request failed with status code:", response.status_code)

    except requests.exceptions.RequestException as e:
        # Handle any exceptions that may occur during the request
        print("Request error:", e)        

    return data

def find_following_tuesday():
    # Get the current date
    current_date = datetime.date.today()
    # Calculate the days until the next Tuesday (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
    days_until_next_tuesday = (1 - current_date.weekday() + 14) % 7
    # Calculate the date of the next Tuesday from the current date
    next_tuesday = current_date + datetime.timedelta(days=days_until_next_tuesday)
    # If the calculated date is the same as the current date, add 7 days to get the next Tuesday
    if next_tuesday == current_date:
        next_tuesday += datetime.timedelta(days=7)
    # Add 7 days to get the following Tuesday
    following_tuesday = next_tuesday + datetime.timedelta(days=7)

    return following_tuesday

def find_next_tuesday():
    # Get the current date
    current_date = datetime.date.today()
    # Calculate the days until the next Tuesday (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
    days_until_next_tuesday = (1 - current_date.weekday() + 14) % 7
    # Calculate the date of the next Tuesday from the current date
    next_tuesday = current_date + datetime.timedelta(days=days_until_next_tuesday)
    # If the calculated date is the same as the current date, add 7 days to get the next Tuesday
    if next_tuesday == current_date:
        next_tuesday += datetime.timedelta(days=7)

    return next_tuesday

def create_schedule_endpoint():
    """
    """

    key = '6e9259fe-cfa5-4dab-8f0e-fefaf98f696d'
    league = 'NFL'
    todays_dt = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
    #next_tuesday = find_following_tuesday()
    next_tuesday = find_next_tuesday()
    endpoint = 'https://api-external.oddsjam.com/api/v2/schedules/list?key={}&league={}&start_date_after={}&start_date_before={}'.format(key, league, todays_dt, next_tuesday)
    print(endpoint)
    return endpoint



def parse_schedule_data(json_data):
    """
    """
    # Derive date in EST time zone
    est_timezone = pytz.timezone('US/Eastern')
    game_date_dt = datetime.datetime.fromisoformat(json_data['start_date']).astimezone(est_timezone)
    game_date_str = datetime.datetime.strftime(game_date_dt, '%Y-%m-%d')

    away_team_abbr = json_data['away_team_abb'].replace('WSH','WAS')
    home_team_abbr = json_data['home_team_abb'].replace('WSH','WAS')

    game_id = json_data['game_id']
    game_id2 = game_date_str + '-' + away_team_abbr + '-' + home_team_abbr

    schedule_data = {'GameId': [game_id], 
                     'GameId2': [game_id2],
                     'AwayTeamAbbr':[away_team_abbr],
                     'HomeTeamAbbr':[home_team_abbr]}

    df = pd.DataFrame(schedule_data)

    return df

def transform_data(schedule_data):
    """
    Transform the schedule data that 
    is in AwayAbbr vs HomeAbbr format 
    into TeamAbbr vs OppAbbr format
    to make for easy joining
    """
    # Create a new DataFrame to store the transformed data
    transformed_data = []

    # Iterate through each row in the input DataFrame
    for _, row in schedule_data.iterrows():
        # Create a new row for Away Team
        away_row = {
            'GameId2': row['GameId2'],
            'TeamAbbr': row['AwayTeamAbbr'],
            'OppAbbr': row['HomeTeamAbbr']
        }
        # Create a new row for Home Team
        home_row = {
            'GameId2': row['GameId2'],
            'TeamAbbr': row['HomeTeamAbbr'],
            'OppAbbr': row['AwayTeamAbbr']
        }
        # Append the rows to the transformed data
        transformed_data.extend([away_row, home_row])

    # Create a new DataFrame from the transformed data
    transformed_df = pd.DataFrame(transformed_data)

    return transformed_df

def create_schedule_data():
    """
    """
    # Make API call and get data
    endpoint = create_schedule_endpoint()
    json_data = get_data(endpoint)
    data_list = json_data['data']
    # Parse the data to get current weeks schedule data
    schedule_data = pd.DataFrame()
    for i in data_list:
        df = parse_schedule_data(i)
        schedule_data = pd.concat([schedule_data, df])
    # Transform the Home vs Away into Team vs Opp
    team_opp_data = transform_data(schedule_data)

    return team_opp_data

if __name__ == '__main__':

    print(create_schedule_data())
RAW_PLAYER_STATS_COLS = {
    'BIGDATABALL\nDATASET'   : 'DatasetName',
    'DATASET'                : 'DatasetName',
    'GAME-ID'                : 'GameId',
    'DATE'                   : 'GameDate',
    'WEEK\n#'                : 'Week',
    'WEEK'                   : 'Week',
    'Season'                 : 'Season',
    'PLAYER ID'              : 'PlayerId',
    'PLAYER'                 : 'Player',
    'POSITION'               : 'Position',
    'TEAM'                   : 'Team',
    'OPPONENT'               : 'Opp',
    'STARTER\n(Y/N)'         : 'StarterFlag',
    'VENUE\n(R/H/N)'         : 'Venue',
    'COMP'                   : 'PassComp',
    'ATT'                    : 'PassAtt',
    'YDS'                    : 'PassYards',
    'TD'                     : 'PassTd',
    'INT'                    : 'PassInt',
    'SK'                     : 'TimesSacked',
    'YDS.1'                  : 'SackYardsLost',
    'LG'                     : 'LongestPassComp',
    'ATT.1'                  : 'RushAtt',
    'YDS.2'                  : 'RushYards',
    'TD.1'                   : 'RushTd',
    'LG.1'                   : 'LongestRush',
    'TAR'                    : 'RecTargets',
    'REC'                    : 'Receptions',
    'YDS.3'                  : 'RecYards',
    'TD.2'                   : 'RecTd',
    'LG.2'                   : 'LongestRec',
    'FMB'                    : 'TotalFumbles',
    'FL'                     : 'FumblesLost',
    'INT.1'                  : 'DefInt',
    'YDS.4'                  : 'DefIntYards',
    'TD.3'                   : 'DefIntTd',#
    'LG.3'                   : 'DefLongestInt',
    'SK.1'                   : 'DefSacks',
    'TKL'                    : 'DefTackles',
    'AST'                    : 'DefTackleAst',
    'FR'                     : 'FumbleRec',
    'YDS.5'                  : 'DefFumbleYards',
    'TD.4'                   : 'FumbleRecTd',
    'FF'                     : 'DefForcedFumbles',
    'RT'                     : 'KickoffReturns',
    'YDS.6'                  : 'KickoffReturnYards',
    'TD.5'                   : 'KickoffRetTd',
    'LG.4'                   : 'LongestKickoffRet',
    'RET'                    : 'PuntsReturned',
    'YDS.7'                  : 'PuntReturnYards',
    'TD.6'                   : 'PuntReturnTd',
    'LG.5'                   : 'LongestPuntReturn',
    'XPM'                    : 'XPMade',
    'XPA'                    : 'XPAtt',
    'PASS'                   : 'TwoPointConvPass',
    'SCORED'                 : 'TwoPointConvRecRush',
    'FGM'                    : 'FgMade',
    'FGA'                    : 'FgAtt',
    'FGM\n0-19'              : 'FgMade0_19',
    'FGA\n0-19'              : 'FgAtt0_19',
    'FGM\n20-29'             : 'FgMade20_29', 
    'FGA\n20-29'             : 'FgAtt20_29',
    'FGM\n30-39'             : 'FgMade30_39',
    'FGA\n30-39'             : 'FgAtt30_39',
    'FGM\n40-49'             : 'FgMade40_49',
    'FGA\n40-49'             : 'FgAtt40_49',  
    'FGM\n50+'               : 'FgMade50plus',
    'FGA\n50+'               : 'FgAtt50plus',
    'SFTY'                   : 'DefSafeties',
    'PNT'                    : 'PuntsKicked',
    'YDS.8'                  : 'PuntsKickedYards',
    'LG.6'                   : 'LongestKickedPunt',
    'SNAPS'                  : 'OffSnaps',
    'PCT'                    : 'OffSnapsPct',
    'SNAPS.1'                : 'DefSnaps',
    'PCT.1'                  : 'DefSnapsPct',
    'SNAPS.2'                : 'StSnaps',
    'PCT.2'                  : 'StSnapsPct'
}

PLAYER_INFO_COLS = ['GameId','GameId2','GameDate','Season','PlayerId','Player','TeamAbbr']

# player-passing
PLAYER_PASSING_BASE_STAT_COLS = ['PassComp','PassAtt','PassYards','PassTd','PassInt',
                                 'TimesSacked','SackYardsLost','LongestPassComp']
PLAYER_PASSING_EXTRA_COLS = ['OffSnaps']
PLAYER_PASSING_TEAM_COLS = ['TeamPassAtt','TeamPassComp','TeamTotalPlays']
PLAYER_PASSING_STAT_COLS = PLAYER_PASSING_BASE_STAT_COLS + PLAYER_PASSING_TEAM_COLS + PLAYER_PASSING_EXTRA_COLS
PLAYER_PASSING_COLS = PLAYER_INFO_COLS + PLAYER_PASSING_STAT_COLS

# player-rushing
PLAYER_RUSHING_BASE_STAT_COLS = ['RushAtt','RushYards','RushTd','LongestRush']
PLAYER_RUSHING_EXTRA_COLS = ['OffSnaps']
PLAYER_RUSHING_TEAM_COLS = ['TeamRushAtt','TeamTotalPlays']
PLAYER_RUSHING_STAT_COLS = PLAYER_RUSHING_BASE_STAT_COLS + PLAYER_RUSHING_TEAM_COLS + PLAYER_RUSHING_EXTRA_COLS
PLAYER_RUSHING_COLS = PLAYER_INFO_COLS + PLAYER_RUSHING_STAT_COLS

# player-defense
PLAYER_DEFENSE_STAT_COLS = ['DefInt','DefIntYards','DefIntTd','DefLongestInt','DefSacks',
                            'DefTackles','DefTackleAst','FumbleRec','DefFumbleYards']
PLAYER_DEFENSE_COLS = PLAYER_INFO_COLS + PLAYER_DEFENSE_STAT_COLS

# player-kicking
PLAYER_KICKING_STAT_COLS = ['XPMade','XPAtt','FgMade','FgAtt','FgMade0_19',
                            'FgAtt0_19','FgMade20_29', 'FgAtt20_29','FgMade30_39',
                            'FgAtt30_39','FgMade40_49','FgAtt40_49', 'FgMade50plus',
                            'FgAtt50plus','PuntsKicked','PuntsKickedYards','LongestKickedPunt']
PLAYER_KICKING_COLS = PLAYER_INFO_COLS + PLAYER_KICKING_STAT_COLS

# player-snaps
# Include some extra columns for feature calculations (we will remove later)
PLAYER_SNAPS_EXTRA_COLS = ['PassAtt','PassComp','PassYards',
                           'RushAtt','RushYards',
                           'RecTargets','RecYards']
PLAYER_SNAPS_STAT_COLS = ['OffSnaps','OffSnapsPct','DefSnaps','DefSnapsPct',
                          'StSnaps','StSnapsPct'] + PLAYER_SNAPS_EXTRA_COLS
PLAYER_SNAPS_COLS = PLAYER_INFO_COLS + PLAYER_SNAPS_STAT_COLS

# player-receiving
PLAYER_RECEIVING_BASE_STAT_COLS = ['RecTargets','Receptions','RecYards','RecTd','LongestRec']
PLAYER_RECEIVING_EXTRA_COLS = ['OffSnaps']
PLAYER_RECEIVING_TEAM_COLS = ['TeamPassYards','TeamPassComp','TeamPassAtt','TeamTotalPlays']
PLAYER_RECEIVING_STAT_COLS = PLAYER_RECEIVING_BASE_STAT_COLS + PLAYER_RECEIVING_EXTRA_COLS + PLAYER_RECEIVING_TEAM_COLS
PLAYER_RECEIVING_COLS = PLAYER_INFO_COLS + PLAYER_RECEIVING_STAT_COLS

# player-positions
PLAYER_POSITIONS_COLS = ['PlayerId','Player','TeamAbbr','GameDate','Position','StarterFlag']

# BEGIN TEAM COLUMNS
RAW_TEAM_STATS_COLS = {
    'BIGDATABALL\nDATASET'             : 'DatasetName',
    'DATASET'                          : 'DatasetName',
    'GAME-ID'                          : 'GameId',
    'DATE'                             : 'GameDate',
    'WEEK\n#'                          : 'Week',
    'WEEK#'                            : 'Week',
    'WEEK'                             : 'Week',
    'START\nTIME\n(ET)'                : 'StartTime',
    'TEAM'                             : 'Team',
    'VENUE'                            : 'Venue',
    '1'                                : 'Q1',
    '2'                                : 'Q2',
    '3'                                : 'Q3',
    '4'                                : 'Q4',
    'OT'                               : 'OT',
    'FINAL'                            : 'FinalScore',
    'FIRST DOWNS'                      : 'TeamFirstDowns',
    'RUSH'                             : 'TeamRushAtt',
    'YDS'                              : 'TeamRushYards',
    'TD'                               : 'TeamRushTd',
    'COMP'                             : 'TeamPassComp',
    'ATT'                              : 'TeamPassAtt',
    'YDS.1'                            : 'TeamPassYards',
    'TD.1'                             : 'TeamPassTd',
    'INT'                              : 'TeamPassInt',
    'SACKED'                           : 'TeamTimesSacked',
    'YARDS'                            : 'TeamSackYardsLost',
    'NET \nPASS \nYARDS'               : 'TeamNetPassYards',
    'TOTAL \nYARDS'                    : 'TeamTotalYards',
    'FUMBLES'                          : 'TeamTotalFumbles',
    'LOST'                             : 'TeamFumblesLost',
    'TURNOVERS'                        : 'TeamTotalTurnovers',
    'PENALTIES'                        : 'TeamPenalties',
    'YARDS.1'                          : 'TeamPenaltyYards',
    'THIRD \nDOWNs \nMADE'             : 'Team3rdDownMade',
    'THIRD DOWNs ATTEMPTED'            : 'Team3rdDownAtt',
    'FOURTH DOWNs MADE'                : 'Team4thDownMade',
    'FOURTH DOWNs ATTEMPTED'           : 'Team4thDownAtt',
    'TOTAL \nPLAYS'                    : 'TeamTotalPlays',
    'TIME \nOF \nPOSSESSION'           : 'TeamToP',
    'SACKS'                            : 'TeamDefSacks',
    'OPPONENT FUMBLES RECOVERED'       : 'TeamDefFumbleRec',
    'DEFENSIVE FUMBLE RECOVERY TD'     : 'TeamDefFumbleRecTd',
    'INTERCEPTION RETURN TD'           : 'TeamDefIntTd',
    'BLOCKED PUNT/FG RETURN TD'        : 'TeamBlockPuntFgRetTd',
    'PUNT/KICKOFF/FG RETURN TD'        : 'TeamPuntKickoffFgRetTd',
    'EXTRA POINT RETURN'               : 'TeamExtraPointRet',
    'DEFENSIVE 2PT CONVERSION RETURN'  : 'TeamDef2PtConvRet',
    'SAFETIES'                         : 'TeamDefSafeties',
    'BLOCKED KICK/PUNT'                : 'TeamBlockKickPunt',
    'INTERCEPTIONS MADE'               : 'TeamDefInt',
    '2P CONVERSIONS MADE'              : 'TeamTwoPointConversionsMade',
    'EXTRA POINTS MADE'                : 'TeamExtraPointsMade',
    'FIELD GOALS MADE'                 : 'TeamFgMade',
    'POINTS ALLOWED \nBY DEFENSE'      : 'TeamDefPointsAllowed',
    'OPENING ODDS'                     : 'OpeningOdds',
    'OPENING SPREAD'                   : 'OpeningSpread',
    'OPENING TOTAL'                    : 'OpeningTotal',
    'LINE \nMOVEMENTS\n(#1)'           : 'LineMovement1',
    'LINE \nMOVEMENTS\n(#2)'           : 'LineMovement2',
    'LINE \nMOVEMENTS\n(#3)'           : 'LineMovement3',
    'CLOSING ODDS'                     : 'ClosingOdds',
    'CLOSING SPREAD'                   : 'ClosingSpread',
    'CLOSING TOTAL'                    : 'ClosingTotal',
    'MONEYLINE'                        : 'ClosingMoneyline',
    'OPENING\nMONEYLINE'               : 'OpeningMoneyline',
    'CLOSING\nMONEYLINE'               : 'ClosingMoneyline',
    'HALFTIME'                         : 'Halftime'
}

TEAM_INFO_COLS = ['GameId','GameId2','GameDate','Season','TeamAbbr']

# team-passing
TEAM_PASSING_STAT_COLS = ['TeamPassComp','TeamPassAtt','TeamPassYards','TeamPassTd','TeamPassInt',
                          'TeamTimesSacked','TeamSackYardsLost','TeamNetPassYards']
#TEAM_PASSING_START_COLS = TEAM_INFO_COLS + TEAM_PASSING_STAT_COLS
TEAM_PASSING_WEIGHTED_COLS = ['TeamPassCompPct','TeamPassIntPct','TeamPassTdPct','TeamPassYardsPerComp',
                              'TeamPassYardsPerAtt','TeamAdjPassYardsPerAtt']
TEAM_PASSING_COLS = TEAM_INFO_COLS + TEAM_PASSING_STAT_COLS                             
#TEAM_PASSING_COLS = TEAM_INFO_COLS + TEAM_PASSING_STAT_COLS + TEAM_PASSING_WEIGHTED_COLS                              

# team-rushing
TEAM_RUSHING_STAT_COLS = ['TeamRushAtt','TeamRushYards','TeamRushTd']
TEAM_RUSHING_COLS = TEAM_INFO_COLS + TEAM_RUSHING_STAT_COLS

# team-defense
# These are actually offensive cols that will be flipped to reflect team defense allowed 
TEAM_STAT_COLS = TEAM_PASSING_STAT_COLS + TEAM_RUSHING_STAT_COLS + ['TeamTotalPlays','TeamToP']
TEAM_DEFENSE_STAT_COLS = TEAM_STAT_COLS + ['TeamDefSacks','TeamDefFumbleRec','TeamDefFumbleRecTd','TeamDefIntTd','TeamDefSafeties','TeamDefInt', 'TeamDefPointsAllowed']
TEAM_DEFENSE_ALLOWED_STAT_COLS = [i + 'Allowed' for i in TEAM_STAT_COLS] + ['TeamDefSacks','TeamDefFumbleRec','TeamDefFumbleRecTd','TeamDefIntTd','TeamDefSafeties','TeamDefInt','TeamDefPointsAllowed']
TEAM_DEFENSE_COLS = TEAM_INFO_COLS + TEAM_DEFENSE_STAT_COLS

# team-plays
TEAM_PLAYS_EXTRA_COLS = ['TeamRushAtt','TeamPassAtt']
TEAM_PLAYS_STAT_COLS = TEAM_PLAYS_EXTRA_COLS + ['TeamTotalPlays','TeamToP','Team3rdDownMade','Team3rdDownAtt',
                                                'Team4thDownAtt','Team4thDownMade']
TEAM_PLAYS_COLS = TEAM_INFO_COLS + TEAM_PLAYS_STAT_COLS

# team-receiving
TEAM_RECEIVING_STAT_COLS = ['TeamRecTargets','TeamReceptions','TeamRecYards','TeamRecTd','TeamLongestRec']
TEAM_RECEIVING_COLS = TEAM_INFO_COLS + TEAM_RECEIVING_STAT_COLS

# LEAGUE COLUMNS
LEAGUE_STAT_COLS = [i.replace('Team','League') for i in TEAM_STAT_COLS]
LEAGUE_INFO_COLS = ['GameDate','Season']

# league-passing
LEAGUE_PASSING_STAT_COLS = ['LeaguePassComp', 'LeaguePassAtt', 'LeaguePassYards', 
                            'LeaguePassTd', 'LeaguePassInt', 'LeagueTimesSacked', 
                            'LeagueSackYardsLost', 'LeagueNetPassYards']
LEAGUE_PASSING_WEIGHTED_COLS = ['LeaguePassCompPct','LeaguePassIntPct','LeaguePassTdPct','LeaguePassYardsPerComp',
                                'LeaguePassYardsPerAtt','LeagueAdjPassYardsPerAtt']                            
LEAGUE_PASSING_COLS = LEAGUE_INFO_COLS + LEAGUE_PASSING_STAT_COLS

# league-rushing
LEAGUE_RUSHING_STAT_COLS = ['LeagueRushAtt','LeagueRushYards','LeagueRushTd']
LEAGUE_RUSHING_WEIGHTED_COLS = ['LeagueYardsPerRush','LeagueRushTdPct']                            
LEAGUE_RUSHING_COLS = LEAGUE_INFO_COLS + LEAGUE_RUSHING_STAT_COLS

# league-plays
LEAGUE_PLAYS_EXTRA_COLS = ['LeagueRushAtt','LeaguePassAtt']
LEAGUE_PLAYS_STAT_COLS = LEAGUE_PLAYS_EXTRA_COLS + ['LeagueTotalPlays','LeagueToP','League3rdDownMade','League3rdDownAtt',
                                                    'League4thDownAtt','League4thDownMade']
LEAGUE_PLAYS_WEIGHTED_COLS = ['LeagueRushPlayPct','LeaguePassPlayPct','League3rdDownPct','League4thDownPct']
LEAGUE_PLAYS_COLS = LEAGUE_INFO_COLS + LEAGUE_PLAYS_STAT_COLS

# league-receiving
LEAGUE_RECEIVING_STAT_COLS = ['LeagueRecTargets','LeagueReceptions','LeagueRecYards','LeagueRecTd','LeagueLongestRec']
LEAGUE_RECEIVING_COLS = LEAGUE_INFO_COLS + LEAGUE_RECEIVING_STAT_COLS

# league-defense
# These are actually offensive cols that will be flipped to reflect team defense allowed 
#LEAGUE_STAT_COLS = LEAGUE_PASSING_STAT_COLS + LEAGUE_RUSHING_STAT_COLS + ['LeagueTotalPlays','LeagueToP']
#LEAGUE_DEFENSE_STAT_COLS = LEAGUE_STAT_COLS + ['LeagueDefSacks','LeagueDefFumbleRec','LeagueDefFumbleRecTd','LeagueDefIntTd','LeagueDefSafeties','LeagueDefInt', 'DefLeaguePointsAllowed']
#LEAGUE_DEFENSE_STAT_COLS = [i + 'Allowed' for i in LEAGUE_DEFENSE_STAT_COLS]
#LEAGUE_DEFENSE_ALLOWED_STAT_COLS = [i + 'Allowed' for i in LEAGUE_STAT_COLS] + ['LeagueDefSacks','LeagueDefFumbleRec','LeagueDefFumbleRecTd','LeagueDefIntTd','LeagueDefSafeties','LeagueDefInt','DefLeaguePointsAllowed']
#LEAGUE_DEFENSE_COLS = LEAGUE_INFO_COLS + LEAGUE_DEFENSE_STAT_COLS

GAME_INFO_COLS = ['GameId','GameId2','GameDate','Season','Week', 
                  'GameType','HomeFlag','RoadFlag','NeutralFlag',
                  'Team','TeamAbbr','OppAbbr','Venue',
                  'Q1','Q2','Q3','Q4','OT','FinalScore',
                  'OpeningSpread','ClosingSpread','OpeningTotal','ClosingTotal','ClosingMoneyline']
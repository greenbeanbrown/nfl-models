-- PRODUCTION
WITH LatestPlayerReceiving AS (
    SELECT 
        prec.*,  
        ppos.Position,
        ppos.PositionId,
        ROW_NUMBER() OVER(PARTITION BY prec.PlayerId ORDER BY prec.GameDate DESC) AS RowNum
    FROM player_receiving AS prec
    JOIN player_positions AS ppos
        ON prec.PlayerId = ppos.PlayerId
        AND prec.Player = ppos.Player
        AND prec.GameDate = ppos.GameDate
        AND prec.TeamAbbr = ppos.TeamAbbr
    WHERE ppos.Position IN ("WR")
)

SELECT 
    -- latest-player-receiving
    lprec.*,  -- Select all columns from player_rushing
    
    -- team-passing
    tpas."3GameAvgTeamNetPassYards",
    tpas."3GameAvgTeamPassAtt",
    tpas."3GameAvgTeamPassComp",
    tpas."3GameAvgTeamPassInt",
    tpas."3GameAvgTeamPassTd",
    tpas."3GameAvgTeamPassYards",
    tpas."3GameAvgTeamSackYardsLost",
    tpas."3GameAvgTeamTimesSacked",
    tpas."3GameMedianTeamNetPassYards",
    tpas."3GameMedianTeamPassAtt",
    tpas."3GameMedianTeamPassComp",
    tpas."3GameMedianTeamPassInt",
    tpas."3GameMedianTeamPassTd",
    tpas."3GameMedianTeamPassYards",
    tpas."3GameMedianTeamSackYardsLost",
    tpas."3GameMedianTeamTimesSacked",
    tpas."3GameStdTeamNetPassYards",
    tpas."3GameStdTeamPassAtt",
    tpas."3GameStdTeamPassComp",
    tpas."3GameStdTeamPassInt",
    tpas."3GameStdTeamPassTd",
    tpas."3GameStdTeamPassYards",
    tpas."3GameStdTeamSackYardsLost",
    tpas."3GameStdTeamTimesSacked",
    tpas."3GameTeamAdjPassYardsPerAtt",
    tpas."3GameTeamPassCompPct",
    tpas."3GameTeamPassIntPct",
    tpas."3GameTeamPassTdPct",
    tpas."3GameTeamPassYardsPerAtt",
    tpas."3GameTeamPassYardsPerComp",
    tpas."6GameAvgTeamNetPassYards",
    tpas."6GameAvgTeamPassAtt",
    tpas."6GameAvgTeamPassComp",
    tpas."6GameAvgTeamPassInt",
    tpas."6GameAvgTeamPassTd",
    tpas."6GameAvgTeamPassYards",
    tpas."6GameAvgTeamSackYardsLost",
    tpas."6GameAvgTeamTimesSacked",
    tpas."6GameMedianTeamNetPassYards",
    tpas."6GameMedianTeamPassAtt",
    tpas."6GameMedianTeamPassComp",
    tpas."6GameMedianTeamPassInt",
    tpas."6GameMedianTeamPassTd",
    tpas."6GameMedianTeamPassYards",
    tpas."6GameMedianTeamSackYardsLost",
    tpas."6GameMedianTeamTimesSacked",
    tpas."6GameStdTeamNetPassYards",
    tpas."6GameStdTeamPassAtt",
    tpas."6GameStdTeamPassComp",
    tpas."6GameStdTeamPassInt",
    tpas."6GameStdTeamPassTd",
    tpas."6GameStdTeamPassYards",
    tpas."6GameStdTeamSackYardsLost",
    tpas."6GameStdTeamTimesSacked",
    tpas."6GameTeamAdjPassYardsPerAtt",
    tpas."6GameTeamPassCompPct",
    tpas."6GameTeamPassIntPct",
    tpas."6GameTeamPassTdPct",
    tpas."6GameTeamPassYardsPerAtt",
    tpas."6GameTeamPassYardsPerComp",
    tpas.CareerAvgTeamNetPassYards,
    tpas.CareerAvgTeamPassAtt,
    tpas.CareerAvgTeamPassComp,
    tpas.CareerAvgTeamPassInt,
    tpas.CareerAvgTeamPassTd,
    tpas.CareerAvgTeamPassYards,
    tpas.CareerAvgTeamSackYardsLost,
    tpas.CareerAvgTeamTimesSacked,
    tpas.CareerMedianTeamNetPassYards,
    tpas.CareerMedianTeamPassAtt,
    tpas.CareerMedianTeamPassComp,
    tpas.CareerMedianTeamPassInt,
    tpas.CareerMedianTeamPassTd,
    tpas.CareerMedianTeamPassYards,
    tpas.CareerMedianTeamSackYardsLost,
    tpas.CareerMedianTeamTimesSacked,
    tpas.CareerStdTeamNetPassYards,
    tpas.CareerStdTeamPassAtt,
    tpas.CareerStdTeamPassComp,
    tpas.CareerStdTeamPassInt,
    tpas.CareerStdTeamPassTd,
    tpas.CareerStdTeamPassYards,
    tpas.CareerStdTeamSackYardsLost,
    tpas.CareerStdTeamTimesSacked,
    tpas.CareerTeamAdjPassYardsPerAtt,
    tpas.CareerTeamPassCompPct,
    tpas.CareerTeamPassIntPct,
    tpas.CareerTeamPassTdPct,
    tpas.CareerTeamPassYardsPerAtt,
    tpas.CareerTeamPassYardsPerComp,
    tpas.Season3GameTeamAdjPassYardsPerAtt,
    tpas.Season3GameTeamPassCompPct,
    tpas.Season3GameTeamPassIntPct,
    tpas.Season3GameTeamPassTdPct,
    tpas.Season3GameTeamPassYardsPerAtt,
    tpas.Season3GameTeamPassYardsPerComp,
    tpas.Season6GameTeamAdjPassYardsPerAtt,
    tpas.Season6GameTeamPassCompPct,
    tpas.Season6GameTeamPassIntPct,
    tpas.Season6GameTeamPassTdPct,
    tpas.Season6GameTeamPassYardsPerAtt,
    tpas.Season6GameTeamPassYardsPerComp,
    tpas.SeasonAvgTeamNetPassYards,
    tpas.SeasonAvgTeamPassAtt,
    tpas.SeasonAvgTeamPassComp,
    tpas.SeasonAvgTeamPassInt,
    tpas.SeasonAvgTeamPassTd,
    tpas.SeasonAvgTeamPassYards,
    tpas.SeasonAvgTeamSackYardsLost,
    tpas.SeasonAvgTeamTimesSacked,
    tpas.SeasonMedianTeamNetPassYards,
    tpas.SeasonMedianTeamPassAtt,
    tpas.SeasonMedianTeamPassComp,
    tpas.SeasonMedianTeamPassInt,
    tpas.SeasonMedianTeamPassTd,
    tpas.SeasonMedianTeamPassYards,
    tpas.SeasonMedianTeamSackYardsLost,
    tpas.SeasonMedianTeamTimesSacked,
    tpas.SeasonStdTeamNetPassYards,
    tpas.SeasonStdTeamPassAtt,
    tpas.SeasonStdTeamPassComp,
    tpas.SeasonStdTeamPassInt,
    tpas.SeasonStdTeamPassTd,
    tpas.SeasonStdTeamPassYards,
    tpas.SeasonStdTeamSackYardsLost,
    tpas.SeasonStdTeamTimesSacked,
    tpas.SeasonTeamAdjPassYardsPerAtt,
    tpas.SeasonTeamPassCompPct,
    tpas.SeasonTeamPassIntPct,
    tpas.SeasonTeamPassTdPct,
    tpas.SeasonTeamPassYardsPerAtt,
    tpas.SeasonTeamPassYardsPerComp,
    tpas.TeamAdjPassYardsPerAtt,
    tpas.TeamNetPassYards,
    tpas.TeamPassAtt,
    tpas.TeamPassComp,
    tpas.TeamPassCompPct,
    tpas.TeamPassInt,
    tpas.TeamPassIntPct,
    tpas.TeamPassTd,
    tpas.TeamPassTdPct,
    tpas.TeamPassYards,
    tpas.TeamPassYardsPerAtt,
    tpas.TeamPassYardsPerComp,
    tpas.TeamSackYardsLost,
    tpas.TeamTimesSacked,

    -- team-rushing
    trus."3GameAvgTeamRushAtt",
    trus."3GameAvgTeamRushTd",
    trus."3GameAvgTeamRushYards",
    trus."3GameMedianTeamRushAtt",
    trus."3GameMedianTeamRushTd",
    trus."3GameMedianTeamRushYards",
    trus."3GameTeamRushTdPct",
    trus."3GameTeamYardsPerRush",
    trus."6GameAvgTeamRushAtt",
    trus."6GameAvgTeamRushTd",
    trus."6GameAvgTeamRushYards",
    trus."6GameMedianTeamRushAtt",
    trus."6GameMedianTeamRushTd",
    trus."6GameMedianTeamRushYards",
    trus."6GameTeamRushTdPct",
    trus."6GameTeamYardsPerRush",
    trus.CareerAvgTeamRushAtt,
    trus.CareerAvgTeamRushTd,
    trus.CareerAvgTeamRushYards,
    trus.CareerMedianTeamRushAtt,
    trus.CareerMedianTeamRushTd,
    trus.CareerMedianTeamRushYards,
    trus.CareerTeamRushTdPct,
    trus.CareerTeamYardsPerRush,
    trus.Season3GameTeamRushTdPct,
    trus.Season3GameTeamYardsPerRush,
    trus.Season6GameTeamRushTdPct,
    trus.Season6GameTeamYardsPerRush,
    trus.SeasonAvgTeamRushAtt,
    trus.SeasonAvgTeamRushTd,
    trus.SeasonAvgTeamRushYards,
    trus.SeasonMedianTeamRushAtt,
    trus.SeasonMedianTeamRushTd,
    trus.SeasonMedianTeamRushYards,
    trus.SeasonTeamRushTdPct,
    trus.SeasonTeamYardsPerRush,
    trus.TeamRushAtt,
    trus.TeamRushTd,
    trus.TeamRushTdPct,
    trus.TeamRushYards,
    trus.TeamYardsPerRush,    
    
    -- team-receiving
    trec."3GameAvgTeamLongestRec",
    trec."3GameAvgTeamRecTargets",
    trec."3GameAvgTeamRecTd",
    trec."3GameAvgTeamRecYards",
    trec."3GameAvgTeamReceptions",
    trec."3GameMedianTeamLongestRec",
    trec."3GameMedianTeamRecTargets",
    trec."3GameMedianTeamRecTd",
    trec."3GameMedianTeamRecYards",
    trec."3GameMedianTeamReceptions",
    trec."3GameTeamCatchPct",
    trec."3GameTeamYardsPerRec",
    trec."3GameTeamYardsPerRecTarget",
    trec."6GameAvgTeamLongestRec",
    trec."6GameAvgTeamRecTargets",
    trec."6GameAvgTeamRecTd",
    trec."6GameAvgTeamRecYards",
    trec."6GameAvgTeamReceptions",
    trec."6GameMedianTeamLongestRec",
    trec."6GameMedianTeamRecTargets",
    trec."6GameMedianTeamRecTd",
    trec."6GameMedianTeamRecYards",
    trec."6GameMedianTeamReceptions",
    trec."6GameTeamCatchPct",
    trec."6GameTeamYardsPerRec",
    trec."6GameTeamYardsPerRecTarget",
    trec.CareerAvgTeamLongestRec,
    trec.CareerAvgTeamRecTargets,
    trec.CareerAvgTeamRecTd,
    trec.CareerAvgTeamRecYards,
    trec.CareerAvgTeamReceptions,
    trec.CareerMedianTeamLongestRec,
    trec.CareerMedianTeamRecTargets,
    trec.CareerMedianTeamRecTd,
    trec.CareerMedianTeamRecYards,
    trec.CareerMedianTeamReceptions,
    trec.CareerTeamCatchPct,
    trec.CareerTeamYardsPerRec,
    trec.CareerTeamYardsPerRecTarget,
    trec.Season3GameTeamCatchPct,
    trec.Season3GameTeamYardsPerRec,
    trec.Season3GameTeamYardsPerRecTarget,
    trec.Season6GameTeamCatchPct,
    trec.Season6GameTeamYardsPerRec,
    trec.Season6GameTeamYardsPerRecTarget,
    trec.SeasonAvgTeamLongestRec,
    trec.SeasonAvgTeamRecTargets,
    trec.SeasonAvgTeamRecTd,
    trec.SeasonAvgTeamRecYards,
    trec.SeasonAvgTeamReceptions,
    trec.SeasonMedianTeamLongestRec,
    trec.SeasonMedianTeamRecTargets,
    trec.SeasonMedianTeamRecTd,
    trec.SeasonMedianTeamRecYards,
    trec.SeasonMedianTeamReceptions,
    trec.SeasonTeamCatchPct,
    trec.SeasonTeamYardsPerRec,
    trec.SeasonTeamYardsPerRecTarget,
    trec.TeamCatchPct,
    trec.TeamLongestRec,
    trec.TeamRecTargets,
    trec.TeamRecTd,
    trec.TeamRecYards,
    trec.TeamReceptions,
    trec.TeamYardsPerRec,
    trec.TeamYardsPerRecTarget,    
    
    -- team-plays
    tpla."3GameAvgTeam3rdDownAtt",
    tpla."3GameAvgTeam3rdDownMade",
    tpla."3GameAvgTeam4thDownAtt",
    tpla."3GameAvgTeam4thDownMade",
    tpla."3GameAvgTeamToP",
    tpla."3GameAvgTeamTotalPlays",
    tpla."3GameMedianTeam3rdDownAtt",
    tpla."3GameMedianTeam3rdDownMade",
    tpla."3GameMedianTeam4thDownAtt",
    tpla."3GameMedianTeam4thDownMade",
    tpla."3GameMedianTeamToP",
    tpla."3GameMedianTeamTotalPlays",
    tpla."3GameTeam3rdDownPct",
    tpla."3GameTeam4thDownPct",
    tpla."3GameTeamPassPlayPct",
    tpla."3GameTeamRushPlayPct",
    tpla."6GameAvgTeam3rdDownAtt",
    tpla."6GameAvgTeam3rdDownMade",
    tpla."6GameAvgTeam4thDownAtt",
    tpla."6GameAvgTeam4thDownMade",
    tpla."6GameAvgTeamToP",
    tpla."6GameAvgTeamTotalPlays",
    tpla."6GameMedianTeam3rdDownAtt",
    tpla."6GameMedianTeam3rdDownMade",
    tpla."6GameMedianTeam4thDownAtt",
    tpla."6GameMedianTeam4thDownMade",
    tpla."6GameMedianTeamToP",
    tpla."6GameMedianTeamTotalPlays",
    tpla."6GameTeam3rdDownPct",
    tpla."6GameTeam4thDownPct",
    tpla."6GameTeamPassPlayPct",
    tpla."6GameTeamRushPlayPct",
    tpla.CareerAvgTeam3rdDownAtt,
    tpla.CareerAvgTeam3rdDownMade,
    tpla.CareerAvgTeam4thDownAtt,
    tpla.CareerAvgTeam4thDownMade,
    tpla.CareerAvgTeamToP,
    tpla.CareerAvgTeamTotalPlays,
    tpla.CareerMedianTeam3rdDownAtt,
    tpla.CareerMedianTeam3rdDownMade,
    tpla.CareerMedianTeam4thDownAtt,
    tpla.CareerMedianTeam4thDownMade,
    tpla.CareerMedianTeamToP,
    tpla.CareerMedianTeamTotalPlays,
    tpla.CareerTeam3rdDownPct,
    tpla.CareerTeam4thDownPct,
    tpla.CareerTeamPassPlayPct,
    tpla.CareerTeamRushPlayPct,
    tpla.Season3GameTeam3rdDownPct,
    tpla.Season3GameTeam4thDownPct,
    tpla.Season3GameTeamPassPlayPct,
    tpla.Season3GameTeamRushPlayPct,
    tpla.Season6GameTeam3rdDownPct,
    tpla.Season6GameTeam4thDownPct,
    tpla.Season6GameTeamPassPlayPct,
    tpla.Season6GameTeamRushPlayPct,
    tpla.SeasonAvgTeam3rdDownAtt,
    tpla.SeasonAvgTeam3rdDownMade,
    tpla.SeasonAvgTeam4thDownAtt,
    tpla.SeasonAvgTeam4thDownMade,
    tpla.SeasonAvgTeamToP,
    tpla.SeasonAvgTeamTotalPlays,
    tpla.SeasonMedianTeam3rdDownAtt,
    tpla.SeasonMedianTeam3rdDownMade,
    tpla.SeasonMedianTeam4thDownAtt,
    tpla.SeasonMedianTeam4thDownMade,
    tpla.SeasonMedianTeamToP,
    tpla.SeasonMedianTeamTotalPlays,
    tpla.SeasonTeam3rdDownPct,
    tpla.SeasonTeam4thDownPct,
    tpla.SeasonTeamPassPlayPct,
    tpla.SeasonTeamRushPlayPct,
    tpla.Team3rdDownAtt,
    tpla.Team3rdDownMade,
    tpla.Team3rdDownPct,
    tpla.Team4thDownAtt,
    tpla.Team4thDownMade,
    tpla.Team4thDownPct,
    tpla.TeamPassPlayPct,
    tpla.TeamRushPlayPct,
    tpla.TeamToP,
    tpla.TeamTotalPlays,

    -- league-passing
    lpas."3GameAvgLeagueNetPassYards",
    lpas."3GameAvgLeaguePassAtt",
    lpas."3GameAvgLeaguePassComp",
    lpas."3GameAvgLeaguePassInt",
    lpas."3GameAvgLeaguePassTd",
    lpas."3GameAvgLeaguePassYards",
    lpas."3GameAvgLeagueSackYardsLost",
    lpas."3GameAvgLeagueTimesSacked",
    lpas."3GameLeagueAdjPassYardsPerAtt",
    lpas."3GameLeaguePassCompPct",
    lpas."3GameLeaguePassIntPct",
    lpas."3GameLeaguePassTdPct",
    lpas."3GameLeaguePassYardsPerAtt",
    lpas."3GameLeaguePassYardsPerComp",
    lpas."6GameAvgLeagueNetPassYards",
    lpas."6GameAvgLeaguePassAtt",
    lpas."6GameAvgLeaguePassComp",
    lpas."6GameAvgLeaguePassInt",
    lpas."6GameAvgLeaguePassTd",
    lpas."6GameAvgLeaguePassYards",
    lpas."6GameAvgLeagueSackYardsLost",
    lpas."6GameAvgLeagueTimesSacked",
    lpas."6GameLeagueAdjPassYardsPerAtt",
    lpas."6GameLeaguePassCompPct",
    lpas."6GameLeaguePassIntPct",
    lpas."6GameLeaguePassTdPct",
    lpas."6GameLeaguePassYardsPerAtt",
    lpas."6GameLeaguePassYardsPerComp",
    lpas.CareerAvgLeagueNetPassYards,
    lpas.CareerAvgLeaguePassAtt,
    lpas.CareerAvgLeaguePassComp,
    lpas.CareerAvgLeaguePassInt,
    lpas.CareerAvgLeaguePassTd,
    lpas.CareerAvgLeaguePassYards,
    lpas.CareerAvgLeagueSackYardsLost,
    lpas.CareerAvgLeagueTimesSacked,
    lpas.CareerLeagueAdjPassYardsPerAtt,
    lpas.CareerLeaguePassCompPct,
    lpas.CareerLeaguePassIntPct,
    lpas.CareerLeaguePassTdPct,
    lpas.CareerLeaguePassYardsPerAtt,
    lpas.CareerLeaguePassYardsPerComp,
    lpas.LeagueAdjPassYardsPerAtt,
    lpas.LeagueNetPassYards,
    lpas.LeaguePassAtt,
    lpas.LeaguePassComp,
    lpas.LeaguePassCompPct,
    lpas.LeaguePassInt,
    lpas.LeaguePassIntPct,
    lpas.LeaguePassTd,
    lpas.LeaguePassTdPct,
    lpas.LeaguePassYards,
    lpas.LeaguePassYardsPerAtt,
    lpas.LeaguePassYardsPerComp,
    lpas.LeagueSackYardsLost,
    lpas.LeagueTimesSacked,
    lpas.Season3GameAvgLeagueNetPassYards,
    lpas.Season3GameAvgLeaguePassAtt,
    lpas.Season3GameAvgLeaguePassComp,
    lpas.Season3GameAvgLeaguePassInt,
    lpas.Season3GameAvgLeaguePassTd,
    lpas.Season3GameAvgLeaguePassYards,
    lpas.Season3GameAvgLeagueSackYardsLost,
    lpas.Season3GameAvgLeagueTimesSacked,
    lpas.Season3GameLeagueAdjPassYardsPerAtt,
    lpas.Season3GameLeaguePassCompPct,
    lpas.Season3GameLeaguePassIntPct,
    lpas.Season3GameLeaguePassTdPct,
    lpas.Season3GameLeaguePassYardsPerAtt,
    lpas.Season3GameLeaguePassYardsPerComp,
    lpas.Season6GameAvgLeagueNetPassYards,
    lpas.Season6GameAvgLeaguePassAtt,
    lpas.Season6GameAvgLeaguePassComp,
    lpas.Season6GameAvgLeaguePassInt,
    lpas.Season6GameAvgLeaguePassTd,
    lpas.Season6GameAvgLeaguePassYards,
    lpas.Season6GameAvgLeagueSackYardsLost,
    lpas.Season6GameAvgLeagueTimesSacked,
    lpas.Season6GameLeagueAdjPassYardsPerAtt,
    lpas.Season6GameLeaguePassCompPct,
    lpas.Season6GameLeaguePassIntPct,
    lpas.Season6GameLeaguePassTdPct,
    lpas.Season6GameLeaguePassYardsPerAtt,
    lpas.Season6GameLeaguePassYardsPerComp,
    lpas.SeasonAvgLeagueNetPassYards,
    lpas.SeasonAvgLeaguePassAtt,
    lpas.SeasonAvgLeaguePassComp,
    lpas.SeasonAvgLeaguePassInt,
    lpas.SeasonAvgLeaguePassTd,
    lpas.SeasonAvgLeaguePassYards,
    lpas.SeasonAvgLeagueSackYardsLost,
    lpas.SeasonAvgLeagueTimesSacked,
    lpas.SeasonLeagueAdjPassYardsPerAtt,
    lpas.SeasonLeaguePassCompPct,
    lpas.SeasonLeaguePassIntPct,
    lpas.SeasonLeaguePassTdPct,
    lpas.SeasonLeaguePassYardsPerAtt,
    lpas.SeasonLeaguePassYardsPerComp,    
    
    -- league-rushing
    lrus."3GameAvgLeagueRushAtt",
    lrus."3GameAvgLeagueRushTd",
    lrus."3GameAvgLeagueRushYards",
    lrus."3GameLeagueRushTdPct",
    lrus."3GameLeagueYardsPerRush",
    lrus."6GameAvgLeagueRushAtt",
    lrus."6GameAvgLeagueRushTd",
    lrus."6GameAvgLeagueRushYards",
    lrus."6GameLeagueRushTdPct",
    lrus."6GameLeagueYardsPerRush",
    lrus.CareerAvgLeagueRushAtt,
    lrus.CareerAvgLeagueRushTd,
    lrus.CareerAvgLeagueRushYards,
    lrus.CareerLeagueRushTdPct,
    lrus.CareerLeagueYardsPerRush,
    lrus.LeagueRushAtt,
    lrus.LeagueRushTd,
    lrus.LeagueRushTdPct,
    lrus.LeagueRushYards,
    lrus.LeagueYardsPerRush,
    lrus.Season3GameAvgLeagueRushAtt,
    lrus.Season3GameAvgLeagueRushTd,
    lrus.Season3GameAvgLeagueRushYards,
    lrus.Season3GameLeagueRushTdPct,
    lrus.Season3GameLeagueYardsPerRush,
    lrus.Season6GameAvgLeagueRushAtt,
    lrus.Season6GameAvgLeagueRushTd,
    lrus.Season6GameAvgLeagueRushYards,
    lrus.Season6GameLeagueRushTdPct,
    lrus.Season6GameLeagueYardsPerRush,
    lrus.SeasonAvgLeagueRushAtt,
    lrus.SeasonAvgLeagueRushTd,
    lrus.SeasonAvgLeagueRushYards,
    lrus.SeasonLeagueRushTdPct,
    lrus.SeasonLeagueYardsPerRush,
    
    -- league-receiving
    lrec."3GameAvgLeagueLongestRec",
    lrec."3GameAvgLeagueRecTargets",
    lrec."3GameAvgLeagueRecTd",
    lrec."3GameAvgLeagueRecYards",
    lrec."3GameAvgLeagueReceptions",
    lrec."3GameLeagueCatchPct",
    lrec."3GameLeagueYardsPerRec",
    lrec."3GameLeagueYardsPerRecTarget",
    lrec."6GameAvgLeagueLongestRec",
    lrec."6GameAvgLeagueRecTargets",
    lrec."6GameAvgLeagueRecTd",
    lrec."6GameAvgLeagueRecYards",
    lrec."6GameAvgLeagueReceptions",
    lrec."6GameLeagueCatchPct",
    lrec."6GameLeagueYardsPerRec",
    lrec."6GameLeagueYardsPerRecTarget",
    lrec.CareerAvgLeagueLongestRec,
    lrec.CareerAvgLeagueRecTargets,
    lrec.CareerAvgLeagueRecTd,
    lrec.CareerAvgLeagueRecYards,
    lrec.CareerAvgLeagueReceptions,
    lrec.CareerLeagueCatchPct,
    lrec.CareerLeagueYardsPerRec,
    lrec.CareerLeagueYardsPerRecTarget,
    lrec.LeagueCatchPct,
    lrec.LeagueLongestRec,
    lrec.LeagueRecTargets,
    lrec.LeagueRecTd,
    lrec.LeagueRecYards,
    lrec.LeagueReceptions,
    lrec.LeagueYardsPerRec,
    lrec.LeagueYardsPerRecTarget,
    lrec.Season3GameAvgLeagueLongestRec,
    lrec.Season3GameAvgLeagueRecTargets,
    lrec.Season3GameAvgLeagueRecTd,
    lrec.Season3GameAvgLeagueRecYards,
    lrec.Season3GameAvgLeagueReceptions,
    lrec.Season3GameLeagueCatchPct,
    lrec.Season3GameLeagueYardsPerRec,
    lrec.Season3GameLeagueYardsPerRecTarget,
    lrec.Season6GameAvgLeagueLongestRec,
    lrec.Season6GameAvgLeagueRecTargets,
    lrec.Season6GameAvgLeagueRecTd,
    lrec.Season6GameAvgLeagueRecYards,
    lrec.Season6GameAvgLeagueReceptions,
    lrec.Season6GameLeagueCatchPct,
    lrec.Season6GameLeagueYardsPerRec,
    lrec.Season6GameLeagueYardsPerRecTarget,
    lrec.SeasonAvgLeagueLongestRec,
    lrec.SeasonAvgLeagueRecTargets,
    lrec.SeasonAvgLeagueRecTd,
    lrec.SeasonAvgLeagueRecYards,
    lrec.SeasonAvgLeagueReceptions,
    lrec.SeasonLeagueCatchPct,
    lrec.SeasonLeagueYardsPerRec,
    lrec.SeasonLeagueYardsPerRecTarget,  
    
    -- league-plays
    lpla."3GameAvgLeague3rdDownAtt",
    lpla."3GameAvgLeague3rdDownMade",
    lpla."3GameAvgLeague4thDownAtt",
    lpla."3GameAvgLeague4thDownMade",
    lpla."3GameAvgLeagueToP",
    lpla."3GameAvgLeagueTotalPlays",
    lpla."3GameLeague3rdDownPct",
    lpla."3GameLeague4thDownPct",
    lpla."3GameLeaguePassPlayPct",
    lpla."3GameLeagueRushPlayPct",
    lpla."6GameAvgLeague3rdDownAtt",
    lpla."6GameAvgLeague3rdDownMade",
    lpla."6GameAvgLeague4thDownAtt",
    lpla."6GameAvgLeague4thDownMade",
    lpla."6GameAvgLeagueToP",
    lpla."6GameAvgLeagueTotalPlays",
    lpla."6GameLeague3rdDownPct",
    lpla."6GameLeague4thDownPct",
    lpla."6GameLeaguePassPlayPct",
    lpla."6GameLeagueRushPlayPct",
    lpla.CareerAvgLeague3rdDownAtt,
    lpla.CareerAvgLeague3rdDownMade,
    lpla.CareerAvgLeague4thDownAtt,
    lpla.CareerAvgLeague4thDownMade,
    lpla.CareerAvgLeagueToP,
    lpla.CareerAvgLeagueTotalPlays,
    lpla.CareerLeague3rdDownPct,
    lpla.CareerLeague4thDownPct,
    lpla.CareerLeaguePassPlayPct,
    lpla.CareerLeagueRushPlayPct,
    lpla.League3rdDownAtt,
    lpla.League3rdDownMade,
    lpla.League3rdDownPct,
    lpla.League4thDownAtt,
    lpla.League4thDownMade,
    lpla.League4thDownPct,
    lpla.LeaguePassPlayPct,
    lpla.LeagueRushPlayPct,
    lpla.LeagueToP,
    lpla.LeagueTotalPlays,
    lpla.Season3GameAvgLeague3rdDownAtt,
    lpla.Season3GameAvgLeague3rdDownMade,
    lpla.Season3GameAvgLeague4thDownAtt,
    lpla.Season3GameAvgLeague4thDownMade,
    lpla.Season3GameAvgLeagueToP,
    lpla.Season3GameAvgLeagueTotalPlays,
    lpla.Season3GameLeague3rdDownPct,
    lpla.Season3GameLeague4thDownPct,
    lpla.Season3GameLeaguePassPlayPct,
    lpla.Season3GameLeagueRushPlayPct,
    lpla.Season6GameAvgLeague3rdDownAtt,
    lpla.Season6GameAvgLeague3rdDownMade,
    lpla.Season6GameAvgLeague4thDownAtt,
    lpla.Season6GameAvgLeague4thDownMade,
    lpla.Season6GameAvgLeagueToP,
    lpla.Season6GameAvgLeagueTotalPlays,
    lpla.Season6GameLeague3rdDownPct,
    lpla.Season6GameLeague4thDownPct,
    lpla.Season6GameLeaguePassPlayPct,
    lpla.Season6GameLeagueRushPlayPct,
    lpla.SeasonAvgLeague3rdDownAtt,
    lpla.SeasonAvgLeague3rdDownMade,
    lpla.SeasonAvgLeague4thDownAtt,
    lpla.SeasonAvgLeague4thDownMade,
    lpla.SeasonAvgLeagueToP,
    lpla.SeasonAvgLeagueTotalPlays,
    lpla.SeasonLeague3rdDownPct,
    lpla.SeasonLeague4thDownPct,
    lpla.SeasonLeaguePassPlayPct,
    lpla.SeasonLeagueRushPlayPct

    
FROM LatestPlayerReceiving AS lprec

JOIN player_positions AS ppos
    ON lprec.PlayerId = ppos.PlayerId
    AND lprec.Player = ppos.Player
    AND lprec.GameDate = ppos.GameDate
    AND lprec.TeamAbbr = ppos.TeamAbbr

JOIN team_passing AS tpas
    ON lprec.GameId = tpas.GameId
    AND lprec.GameId2 = tpas.GameId2
    AND lprec.GameDate = tpas.GameDate
    AND lprec.TeamAbbr = tpas.TeamAbbr  

JOIN team_rushing AS trus
    ON lprec.GameId = trus.GameId
    AND lprec.GameId2 = trus.GameId2
    AND lprec.GameDate = trus.GameDate
    AND lprec.TeamAbbr = trus.TeamAbbr
    AND lprec.Season = trus.Season   

JOIN team_receiving AS trec
    ON lprec.GameId = trec.GameId
    AND lprec.GameId2 = trec.GameId2
    AND lprec.GameDate = trec.GameDate
    AND lprec.TeamAbbr = trec.TeamAbbr
    
JOIN team_plays AS tpla
    ON lprec.GameId = tpla.GameId
    AND lprec.GameId2 = tpla.GameId2
    AND lprec.GameDate = tpla.GameDate
    AND lprec.TeamAbbr = tpla.TeamAbbr    
    
JOIN league_passing AS lpas
    ON lprec.GameDate = lpas.GameDate  
    AND lprec.Season = lpas.Season   
 
JOIN league_rushing AS lrus
    ON lprec.GameDate = lrus.GameDate
    AND lprec.Season = lrus.Season     

JOIN league_receiving AS lrec
    ON lprec.GameDate = lrec.GameDate
    AND lprec.Season = lrec.Season   

JOIN league_plays AS lpla
    ON lprec.GameDate = lpla.GameDate      
    AND lprec.Season = lpla.Season  
     
WHERE lprec.RowNum = 1
  

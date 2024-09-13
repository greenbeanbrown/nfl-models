-- MODEL TRAINING
SELECT 
    -- Target Variable
    ppas.PassYards AS TargetPassYards,
    
    -- player-positions
    ppos.PlayerId,
    ppos.Player,
    ppos.GameDate,
    ppos.Position, 
    
    -- lag-player-passing
    lppas.Lag3GameAvgLongestPassComp,
    lppas.Lag3GameAvgPassAtt,
    lppas.Lag3GameAvgPassComp,
    lppas.Lag3GameAvgPassInt,
    lppas.Lag3GameAvgPassTd,
    lppas.Lag3GameAvgPassYards,
    lppas.Lag3GameAvgQBR,
    lppas.Lag3GameAvgSackYardsLost,
    lppas.Lag3GameAvgTimesSacked,
    lppas.Lag3GameMedianLongestPassComp,
    lppas.Lag3GameMedianPassAtt,
    lppas.Lag3GameMedianPassComp,
    lppas.Lag3GameMedianPassInt,
    lppas.Lag3GameMedianPassTd,
    lppas.Lag3GameMedianPassYards,
    lppas.Lag3GameMedianQBR,
    lppas.Lag3GameMedianSackYardsLost,
    lppas.Lag3GameMedianTimesSacked,
    lppas.Lag3GamePassAttPerTeamPlay,
    lppas.Lag3GamePassCompPct,
    lppas.Lag3GamePassCompPerTeamPlay,
    lppas.Lag3GamePassIntPct,
    lppas.Lag3GamePassIntPerTeamPlay,
    lppas.Lag3GamePassTdPct,
    lppas.Lag3GamePassTdPerTeamPlay,
    lppas.Lag3GamePassYardsPerAtt,
    lppas.Lag3GamePassYardsPerComp,
    lppas.Lag3GamePassYardsPerTeamPlay,
    lppas.Lag6GameAdjPassYardsPerAtt,
    lppas.Lag6GameAvgLongestPassComp,
    lppas.Lag6GameAvgPassAtt,
    lppas.Lag6GameAvgPassComp,
    lppas.Lag6GameAvgPassInt,
    lppas.Lag6GameAvgPassTd,
    lppas.Lag6GameAvgPassYards,
    lppas.Lag6GameAvgQBR,
    lppas.Lag6GameAvgSackYardsLost,
    lppas.Lag6GameAvgTimesSacked,
    lppas.Lag6GameMedianLongestPassComp,
    lppas.Lag6GameMedianPassAtt,
    lppas.Lag6GameMedianPassComp,
    lppas.Lag6GameMedianPassInt,
    lppas.Lag6GameMedianPassTd,
    lppas.Lag6GameMedianPassYards,
    lppas.Lag6GameMedianQBR,
    lppas.Lag6GameMedianSackYardsLost,
    lppas.Lag6GameMedianTimesSacked,
    lppas.Lag6GamePassAttPerTeamPlay,
    lppas.Lag6GamePassCompPct,
    lppas.Lag6GamePassCompPerTeamPlay,
    lppas.Lag6GamePassIntPct,
    lppas.Lag6GamePassIntPerTeamPlay,
    lppas.Lag6GamePassTdPct,
    lppas.Lag6GamePassTdPerTeamPlay,
    lppas.Lag6GamePassYardsPerAtt,
    lppas.Lag6GamePassYardsPerComp,
    lppas.Lag6GamePassYardsPerTeamPlay,
    lppas.LagCareerAdjPassYardsPerAtt,
    lppas.LagCareerAvgLongestPassComp,
    lppas.LagCareerAvgPassAtt,
    lppas.LagCareerAvgPassComp,
    lppas.LagCareerAvgPassInt,
    lppas.LagCareerAvgPassTd,
    lppas.LagCareerAvgPassYards,
    lppas.LagCareerAvgQBR,
    lppas.LagCareerAvgSackYardsLost,
    lppas.LagCareerAvgTimesSacked,
    lppas.LagCareerMedianLongestPassComp,
    lppas.LagCareerMedianPassAtt,
    lppas.LagCareerMedianPassComp,
    lppas.LagCareerMedianPassInt,
    lppas.LagCareerMedianPassTd,
    lppas.LagCareerMedianPassYards,
    lppas.LagCareerMedianQBR,
    lppas.LagCareerMedianSackYardsLost,
    lppas.LagCareerMedianTimesSacked,
    lppas.LagCareerPassAttPerTeamPlay,
    lppas.LagCareerPassCompPct,
    lppas.LagCareerPassCompPerTeamPlay,
    lppas.LagCareerPassIntPct,
    lppas.LagCareerPassIntPerTeamPlay,
    lppas.LagCareerPassTdPct,
    lppas.LagCareerPassTdPerTeamPlay,
    lppas.LagCareerPassYardsPerAtt,
    lppas.LagCareerPassYardsPerComp,
    lppas.LagCareerPassYardsPerTeamPlay,
    lppas.LagSeasonAdjPassYardsPerAtt,
    lppas.LagSeasonAvgLongestPassComp,
    lppas.LagSeasonAvgPassAtt,
    lppas.LagSeasonAvgPassComp,
    lppas.LagSeasonAvgPassInt,
    lppas.LagSeasonAvgPassTd,
    lppas.LagSeasonAvgPassYards,
    lppas.LagSeasonAvgQBR,
    lppas.LagSeasonAvgSackYardsLost,
    lppas.LagSeasonAvgTimesSacked,
    lppas.LagSeasonMedianLongestPassComp,
    lppas.LagSeasonMedianPassAtt,
    lppas.LagSeasonMedianPassComp,
    lppas.LagSeasonMedianPassInt,
    lppas.LagSeasonMedianPassTd,
    lppas.LagSeasonMedianPassYards,
    lppas.LagSeasonMedianQBR,
    lppas.LagSeasonMedianSackYardsLost,
    lppas.LagSeasonMedianTimesSacked,
    lppas.LagSeasonPassAttPerTeamPlay,
    lppas.LagSeasonPassCompPct,
    lppas.LagSeasonPassCompPerTeamPlay,
    lppas.LagSeasonPassIntPct,
    lppas.LagSeasonPassIntPerTeamPlay,
    lppas.LagSeasonPassTdPct,
    lppas.LagSeasonPassTdPerTeamPlay,
    lppas.LagSeasonPassYardsPerAtt,
    lppas.LagSeasonPassYardsPerComp,
    lppas.LagSeasonPassYardsPerTeamPlay,
    

    -- lag-team-passing
    ltpas.Lag3GameAvgTeamNetPassYards,
    ltpas.Lag3GameAvgTeamPassAtt,
    ltpas.Lag3GameAvgTeamPassComp,
    ltpas.Lag3GameAvgTeamPassInt,
    ltpas.Lag3GameAvgTeamPassTd,
    ltpas.Lag3GameAvgTeamPassYards,
    ltpas.Lag3GameAvgTeamSackYardsLost,
    ltpas.Lag3GameAvgTeamTimesSacked,
    ltpas.Lag3GameMedianTeamNetPassYards,
    ltpas.Lag3GameMedianTeamPassAtt,
    ltpas.Lag3GameMedianTeamPassComp,
    ltpas.Lag3GameMedianTeamPassInt,
    ltpas.Lag3GameMedianTeamPassTd,
    ltpas.Lag3GameMedianTeamPassYards,
    ltpas.Lag3GameMedianTeamSackYardsLost,
    ltpas.Lag3GameMedianTeamTimesSacked,
    ltpas.Lag3GameTeamAdjPassYardsPerAtt,
    ltpas.Lag3GameTeamPassCompPct,
    ltpas.Lag3GameTeamPassIntPct,
    ltpas.Lag3GameTeamPassTdPct,
    ltpas.Lag3GameTeamPassYardsPerAtt,
    ltpas.Lag3GameTeamPassYardsPerComp,
    ltpas.Lag6GameAvgTeamNetPassYards,
    ltpas.Lag6GameAvgTeamPassAtt,
    ltpas.Lag6GameAvgTeamPassComp,
    ltpas.Lag6GameAvgTeamPassInt,
    ltpas.Lag6GameAvgTeamPassTd,
    ltpas.Lag6GameAvgTeamPassYards,
    ltpas.Lag6GameAvgTeamSackYardsLost,
    ltpas.Lag6GameAvgTeamTimesSacked,
    ltpas.Lag6GameMedianTeamNetPassYards,
    ltpas.Lag6GameMedianTeamPassAtt,
    ltpas.Lag6GameMedianTeamPassComp,
    ltpas.Lag6GameMedianTeamPassInt,
    ltpas.Lag6GameMedianTeamPassTd,
    ltpas.Lag6GameMedianTeamPassYards,
    ltpas.Lag6GameMedianTeamSackYardsLost,
    ltpas.Lag6GameMedianTeamTimesSacked,
    ltpas.Lag6GameTeamAdjPassYardsPerAtt,
    ltpas.Lag6GameTeamPassCompPct,
    ltpas.Lag6GameTeamPassIntPct,
    ltpas.Lag6GameTeamPassTdPct,
    ltpas.Lag6GameTeamPassYardsPerAtt,
    ltpas.Lag6GameTeamPassYardsPerComp,
    ltpas.LagCareerAvgTeamNetPassYards,
    ltpas.LagCareerAvgTeamPassAtt,
    ltpas.LagCareerAvgTeamPassComp,
    ltpas.LagCareerAvgTeamPassInt,
    ltpas.LagCareerAvgTeamPassTd,
    ltpas.LagCareerAvgTeamPassYards,
    ltpas.LagCareerAvgTeamSackYardsLost,
    ltpas.LagCareerAvgTeamTimesSacked,
    ltpas.LagCareerMedianTeamNetPassYards,
    ltpas.LagCareerMedianTeamPassAtt,
    ltpas.LagCareerMedianTeamPassComp,
    ltpas.LagCareerMedianTeamPassInt,
    ltpas.LagCareerMedianTeamPassTd,
    ltpas.LagCareerMedianTeamPassYards,
    ltpas.LagCareerMedianTeamSackYardsLost,
    ltpas.LagCareerMedianTeamTimesSacked,
    ltpas.LagCareerTeamAdjPassYardsPerAtt,
    ltpas.LagCareerTeamPassCompPct,
    ltpas.LagCareerTeamPassIntPct,
    ltpas.LagCareerTeamPassTdPct,
    ltpas.LagCareerTeamPassYardsPerAtt,
    ltpas.LagCareerTeamPassYardsPerComp,
    ltpas.LagSeasonAvgTeamNetPassYards,
    ltpas.LagSeasonAvgTeamPassAtt,
    ltpas.LagSeasonAvgTeamPassComp,
    ltpas.LagSeasonAvgTeamPassInt,
    ltpas.LagSeasonAvgTeamPassTd,
    ltpas.LagSeasonAvgTeamPassYards,
    ltpas.LagSeasonAvgTeamSackYardsLost,
    ltpas.LagSeasonAvgTeamTimesSacked,
    ltpas.LagSeasonMedianTeamNetPassYards,
    ltpas.LagSeasonMedianTeamPassAtt,
    ltpas.LagSeasonMedianTeamPassComp,
    ltpas.LagSeasonMedianTeamPassInt,
    ltpas.LagSeasonMedianTeamPassTd,
    ltpas.LagSeasonMedianTeamPassYards,
    ltpas.LagSeasonMedianTeamSackYardsLost,
    ltpas.LagSeasonMedianTeamTimesSacked,
    ltpas.LagSeasonTeamAdjPassYardsPerAtt,
    ltpas.LagSeasonTeamPassCompPct,
    ltpas.LagSeasonTeamPassIntPct,
    ltpas.LagSeasonTeamPassTdPct,
    ltpas.LagSeasonTeamPassYardsPerAtt,
    ltpas.LagSeasonTeamPassYardsPerComp,

    -- lag-team-rushing
    ltrus.Lag3GameAvgTeamRushAtt,
    ltrus.Lag3GameAvgTeamRushTd,
    ltrus.Lag3GameAvgTeamRushYards,
    ltrus.Lag3GameMedianTeamRushAtt,
    ltrus.Lag3GameMedianTeamRushTd,
    ltrus.Lag3GameMedianTeamRushYards,
    ltrus.Lag3GameTeamRushTdPct,
    ltrus.Lag3GameTeamYardsPerRush,
    ltrus.Lag6GameAvgTeamRushAtt,
    ltrus.Lag6GameAvgTeamRushTd,
    ltrus.Lag6GameAvgTeamRushYards,
    ltrus.Lag6GameMedianTeamRushAtt,
    ltrus.Lag6GameMedianTeamRushTd,
    ltrus.Lag6GameMedianTeamRushYards,
    ltrus.Lag6GameTeamRushTdPct,
    ltrus.Lag6GameTeamYardsPerRush,
    ltrus.LagCareerAvgTeamRushAtt,
    ltrus.LagCareerAvgTeamRushTd,
    ltrus.LagCareerAvgTeamRushYards,
    ltrus.LagCareerMedianTeamRushAtt,
    ltrus.LagCareerMedianTeamRushTd,
    ltrus.LagCareerMedianTeamRushYards,
    ltrus.LagCareerTeamRushTdPct,
    ltrus.LagCareerTeamYardsPerRush,
    ltrus.LagSeasonAvgTeamRushAtt,
    ltrus.LagSeasonAvgTeamRushTd,
    ltrus.LagSeasonAvgTeamRushYards,
    ltrus.LagSeasonMedianTeamRushAtt,
    ltrus.LagSeasonMedianTeamRushTd,
    ltrus.LagSeasonMedianTeamRushYards,
    ltrus.LagSeasonTeamRushTdPct,
    ltrus.LagSeasonTeamYardsPerRush,
    
    -- lag-team-receiving
    ltrec.Lag3GameAvgTeamLongestRec,
    ltrec.Lag3GameAvgTeamRecTargets,
    ltrec.Lag3GameAvgTeamRecTd,
    ltrec.Lag3GameAvgTeamRecYards,
    ltrec.Lag3GameAvgTeamReceptions,
    ltrec.Lag3GameMedianTeamLongestRec,
    ltrec.Lag3GameMedianTeamRecTargets,
    ltrec.Lag3GameMedianTeamRecTd,
    ltrec.Lag3GameMedianTeamRecYards,
    ltrec.Lag3GameMedianTeamReceptions,
    ltrec.Lag3GameTeamCatchPct,
    ltrec.Lag3GameTeamYardsPerRec,
    ltrec.Lag3GameTeamYardsPerRecTarget,
    ltrec.Lag6GameAvgTeamLongestRec,
    ltrec.Lag6GameAvgTeamRecTargets,
    ltrec.Lag6GameAvgTeamRecTd,
    ltrec.Lag6GameAvgTeamRecYards,
    ltrec.Lag6GameAvgTeamReceptions,
    ltrec.Lag6GameMedianTeamLongestRec,
    ltrec.Lag6GameMedianTeamRecTargets,
    ltrec.Lag6GameMedianTeamRecTd,
    ltrec.Lag6GameMedianTeamRecYards,
    ltrec.Lag6GameMedianTeamReceptions,
    ltrec.Lag6GameTeamCatchPct,
    ltrec.Lag6GameTeamYardsPerRec,
    ltrec.Lag6GameTeamYardsPerRecTarget,
    ltrec.LagCareerAvgTeamLongestRec,
    ltrec.LagCareerAvgTeamRecTargets,
    ltrec.LagCareerAvgTeamRecTd,
    ltrec.LagCareerAvgTeamRecYards,
    ltrec.LagCareerAvgTeamReceptions,
    ltrec.LagCareerMedianTeamLongestRec,
    ltrec.LagCareerMedianTeamRecTargets,
    ltrec.LagCareerMedianTeamRecTd,
    ltrec.LagCareerMedianTeamRecYards,
    ltrec.LagCareerMedianTeamReceptions,
    ltrec.LagCareerTeamCatchPct,
    ltrec.LagCareerTeamYardsPerRec,
    ltrec.LagCareerTeamYardsPerRecTarget,
    ltrec.LagSeasonAvgTeamLongestRec,
    ltrec.LagSeasonAvgTeamRecTargets,
    ltrec.LagSeasonAvgTeamRecTd,
    ltrec.LagSeasonAvgTeamRecYards,
    ltrec.LagSeasonAvgTeamReceptions,
    ltrec.LagSeasonMedianTeamLongestRec,
    ltrec.LagSeasonMedianTeamRecTargets,
    ltrec.LagSeasonMedianTeamRecTd,
    ltrec.LagSeasonMedianTeamRecYards,
    ltrec.LagSeasonMedianTeamReceptions,
    ltrec.LagSeasonTeamCatchPct,
    ltrec.LagSeasonTeamYardsPerRec,
    ltrec.LagSeasonTeamYardsPerRecTarget,
    
    -- lag-team-defense
    ltdef.Lag3GameAvgTeamDefFumbleRec,
    ltdef.Lag3GameAvgTeamDefFumbleRecTd,
    ltdef.Lag3GameAvgTeamDefInt,
    ltdef.Lag3GameAvgTeamDefIntTd,
    ltdef.Lag3GameAvgTeamDefPointsAllowed,
    ltdef.Lag3GameAvgTeamDefSacks,
    ltdef.Lag3GameAvgTeamDefSafeties,
    ltdef.Lag3GameAvgTeamNetPassYardsAllowed,
    ltdef.Lag3GameAvgTeamPassAttAllowed,
    ltdef.Lag3GameAvgTeamPassCompAllowed,
    ltdef.Lag3GameAvgTeamPassIntAllowed,
    ltdef.Lag3GameAvgTeamPassTdAllowed,
    ltdef.Lag3GameAvgTeamPassYardsAllowed,
    ltdef.Lag3GameAvgTeamRushAttAllowed,
    ltdef.Lag3GameAvgTeamRushTdAllowed,
    ltdef.Lag3GameAvgTeamRushYardsAllowed,
    ltdef.Lag3GameAvgTeamSackYardsLostAllowed,
    ltdef.Lag3GameAvgTeamTimesSackedAllowed,
    ltdef.Lag3GameAvgTeamToPAllowed,
    ltdef.Lag3GameAvgTeamTotalPlaysAllowed,
    ltdef.Lag3GameMedianTeamDefFumbleRec,
    ltdef.Lag3GameMedianTeamDefFumbleRecTd,
    ltdef.Lag3GameMedianTeamDefInt,
    ltdef.Lag3GameMedianTeamDefIntTd,
    ltdef.Lag3GameMedianTeamDefPointsAllowed,
    ltdef.Lag3GameMedianTeamDefSacks,
    ltdef.Lag3GameMedianTeamDefSafeties,
    ltdef.Lag3GameMedianTeamNetPassYardsAllowed,
    ltdef.Lag3GameMedianTeamPassAttAllowed,
    ltdef.Lag3GameMedianTeamPassCompAllowed,
    ltdef.Lag3GameMedianTeamPassIntAllowed,
    ltdef.Lag3GameMedianTeamPassTdAllowed,
    ltdef.Lag3GameMedianTeamPassYardsAllowed,
    ltdef.Lag3GameMedianTeamRushAttAllowed,
    ltdef.Lag3GameMedianTeamRushTdAllowed,
    ltdef.Lag3GameMedianTeamRushYardsAllowed,
    ltdef.Lag3GameMedianTeamSackYardsLostAllowed,
    ltdef.Lag3GameMedianTeamTimesSackedAllowed,
    ltdef.Lag3GameMedianTeamToPAllowed,
    ltdef.Lag3GameMedianTeamTotalPlaysAllowed,
    ltdef.Lag3GameTeamAdjPassYardsPerAttAllowed,
    ltdef.Lag3GameTeamPassCompPctAllowed,
    ltdef.Lag3GameTeamPassIntPctAllowed,
    ltdef.Lag3GameTeamPassTdPctAllowed,
    ltdef.Lag3GameTeamPassYardsPerAttAllowed,
    ltdef.Lag3GameTeamPassYardsPerCompAllowed,
    ltdef.Lag3GameTeamRushTdPctAllowed,
    ltdef.Lag3GameTeamYardsPerRushAllowed,
    ltdef.Lag6GameAvgTeamDefFumbleRec,
    ltdef.Lag6GameAvgTeamDefFumbleRecTd,
    ltdef.Lag6GameAvgTeamDefInt,
    ltdef.Lag6GameAvgTeamDefIntTd,
    ltdef.Lag6GameAvgTeamDefPointsAllowed,
    ltdef.Lag6GameAvgTeamDefSacks,
    ltdef.Lag6GameAvgTeamDefSafeties,
    ltdef.Lag6GameAvgTeamNetPassYardsAllowed,
    ltdef.Lag6GameAvgTeamPassAttAllowed,
    ltdef.Lag6GameAvgTeamPassCompAllowed,
    ltdef.Lag6GameAvgTeamPassIntAllowed,
    ltdef.Lag6GameAvgTeamPassTdAllowed,
    ltdef.Lag6GameAvgTeamPassYardsAllowed,
    ltdef.Lag6GameAvgTeamRushAttAllowed,
    ltdef.Lag6GameAvgTeamRushTdAllowed,
    ltdef.Lag6GameAvgTeamRushYardsAllowed,
    ltdef.Lag6GameAvgTeamSackYardsLostAllowed,
    ltdef.Lag6GameAvgTeamTimesSackedAllowed,
    ltdef.Lag6GameAvgTeamToPAllowed,
    ltdef.Lag6GameAvgTeamTotalPlaysAllowed,
    ltdef.Lag6GameMedianTeamDefFumbleRec,
    ltdef.Lag6GameMedianTeamDefFumbleRecTd,
    ltdef.Lag6GameMedianTeamDefInt,
    ltdef.Lag6GameMedianTeamDefIntTd,
    ltdef.Lag6GameMedianTeamDefPointsAllowed,
    ltdef.Lag6GameMedianTeamDefSacks,
    ltdef.Lag6GameMedianTeamDefSafeties,
    ltdef.Lag6GameMedianTeamNetPassYardsAllowed,
    ltdef.Lag6GameMedianTeamPassAttAllowed,
    ltdef.Lag6GameMedianTeamPassCompAllowed,
    ltdef.Lag6GameMedianTeamPassIntAllowed,
    ltdef.Lag6GameMedianTeamPassTdAllowed,
    ltdef.Lag6GameMedianTeamPassYardsAllowed,
    ltdef.Lag6GameMedianTeamRushAttAllowed,
    ltdef.Lag6GameMedianTeamRushTdAllowed,
    ltdef.Lag6GameMedianTeamRushYardsAllowed,
    ltdef.Lag6GameMedianTeamSackYardsLostAllowed,
    ltdef.Lag6GameMedianTeamTimesSackedAllowed,
    ltdef.Lag6GameMedianTeamToPAllowed,
    ltdef.Lag6GameMedianTeamTotalPlaysAllowed,
    ltdef.Lag6GameTeamAdjPassYardsPerAttAllowed,
    ltdef.Lag6GameTeamPassCompPctAllowed,
    ltdef.Lag6GameTeamPassIntPctAllowed,
    ltdef.Lag6GameTeamPassTdPctAllowed,
    ltdef.Lag6GameTeamPassYardsPerAttAllowed,
    ltdef.Lag6GameTeamPassYardsPerCompAllowed,
    ltdef.Lag6GameTeamRushTdPctAllowed,
    ltdef.Lag6GameTeamYardsPerRushAllowed,
    ltdef.LagCareerAvgTeamDefFumbleRec,
    ltdef.LagCareerAvgTeamDefFumbleRecTd,
    ltdef.LagCareerAvgTeamDefInt,
    ltdef.LagCareerAvgTeamDefIntTd,
    ltdef.LagCareerAvgTeamDefPointsAllowed,
    ltdef.LagCareerAvgTeamDefSacks,
    ltdef.LagCareerAvgTeamDefSafeties,
    ltdef.LagCareerAvgTeamNetPassYardsAllowed,
    ltdef.LagCareerAvgTeamPassAttAllowed,
    ltdef.LagCareerAvgTeamPassCompAllowed,
    ltdef.LagCareerAvgTeamPassIntAllowed,
    ltdef.LagCareerAvgTeamPassTdAllowed,
    ltdef.LagCareerAvgTeamPassYardsAllowed,
    ltdef.LagCareerAvgTeamRushAttAllowed,
    ltdef.LagCareerAvgTeamRushTdAllowed,
    ltdef.LagCareerAvgTeamRushYardsAllowed,
    ltdef.LagCareerAvgTeamSackYardsLostAllowed,
    ltdef.LagCareerAvgTeamTimesSackedAllowed,
    ltdef.LagCareerAvgTeamToPAllowed,
    ltdef.LagCareerAvgTeamTotalPlaysAllowed,
    ltdef.LagCareerMedianTeamDefFumbleRec,
    ltdef.LagCareerMedianTeamDefFumbleRecTd,
    ltdef.LagCareerMedianTeamDefInt,
    ltdef.LagCareerMedianTeamDefIntTd,
    ltdef.LagCareerMedianTeamDefPointsAllowed,
    ltdef.LagCareerMedianTeamDefSacks,
    ltdef.LagCareerMedianTeamDefSafeties,
    ltdef.LagCareerMedianTeamNetPassYardsAllowed,
    ltdef.LagCareerMedianTeamPassAttAllowed,
    ltdef.LagCareerMedianTeamPassCompAllowed,
    ltdef.LagCareerMedianTeamPassIntAllowed,
    ltdef.LagCareerMedianTeamPassTdAllowed,
    ltdef.LagCareerMedianTeamPassYardsAllowed,
    ltdef.LagCareerMedianTeamRushAttAllowed,
    ltdef.LagCareerMedianTeamRushTdAllowed,
    ltdef.LagCareerMedianTeamRushYardsAllowed,
    ltdef.LagCareerMedianTeamSackYardsLostAllowed,
    ltdef.LagCareerMedianTeamTimesSackedAllowed,
    ltdef.LagCareerMedianTeamToPAllowed,
    ltdef.LagCareerMedianTeamTotalPlaysAllowed,
    ltdef.LagCareerTeamAdjPassYardsPerAttAllowed,
    ltdef.LagCareerTeamPassCompPctAllowed,
    ltdef.LagCareerTeamPassIntPctAllowed,
    ltdef.LagCareerTeamPassTdPctAllowed,
    ltdef.LagCareerTeamPassYardsPerAttAllowed,
    ltdef.LagCareerTeamPassYardsPerCompAllowed,
    ltdef.LagCareerTeamRushTdPctAllowed,
    ltdef.LagCareerTeamYardsPerRushAllowed,
    ltdef.LagSeasonAvgTeamDefFumbleRec,
    ltdef.LagSeasonAvgTeamDefFumbleRecTd,
    ltdef.LagSeasonAvgTeamDefInt,
    ltdef.LagSeasonAvgTeamDefIntTd,
    ltdef.LagSeasonAvgTeamDefPointsAllowed,
    ltdef.LagSeasonAvgTeamDefSacks,
    ltdef.LagSeasonAvgTeamDefSafeties,
    ltdef.LagSeasonAvgTeamNetPassYardsAllowed,
    ltdef.LagSeasonAvgTeamPassAttAllowed,
    ltdef.LagSeasonAvgTeamPassCompAllowed,
    ltdef.LagSeasonAvgTeamPassIntAllowed,
    ltdef.LagSeasonAvgTeamPassTdAllowed,
    ltdef.LagSeasonAvgTeamPassYardsAllowed,
    ltdef.LagSeasonAvgTeamRushAttAllowed,
    ltdef.LagSeasonAvgTeamRushTdAllowed,
    ltdef.LagSeasonAvgTeamRushYardsAllowed,
    ltdef.LagSeasonAvgTeamSackYardsLostAllowed,
    ltdef.LagSeasonAvgTeamTimesSackedAllowed,
    ltdef.LagSeasonAvgTeamToPAllowed,
    ltdef.LagSeasonAvgTeamTotalPlaysAllowed,
    ltdef.LagSeasonMedianTeamDefFumbleRec,
    ltdef.LagSeasonMedianTeamDefFumbleRecTd,
    ltdef.LagSeasonMedianTeamDefInt,
    ltdef.LagSeasonMedianTeamDefIntTd,
    ltdef.LagSeasonMedianTeamDefPointsAllowed,
    ltdef.LagSeasonMedianTeamDefSacks,
    ltdef.LagSeasonMedianTeamDefSafeties,
    ltdef.LagSeasonMedianTeamNetPassYardsAllowed,
    ltdef.LagSeasonMedianTeamPassAttAllowed,
    ltdef.LagSeasonMedianTeamPassCompAllowed,
    ltdef.LagSeasonMedianTeamPassIntAllowed,
    ltdef.LagSeasonMedianTeamPassTdAllowed,
    ltdef.LagSeasonMedianTeamPassYardsAllowed,
    ltdef.LagSeasonMedianTeamRushAttAllowed,
    ltdef.LagSeasonMedianTeamRushTdAllowed,
    ltdef.LagSeasonMedianTeamRushYardsAllowed,
    ltdef.LagSeasonMedianTeamSackYardsLostAllowed,
    ltdef.LagSeasonMedianTeamTimesSackedAllowed,
    ltdef.LagSeasonMedianTeamToPAllowed,
    ltdef.LagSeasonMedianTeamTotalPlaysAllowed,
    ltdef.LagSeasonTeamAdjPassYardsPerAttAllowed,
    ltdef.LagSeasonTeamPassCompPctAllowed,
    ltdef.LagSeasonTeamPassIntPctAllowed,
    ltdef.LagSeasonTeamPassTdPctAllowed,
    ltdef.LagSeasonTeamPassYardsPerAttAllowed,
    ltdef.LagSeasonTeamPassYardsPerCompAllowed,
    ltdef.LagSeasonTeamRushTdPctAllowed,
    ltdef.LagSeasonTeamYardsPerRushAllowed,
    
    -- lag-team-plays
    ltpla.Lag3GameAvgTeam3rdDownAtt,
    ltpla.Lag3GameAvgTeam3rdDownMade,
    ltpla.Lag3GameAvgTeam4thDownAtt,
    ltpla.Lag3GameAvgTeam4thDownMade,
    ltpla.Lag3GameAvgTeamToP,
    ltpla.Lag3GameAvgTeamTotalPlays,
    ltpla.Lag3GameMedianTeam3rdDownAtt,
    ltpla.Lag3GameMedianTeam3rdDownMade,
    ltpla.Lag3GameMedianTeam4thDownAtt,
    ltpla.Lag3GameMedianTeam4thDownMade,
    ltpla.Lag3GameMedianTeamToP,
    ltpla.Lag3GameMedianTeamTotalPlays,
    ltpla.Lag3GameTeam3rdDownPct,
    ltpla.Lag3GameTeam4thDownPct,
    ltpla.Lag3GameTeamPassPlayPct,
    ltpla.Lag3GameTeamRushPlayPct,
    ltpla.Lag6GameAvgTeam3rdDownAtt,
    ltpla.Lag6GameAvgTeam3rdDownMade,
    ltpla.Lag6GameAvgTeam4thDownAtt,
    ltpla.Lag6GameAvgTeam4thDownMade,
    ltpla.Lag6GameAvgTeamToP,
    ltpla.Lag6GameAvgTeamTotalPlays,
    ltpla.Lag6GameMedianTeam3rdDownAtt,
    ltpla.Lag6GameMedianTeam3rdDownMade,
    ltpla.Lag6GameMedianTeam4thDownAtt,
    ltpla.Lag6GameMedianTeam4thDownMade,
    ltpla.Lag6GameMedianTeamToP,
    ltpla.Lag6GameMedianTeamTotalPlays,
    ltpla.Lag6GameTeam3rdDownPct,
    ltpla.Lag6GameTeam4thDownPct,
    ltpla.Lag6GameTeamPassPlayPct,
    ltpla.Lag6GameTeamRushPlayPct,
    ltpla.LagCareerAvgTeam3rdDownAtt,
    ltpla.LagCareerAvgTeam3rdDownMade,
    ltpla.LagCareerAvgTeam4thDownAtt,
    ltpla.LagCareerAvgTeam4thDownMade,
    ltpla.LagCareerAvgTeamToP,
    ltpla.LagCareerAvgTeamTotalPlays,
    ltpla.LagCareerMedianTeam3rdDownAtt,
    ltpla.LagCareerMedianTeam3rdDownMade,
    ltpla.LagCareerMedianTeam4thDownAtt,
    ltpla.LagCareerMedianTeam4thDownMade,
    ltpla.LagCareerMedianTeamToP,
    ltpla.LagCareerMedianTeamTotalPlays,
    ltpla.LagCareerTeam3rdDownPct,
    ltpla.LagCareerTeam4thDownPct,
    ltpla.LagCareerTeamPassPlayPct,
    ltpla.LagCareerTeamRushPlayPct,
    ltpla.LagSeasonAvgTeam3rdDownAtt,
    ltpla.LagSeasonAvgTeam3rdDownMade,
    ltpla.LagSeasonAvgTeam4thDownAtt,
    ltpla.LagSeasonAvgTeam4thDownMade,
    ltpla.LagSeasonAvgTeamToP,
    ltpla.LagSeasonAvgTeamTotalPlays,
    ltpla.LagSeasonMedianTeam3rdDownAtt,
    ltpla.LagSeasonMedianTeam3rdDownMade,
    ltpla.LagSeasonMedianTeam4thDownAtt,
    ltpla.LagSeasonMedianTeam4thDownMade,
    ltpla.LagSeasonMedianTeamToP,
    ltpla.LagSeasonMedianTeamTotalPlays,
    ltpla.LagSeasonTeam3rdDownPct,
    ltpla.LagSeasonTeam4thDownPct,
    ltpla.LagSeasonTeamPassPlayPct,
    ltpla.LagSeasonTeamRushPlayPct,

    -- lag-league-passing
    llpas.Lag3GameAvgLeagueNetPassYards,
    llpas.Lag3GameAvgLeaguePassAtt,
    llpas.Lag3GameAvgLeaguePassComp,
    llpas.Lag3GameAvgLeaguePassInt,
    llpas.Lag3GameAvgLeaguePassTd,
    llpas.Lag3GameAvgLeaguePassYards,
    llpas.Lag3GameAvgLeagueSackYardsLost,
    llpas.Lag3GameAvgLeagueTimesSacked,
    llpas.Lag3GameLeagueAdjPassYardsPerAtt,
    llpas.Lag3GameLeaguePassCompPct,
    llpas.Lag3GameLeaguePassIntPct,
    llpas.Lag3GameLeaguePassTdPct,
    llpas.Lag3GameLeaguePassYardsPerAtt,
    llpas.Lag3GameLeaguePassYardsPerComp,
    llpas.Lag6GameAvgLeagueNetPassYards,
    llpas.Lag6GameAvgLeaguePassAtt,
    llpas.Lag6GameAvgLeaguePassComp,
    llpas.Lag6GameAvgLeaguePassInt,
    llpas.Lag6GameAvgLeaguePassTd,
    llpas.Lag6GameAvgLeaguePassYards,
    llpas.Lag6GameAvgLeagueSackYardsLost,
    llpas.Lag6GameAvgLeagueTimesSacked,
    llpas.Lag6GameLeagueAdjPassYardsPerAtt,
    llpas.Lag6GameLeaguePassCompPct,
    llpas.Lag6GameLeaguePassIntPct,
    llpas.Lag6GameLeaguePassTdPct,
    llpas.Lag6GameLeaguePassYardsPerAtt,
    llpas.Lag6GameLeaguePassYardsPerComp,
    llpas.LagCareerAvgLeagueNetPassYards,
    llpas.LagCareerAvgLeaguePassAtt,
    llpas.LagCareerAvgLeaguePassComp,
    llpas.LagCareerAvgLeaguePassInt,
    llpas.LagCareerAvgLeaguePassTd,
    llpas.LagCareerAvgLeaguePassYards,
    llpas.LagCareerAvgLeagueSackYardsLost,
    llpas.LagCareerAvgLeagueTimesSacked,
    llpas.LagCareerLeagueAdjPassYardsPerAtt,
    llpas.LagCareerLeaguePassCompPct,
    llpas.LagCareerLeaguePassIntPct,
    llpas.LagCareerLeaguePassTdPct,
    llpas.LagCareerLeaguePassYardsPerAtt,
    llpas.LagCareerLeaguePassYardsPerComp,
    llpas.LagSeasonAvgLeagueNetPassYards,
    llpas.LagSeasonAvgLeaguePassAtt,
    llpas.LagSeasonAvgLeaguePassComp,
    llpas.LagSeasonAvgLeaguePassInt,
    llpas.LagSeasonAvgLeaguePassTd,
    llpas.LagSeasonAvgLeaguePassYards,
    llpas.LagSeasonAvgLeagueSackYardsLost,
    llpas.LagSeasonAvgLeagueTimesSacked,
    llpas.LagSeasonLeagueAdjPassYardsPerAtt,
    llpas.LagSeasonLeaguePassCompPct,
    llpas.LagSeasonLeaguePassIntPct,
    llpas.LagSeasonLeaguePassTdPct,
    llpas.LagSeasonLeaguePassYardsPerAtt,
    llpas.LagSeasonLeaguePassYardsPerComp,    
 

    -- game-info
    gi.Season,
    gi.ClosingSpread AS PointSpread,  -- Add ClosingSpread from game_info table
    gi.ClosingTotal AS PointTotal,   -- Add ClosingTotal from game_info table
    gi.ClosingMoneyline AS Moneyline,  -- Add ClosingMoneyline from game_info table
    gi.HomeFlag,  -- Add HomeFlag from game_info table
    gi.OppAbbr
    
FROM lag_player_passing AS lppas
    
JOIN player_positions AS ppos
    ON lppas.PlayerId = ppos.PlayerId
    AND lppas.Player = ppos.Player
    AND lppas.GameDate = ppos.GameDate
    AND lppas.TeamAbbr = ppos.TeamAbbr

JOIN player_passing AS ppas
    ON lppas.PlayerId = ppas.PlayerId
    AND lppas.Player = ppas.Player
    AND lppas.GameDate = ppas.GameDate
    AND lppas.TeamAbbr = ppas.TeamAbbr

JOIN lag_team_passing AS ltpas
    ON lppas.GameId = ltpas.GameId
    AND lppas.GameId2 = ltpas.GameId2
    AND lppas.GameDate = ltpas.GameDate
    AND lppas.TeamAbbr = ltpas.TeamAbbr
    AND lppas.Season = ltpas.Season

JOIN lag_team_rushing AS ltrus
    ON lppas.GameId = ltrus.GameId
    AND lppas.GameId2 = ltrus.GameId2
    AND lppas.GameDate = ltrus.GameDate
    AND lppas.TeamAbbr = ltrus.TeamAbbr
    AND lppas.Season = ltrus.Season    
    
JOIN lag_team_receiving AS ltrec
    ON lppas.GameId = ltrec.GameId
    AND lppas.GameId2 = ltrec.GameId2
    AND lppas.GameDate = ltrec.GameDate
    AND lppas.TeamAbbr = ltrec.TeamAbbr
    AND lppas.Season = ltrec.Season

JOIN lag_team_defense AS ltdef
    ON lppas.GameDate = ltdef.GameDate
    AND lppas.GameId = ltdef.GameId
    AND lppas.GameId2 = ltdef.GameId2
    AND lppas.Season = ltdef.Season
    AND gi.OppAbbr = ltdef.TeamAbbr

JOIN lag_team_plays as ltpla
    ON lppas.GameDate = ltpla.GameDate
    AND lppas.GameId = ltpla.GameId
    AND lppas.GameId2 = ltpla.GameId2
    AND lppas.Season = ltpla.Season
    AND gi.TeamAbbr = ltpla.TeamAbbr    
    
JOIN lag_league_passing AS llpas
    ON lppas.GameDate = llpas.GameDate
    AND lppas.Season = llpas.Season 
    
JOIN game_info AS gi
    ON lppas.GameDate = gi.GameDate
    AND lppas.GameId = gi.GameId
    AND lppas.GameId2 = gi.GameId2
    AND lppas.TeamAbbr = gi.TeamAbbr    
    
-- WHERE ppos.Position = "QB" AND ppas.PassComp > 4 AND ppos.StarterFlag = "Y" 
WHERE ppos.Position = "QB"
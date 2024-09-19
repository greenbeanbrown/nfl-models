-- MODEL TRAINING
SELECT 
    -- Target Variable
    prec.Receptions AS TargetReceptions,
    
    -- player-positions
    ppos.PlayerId,
    ppos.Player,
    ppos.GameDate,
    ppos.Position, 
    ppos.PositionId, 
    
    -- lag-player-receiving
    lprec.Lag3GameAvgLongestRec,
    lprec.Lag3GameAvgRecTargets,
    lprec.Lag3GameAvgRecTd,
    lprec.Lag3GameAvgRecYards,
    lprec.Lag3GameAvgReceptions,
    lprec.Lag3GameCatchPct,
    lprec.Lag3GameMedianLongestRec,
    lprec.Lag3GameMedianRecTargets,
    lprec.Lag3GameMedianRecTd,
    lprec.Lag3GameMedianRecYards,
    lprec.Lag3GameMedianReceptions,
    lprec.Lag3GameRecPerTeamPlay,
    lprec.Lag3GameRecShare,
    lprec.Lag3GameRecTargetShare,
    lprec.Lag3GameRecTargetsPerTeamPlay,
    lprec.Lag3GameRecYardsPerTeamPlay,
    lprec.Lag3GameRecYardsShare,
    lprec.Lag3GameYardsPerRecTarget,
    lprec.Lag3GameYardsPerRush,
    lprec.Lag6GameAvgLongestRec,
    lprec.Lag6GameAvgRecTargets,
    lprec.Lag6GameAvgRecTd,
    lprec.Lag6GameAvgRecYards,
    lprec.Lag6GameAvgReceptions,
    lprec.Lag6GameCatchPct,
    lprec.Lag6GameMedianLongestRec,
    lprec.Lag6GameMedianRecTargets,
    lprec.Lag6GameMedianRecTd,
    lprec.Lag6GameMedianRecYards,
    lprec.Lag6GameMedianReceptions,
    lprec.Lag6GameRecPerTeamPlay,
    lprec.Lag6GameRecShare,
    lprec.Lag6GameRecTargetShare,
    lprec.Lag6GameRecTargetsPerTeamPlay,
    lprec.Lag6GameRecYardsPerTeamPlay,
    lprec.Lag6GameRecYardsShare,
    lprec.Lag6GameYardsPerRecTarget,
    lprec.Lag6GameYardsPerRush,
    lprec.LagCareerAvgLongestRec,
    lprec.LagCareerAvgRecTargets,
    lprec.LagCareerAvgRecTd,
    lprec.LagCareerAvgRecYards,
    lprec.LagCareerAvgReceptions,
    lprec.LagCareerCatchPct,
    lprec.LagCareerMedianLongestRec,
    lprec.LagCareerMedianRecTargets,
    lprec.LagCareerMedianRecTd,
    lprec.LagCareerMedianRecYards,
    lprec.LagCareerMedianReceptions,
    lprec.LagCareerPerRecTarget,
    lprec.LagCareerRecPerTeamPlay,
    lprec.LagCareerRecShare,
    lprec.LagCareerRecTargetShare,
    lprec.LagCareerRecTargetsPerTeamPlay,
    lprec.LagCareerRecYardsPerTeamPlay,
    lprec.LagCareerRecYardsShare,
    lprec.LagCareerYardsPerRec,
    lprec.LagSeasonAvgLongestRec,
    lprec.LagSeasonAvgRecTargets,
    lprec.LagSeasonAvgRecTd,
    lprec.LagSeasonAvgRecYards,
    lprec.LagSeasonAvgReceptions,
    lprec.LagSeasonCatchPct,
    lprec.LagSeasonMedianLongestRec,
    lprec.LagSeasonMedianRecTargets,
    lprec.LagSeasonMedianRecTd,
    lprec.LagSeasonMedianRecYards,
    lprec.LagSeasonMedianReceptions,
    lprec.LagSeasonPerRecTarget,
    lprec.LagSeasonRecPerTeamPlay,
    lprec.LagSeasonRecShare,
    lprec.LagSeasonRecTargetShare,
    lprec.LagSeasonRecTargetsPerTeamPlay,
    lprec.LagSeasonRecYardsPerTeamPlay,
    lprec.LagSeasonRecYardsShare,
    lprec.LagSeasonYardsPerRec,
    
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
    
    -- lag-league-receiving
    llrec.Lag3GameAvgLeagueLongestRec,
    llrec.Lag3GameAvgLeagueRecTargets,
    llrec.Lag3GameAvgLeagueRecTd,
    llrec.Lag3GameAvgLeagueRecYards,
    llrec.Lag3GameAvgLeagueReceptions,
    llrec.Lag3GameLeagueCatchPct,
    llrec.Lag3GameLeagueYardsPerRec,
    llrec.Lag3GameLeagueYardsPerRecTarget,
    llrec.Lag6GameAvgLeagueLongestRec,
    llrec.Lag6GameAvgLeagueRecTargets,
    llrec.Lag6GameAvgLeagueRecTd,
    llrec.Lag6GameAvgLeagueRecYards,
    llrec.Lag6GameAvgLeagueReceptions,
    llrec.Lag6GameLeagueCatchPct,
    llrec.Lag6GameLeagueYardsPerRec,
    llrec.Lag6GameLeagueYardsPerRecTarget,
    llrec.LagCareerAvgLeagueLongestRec,
    llrec.LagCareerAvgLeagueRecTargets,
    llrec.LagCareerAvgLeagueRecTd,
    llrec.LagCareerAvgLeagueRecYards,
    llrec.LagCareerAvgLeagueReceptions,
    llrec.LagCareerLeagueCatchPct,
    llrec.LagCareerLeagueYardsPerRec,
    llrec.LagCareerLeagueYardsPerRecTarget,
    llrec.LagSeasonAvgLeagueLongestRec,
    llrec.LagSeasonAvgLeagueRecTargets,
    llrec.LagSeasonAvgLeagueRecTd,
    llrec.LagSeasonAvgLeagueRecYards,
    llrec.LagSeasonAvgLeagueReceptions,
    llrec.LagSeasonLeagueCatchPct,
    llrec.LagSeasonLeagueYardsPerRec,
    llrec.LagSeasonLeagueYardsPerRecTarget,  

    -- game-info
    gi.Season,
    gi.ClosingSpread AS PointSpread,  -- Add ClosingSpread from game_info table
    gi.ClosingTotal AS PointTotal,   -- Add ClosingTotal from game_info table
    gi.ClosingMoneyline AS Moneyline,  -- Add ClosingMoneyline from game_info table
    gi.HomeFlag,  -- Add HomeFlag from game_info table
    gi.OppAbbr
    
FROM lag_player_receiving AS lprec
    
JOIN player_positions AS ppos
    ON lprec.PlayerId = ppos.PlayerId
    AND lprec.Player = ppos.Player
    AND lprec.GameDate = ppos.GameDate
    AND lprec.TeamAbbr = ppos.TeamAbbr

-- Target variable table
JOIN player_receiving AS prec
    ON lprec.PlayerId = prec.PlayerId
    AND lprec.Player = prec.Player
    AND lprec.GameDate = prec.GameDate
    AND lprec.TeamAbbr = prec.TeamAbbr

JOIN lag_team_passing AS ltpas
    ON lprec.GameId = ltpas.GameId
    AND lprec.GameId2 = ltpas.GameId2
    AND lprec.GameDate = ltpas.GameDate
    AND lprec.TeamAbbr = ltpas.TeamAbbr
    AND lprec.Season = ltpas.Season

JOIN lag_team_rushing AS ltrus
    ON lprec.GameId = ltrus.GameId
    AND lprec.GameId2 = ltrus.GameId2
    AND lprec.GameDate = ltrus.GameDate
    AND lprec.TeamAbbr = ltrus.TeamAbbr
    AND lprec.Season = ltrus.Season    
    
JOIN lag_team_receiving AS ltrec
    ON lprec.GameId = ltrec.GameId
    AND lprec.GameId2 = ltrec.GameId2
    AND lprec.GameDate = ltrec.GameDate
    AND lprec.TeamAbbr = ltrec.TeamAbbr
    AND lprec.Season = ltrec.Season

JOIN lag_team_defense AS ltdef
    ON lprec.GameDate = ltdef.GameDate
    AND lprec.GameId = ltdef.GameId
    AND lprec.GameId2 = ltdef.GameId2
    AND lprec.Season = ltdef.Season
    AND gi.OppAbbr = ltdef.TeamAbbr

JOIN lag_team_plays as ltpla
    ON lprec.GameDate = ltpla.GameDate
    AND lprec.GameId = ltpla.GameId
    AND lprec.GameId2 = ltpla.GameId2
    AND lprec.Season = ltpla.Season
    AND gi.TeamAbbr = ltpla.TeamAbbr    

JOIN lag_league_receiving AS llrec
    ON lprec.GameDate = llrec.GameDate
    AND lprec.Season = llrec.Season     
    
JOIN game_info AS gi
    ON lprec.GameDate = gi.GameDate
    AND lprec.GameId = gi.GameId
    AND lprec.GameId2 = gi.GameId2
    AND lprec.TeamAbbr = gi.TeamAbbr    
    
--WHERE ppos.Position  IN ("RB","WR","TE") AND prec.RecTargets > 0
WHERE ppos.Position  IN ("RB") -- AND prec.RecTargets > 0
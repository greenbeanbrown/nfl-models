SELECT td.*
FROM team_defense td
JOIN (
    SELECT TeamAbbr, MAX(GameDate) AS LatestGameDate
    FROM team_defense
    GROUP BY TeamAbbr
) latest_dates
ON td.TeamAbbr = latest_dates.TeamAbbr AND td.GameDate = latest_dates.LatestGameDate
WHERE td.TeamAbbr IN (%s);
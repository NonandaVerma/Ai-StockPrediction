-- 009: project renamed to Market Prediction; the forecast controls live on Home only (data only, no schema change).

UPDATE help_texts SET description = 'Forecasts give insights, not guarantees. This tool is for research and educational purposes only and is not investment advice.',
    updated_at = strftime('%Y-%m-%dT%H:%M:%S+00:00','now')
WHERE key = 'disclaimer_short';

UPDATE help_texts SET description = 'Runs the forecast for the chosen stock and date with the current settings, and shows it on this page. Every model is first re-tested on past data, so this can take up to a minute.',
    updated_at = strftime('%Y-%m-%dT%H:%M:%S+00:00','now')
WHERE key = 'generate_forecast';

UPDATE help_texts SET description = 'Which future trading days to forecast on the Home page: the next 1-20 trading days, one date you pick, or a date range. Weekends and exchange holidays are skipped automatically.',
    updated_at = strftime('%Y-%m-%dT%H:%M:%S+00:00','now')
WHERE key = 'forecast_date';

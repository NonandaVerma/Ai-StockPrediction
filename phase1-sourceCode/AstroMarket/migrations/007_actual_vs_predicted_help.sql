-- 007: help texts for the Actual vs Predicted page (data only, no schema change).

INSERT OR IGNORE INTO help_texts(key, term, category, description) VALUES
('actual_vs_predicted', 'Actual vs Predicted', 'app', 'Pick a past period. The left chart shows the market data exactly as received. The right chart shows what the chosen model predicted for the same days without seeing their prices, so you can compare the two directly.'),
('avp_period', 'Period', 'app', 'Which completed past sessions to compare: the last week, two weeks, month, three months, or a custom range (at most 60 sessions).'),
('avp_mode', 'Prediction method', 'model', 'Strict: one forecast made after the close of the day before the period; no price from the period is used, and day k is a k-sessions-ahead forecast. Rolling: each day is predicted after the previous day''s close, using actual prices up to that day only.'),
('avp_mae', 'Average error', 'statistics', 'Mean absolute difference between the predicted and actual closing price, in percent of the actual close.'),
('avp_direction', 'Direction right', 'statistics', 'Share of days where the model correctly said whether the close would be above or below the reference close (the close before the period in strict mode, the previous close in rolling mode).'),
('avp_coverage', 'Actual inside 80% range', 'statistics', 'Share of days where the actual close fell inside the model''s 80% forecast range. A well-calibrated model is near 80% over long periods.'),
('avp_actual', 'Actual market data', 'market', 'Daily candles from the data provider, unchanged: green = closed above the open, red = closed below.'),
('avp_predicted', 'Predicted for the same dates', 'model', 'Predicted closing price (line) and 80% forecast range (shaded), on the same price scale as the actual chart.'),
('avp_table', 'Day by day', 'model', 'Actual and predicted close for each session, the error, and whether the predicted direction and range were right.');

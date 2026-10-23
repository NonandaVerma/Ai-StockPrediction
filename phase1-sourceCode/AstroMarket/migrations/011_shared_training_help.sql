-- 011: help texts for the shared training recipe and the new accuracy columns (data only, no schema change).

INSERT OR IGNORE INTO help_texts(key, term, category, description) VALUES
('accuracy_edge', 'Accuracy vs majority', 'statistics', 'Out-of-sample accuracy minus the accuracy of always guessing the more common direction. Above 0 means the model knows something; 0 or below means it does no better than always saying "up" (or always "down").'),
('accuracy vs majority', 'Accuracy vs majority', 'statistics', 'Out-of-sample accuracy minus the accuracy of always guessing the more common direction. Above 0 means real skill.'),
('confident_accuracy', 'Confident-call accuracy', 'statistics', 'Accuracy on the days the model made a bullish or bearish call (P(up) outside the neutral band). Fewer calls, but these are the ones a user would act on.'),
('confident-call accuracy', 'Confident-call accuracy', 'statistics', 'Accuracy on the days the model made a bullish or bearish call (P(up) outside the neutral band).'),
('confident_share', 'Share of confident calls', 'statistics', 'Share of days on which the model made a bullish or bearish call instead of neutral. A high confident-call accuracy on a tiny share of days is weak evidence.'),
('avp_confident', 'Confident calls right', 'statistics', 'Of the days in the period where the model made a bullish or bearish call (not neutral), the share where the direction was right.'),
('confident', 'Confident call', 'model', 'True when P(up) is outside the neutral band, i.e. the model made a bullish or bearish call for that day.'),
('eval_start', 'Evaluation start', 'statistics', 'First date of this model''s out-of-sample predictions. Models that need data starting later (e.g. India VIX, weather) begin later; head-to-head tests use only shared dates.'),
('training_recipe', 'Training recipe', 'model', 'Every page (Future Forecast, Home, Model Comparison, Backtesting, Actual vs Predicted) trains models the same way, set in config.yaml under forecast: folds, min_train_sessions, train_window_years, min_move and neutral_band. The same model, data and date give the same prediction on every page.');

-- 006: help texts for the volatility target on Model Comparison (data only, no schema change).

INSERT OR IGNORE INTO help_texts(key, term, category, description) VALUES
('cmp_target', 'Target', 'model', 'What the models predict. Direction: will the price be higher after the window? Volatility: how much will the price swing over the window (annualised realised volatility)? Volatility clusters, so it is far more predictable than direction.'),
('vol_target', 'Volatility target', 'model', 'Each model forecasts the annualised realised volatility of the next sessions. The benchmark is HAR, which combines the last week, month and quarter of volatility. Every other model is HAR plus one data category, tested on the same dates with the same walk-forward folds; all fitting uses training data only.'),
('vol_groups', 'Data categories to add to HAR', 'model', 'Each selected category is tested as HAR + that category. More selections mean more comparisons, which the Holm-adjusted p-value accounts for.'),
('vol_chart', 'Predicted vs realised volatility', 'model', 'Out-of-sample forecasts of the HAR benchmark and the best-scoring category model against the volatility that actually followed.');

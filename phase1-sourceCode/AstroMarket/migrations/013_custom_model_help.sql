-- 013: help texts for choosing data categories and the algorithm on Actual vs Predicted (data only).

INSERT OR IGNORE INTO help_texts(key, term, category, description) VALUES
('avp_categories', 'Data categories', 'model', 'Pick one or more data categories, for example Astrology, Weather and Geopolitics. One model is trained on all selected categories together and predicts the period. Add Market to include the usual price-based features; categories without data for this stock (e.g. India weather for a US stock) are reported, never filled in.'),
('avp_algorithm', 'Algorithm', 'model', 'Linear: regularised linear model (L1 for up/down, ridge for the return). LightGBM: gradient-boosted trees that can pick up non-linear patterns. Linear + LightGBM: the average of both, usually steadier than either alone.');

-- 004: help text for the hyperparameter-tuning toggle (data only, no schema change).

INSERT OR IGNORE INTO help_texts(key, term, category, description) VALUES
('tune_hyperparams', 'Tune hyperparameters', 'model', 'When on, LightGBM/XGBoost pick their own settings per walk-forward fold from a small candidate list, validated on data the fold has not seen (never the test block). Off = the one fixed default used everywhere else. Roughly 4-8x slower; falls back to the default when there is not enough data for a reliable inner test.');

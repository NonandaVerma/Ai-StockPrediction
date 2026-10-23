-- 008: predicted chart now uses candles (data only, no schema change).

UPDATE help_texts SET description = 'Predicted daily candles drawn like the actual chart, in lighter green / red. Close = the model''s forecast; open = the previous predicted close (day 1: the last real close before the period); high / low = the instrument''s typical wicks over the 60 sessions before the forecast date. Turn on the 80% forecast range to see the uncertainty of the close.',
    updated_at = strftime('%Y-%m-%dT%H:%M:%S+00:00','now')
WHERE key = 'avp_predicted';

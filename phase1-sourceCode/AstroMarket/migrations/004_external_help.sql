-- 004: help texts for the geopolitics and weather models (data only, no schema change).

INSERT OR IGNORE INTO help_texts(key, term, category, description) VALUES
('external_value', 'Do geopolitics or weather add predictive value?', 'statistics', 'Each geopolitics or weather model is compared on the same dates with the market-only model of the same type. A difference counts only if its p-value is below 0.05.'),
('geopolitics', 'Geopolitics (GPR)', 'model', 'Geopolitical Risk index by Caldara and Iacoviello, built from newspaper coverage of wars, threats and terrorism: global daily levels, threats, acts and 1-month change, plus a monthly India index for NSE/BSE listings. Daily values are used only 7 days after the date they describe and monthly values 15 days after month end, because they are published with a delay and can be revised.'),
('weather', 'India weather (NASA POWER)', 'model', 'Rainfall and temperature averaged over 10 Indian cities: 30-day rainfall anomaly, monsoon rainfall to date versus normal (June to September) and 7-day temperature anomaly. "Normal" uses earlier years only. Values are used 5 days after the day they describe. Offered for NSE/BSE listings only.');

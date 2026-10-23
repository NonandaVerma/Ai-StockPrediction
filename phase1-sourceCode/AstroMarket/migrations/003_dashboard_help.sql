-- 003: help texts for the beginner dashboard (data only, no schema change).

INSERT OR IGNORE INTO help_texts(key, term, category, description) VALUES
('forecast_date', 'Forecast Date', 'app', 'Which future trading days to forecast: the next 1-20 trading days, one date you pick, or a date range. Weekends and exchange holidays are skipped automatically.'),
('generate_forecast', 'Generate Forecast', 'app', 'Runs the forecast for the chosen stock, date and astrology system. Every model is first re-tested on past data, so this can take up to a minute.'),
('settings', 'Settings', 'app', 'Advanced options: exchange, ayanamsa, lunar node, aspect orb and which model gives the headline forecast. The defaults suit most users.'),
('price_forecast_chart', 'Price & Forecast chart', 'market', 'Candles show actual prices: green = closed higher than it opened, red = closed lower. After the dotted line, the dashed line is the expected path and the shaded area the 80% forecast range.'),
('forecast_summary', 'Forecast summary', 'model', 'The headline forecast in plain words: expected direction, likely return range, how volatile the stock is and how reliable this model has been in past tests.'),
('key_reasons', 'Key Reasons', 'model', 'What the forecast is based on, taken directly from the model output: its probability, the factors it relied on, similar past skies and how well it tested.'),
('current_positions', 'Current Planetary Positions', 'astrology', 'The zodiac sign and degree of each planet right now, in the astrology system you selected. ℞ marks a retrograde planet.'),
('past_similar_dates', 'Past Similar Dates', 'statistics', 'Past trading days when the fast-moving planets were positioned most like the forecast period, and how the price moved afterwards. Matching the Sun also matches the season.'),
('what_this_means', 'What This Means', 'app', 'A plain-language reading of the forecast: the planetary picture during the target period, how similar skies played out before, and what to keep in mind.'),
('keep_in_mind', 'Keep in Mind', 'app', 'This is a probability, not a guarantee. Astrology is not an established predictor of prices; market news, the economy and global events also move prices. Nothing here is investment advice.'),
('disclaimer_short', 'Disclaimer', 'app', 'Astrology gives insights, not guarantees. This tool is for research and educational purposes only and is not investment advice.');

-- 010: Planetary Timeline page removed; Home groups its astrology cards as one data category (data only, no schema change).

INSERT OR IGNORE INTO help_texts(key, term, category, description) VALUES
('astro_category', 'Category: Astrology', 'app', 'Astrology is one of the data categories the models can use, next to market, macro, global, geopolitical and weather data. These cards show the current sky, past dates with a similar sky, and what that sky meant for prices before. Model Comparison tests whether this category adds predictive value.');

DELETE FROM help_texts WHERE key IN ('timeline_range', 'bodies', 'event_types');

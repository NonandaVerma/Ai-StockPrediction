-- 005: help texts for the per-category models on Model Comparison (data only, no schema change).

UPDATE help_texts SET term = 'Does each data category add predictive value?',
    description = 'Each "market + category" model (one per catalog category, including geopolitics and weather) is compared on the same dates with the market-only model of the same type. With many categories some look better by chance, so the Holm column adjusts the p-values for the number of comparisons; a difference counts only if the adjusted p-value is below 0.05.',
    updated_at = strftime('%Y-%m-%dT%H:%M:%S+00:00','now')
WHERE key = 'external_value';

INSERT OR IGNORE INTO help_texts(key, term, category, description) VALUES
('data_coverage', 'Data coverage and lineage', 'statistics', 'For every catalog category: how many features were built for this instrument, the first session where all of them exist, and the share of the last 3 years with any value missing. Categories that need licensed or archived data are listed with the reason instead of being approximated.');

-- 012: updated Model comparison help text (data only, no schema change).
-- Model comparison no longer forces every model onto the dates of the latest-starting one. Only replaces
-- the original wording, so a text edited in the Glossary is kept.
UPDATE help_texts SET description = 'Every model tested walk-forward with the same training recipe as Future Forecast and Actual vs Predicted: trained only on data before each test period, then scored on the period it did not see. Each model is scored on all of its own dates; the head-to-head tests use only dates both models share.'
WHERE key = 'model_comparison' AND description = 'Every model tested walk-forward on identical past dates: trained only on data before each test period, then scored on the period it did not see.';

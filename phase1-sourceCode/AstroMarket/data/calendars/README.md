# Exchange holiday calendars

Future trading sessions are projected as weekdays minus the holidays listed here.
Add one CSV per exchange, named `<EXCHANGE>_holidays.csv` (codes: NSE, BSE, NYSE, NASDAQ, LSE):

```csv
date,name
2027-01-26,Republic Day
```

Take dates from the exchange's official holiday circular. No lists are shipped, because
guessed dates would silently shift forecasts. While a year is missing, the dashboard warns
that only weekends are excluded for that year.
Past sessions do not need this file: they come from the price history itself.

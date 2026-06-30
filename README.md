# BSM Option Pricer

A Streamlit app that prices US equity options using the Black-Scholes-Merton
model, with implied volatility independently solved from live market quotes
(not Yahoo Finance's pre-computed IV).

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. **Ticker resolution** — type a company name ("Apple") or ticker ("AAPL").
   The app searches Yahoo Finance and lets you disambiguate if multiple
   matches are found (e.g. different share classes / exchanges).
2. **Market data** — spot price, dividend yield, and listed option
   expirations are pulled live via `yfinance` and cached (`@st.cache_data`)
   to keep repeated lookups fast.
3. **Risk-free rate** — by default, the app fetches the live US Treasury
   yield curve from Yahoo Finance (`^IRX` 3-month, `^FVX` 5-year, `^TNX`
   10-year, `^TYX` 30-year) and linearly interpolates it to your option's
   exact time-to-expiry (e.g. a 50-day option uses a rate between the
   3-month bill and 5-year note yields, weighted by maturity). Flip the
   "Use live Treasury yield curve" toggle off to enter a flat manual rate
   instead.
4. **Implied volatility** — for the two listed expirations that bracket your
   requested expiration date, the app solves implied volatility **per
   strike** from real bid/ask quotes using Brent's method against the BSM
   formula (Yahoo's own `impliedVolatility` field is intentionally ignored).
   It then:
   - interpolates within each expiry's smile to your exact strike, then
   - linearly interpolates between the two expiries' smiles to your exact
     requested date.
5. **Pricing** — the interpolated IV, along with spot, strike, time to
   expiry, maturity-matched risk-free rate, and dividend yield, is fed into
   the closed-form Black-Scholes-Merton formula to produce the theoretical
   price, Greeks, the IV smile chart, and a payoff diagram.
6. **Knock-out barriers (optional)** — toggle "Knock-out barrier
   (expiry-only)" to price an up-and-out or down-and-out call/put instead of
   a vanilla option. The barrier is checked only at expiry (not
   continuously along the path), which keeps it closed-form under
   Black-Scholes-Merton — no Monte Carlo needed. The same market-implied
   volatility from the vanilla option chain is used as the input. Greeks
   for barrier options are computed via finite differences, since the
   barrier formula has no simple closed-form derivatives.

## Files

- `app.py` — Streamlit UI (dark theme, input panel, results panel, charts).
- `bsm_engine.py` — all financial math and Yahoo Finance data access,
  independent of Streamlit (testable on its own).

## Error handling

The app distinguishes and gives clear messages for: unresolvable
company/ticker, invalid or implausible strike prices, past/invalid
expiration dates, invalid barrier levels (e.g. an "up" barrier already
below current spot), missing option data (e.g. no listed options, empty
chain for a given expiry, insufficient liquid quotes to solve IV), failure
to fetch the live Treasury yield curve (with a prompt to switch to a
manual rate), and other network/API failures — without ever surfacing a
raw traceback.

## Disclaimer

For educational/analytical purposes only. Not investment advice.

"""
bsm_engine.py
=============
Core engine for the Black-Scholes-Merton option pricer.

This module is intentionally UI-agnostic: it knows nothing about Streamlit.
It is responsible for:
    1. Resolving a free-text company name / ticker into a valid Yahoo Finance
       ticker symbol (with disambiguation support for multiple matches).
    2. Pulling market data (spot price, dividend yield, option expirations,
       option chains) via yfinance.
    3. Building a market-implied volatility surface from listed option chains
       (via Brent's method root-finding on the BSM price, NOT yfinance's
       pre-computed `impliedVolatility` column) and interpolating it to the
       user's requested strike and expiration.
    4. Pricing the option with the closed-form Black-Scholes-Merton formula.

All public functions raise the custom exceptions defined below so the UI
layer can catch them and render clean, specific error messages instead of
raw tracebacks.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class OptionPricerError(Exception):
    """Base class for all expected/handled errors in this engine."""


class TickerNotFoundError(OptionPricerError):
    """Raised when a company name / ticker cannot be resolved at all."""


class NetworkError(OptionPricerError):
    """Raised when a call to Yahoo Finance fails for network/API reasons."""


class NoOptionDataError(OptionPricerError):
    """Raised when a ticker has no listed options, or a chain is empty."""


class InvalidExpirationError(OptionPricerError):
    """Raised when the requested expiration date is invalid (e.g. in the past)."""


class InvalidStrikeError(OptionPricerError):
    """Raised when the requested strike price is invalid (e.g. <= 0, absurdly
    far from spot, or no usable strikes exist nearby to interpolate from)."""


class ImpliedVolatilityError(OptionPricerError):
    """Raised when implied volatility cannot be solved/interpolated reliably."""


class RiskFreeRateError(OptionPricerError):
    """Raised when the live Treasury yield curve cannot be retrieved or built."""


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass
class CompanyMatch:
    """A single candidate result from a ticker/company search."""

    symbol: str
    name: str
    exchange: str = ""
    quote_type: str = ""

    @property
    def label(self) -> str:
        """Human-friendly label for display in a selection widget."""
        bits = [self.symbol]
        if self.name:
            bits.append(f"- {self.name}")
        if self.exchange:
            bits.append(f"({self.exchange})")
        return " ".join(bits)


@dataclass
class MarketSnapshot:
    """Everything pulled from Yahoo Finance needed to price one option."""

    ticker: str
    spot_price: float
    dividend_yield: float
    expirations: list[str]  # ISO date strings, as returned by yfinance
    company_name: str = ""


@dataclass
class IVInterpolationResult:
    """Result of interpolating implied volatility to the requested point."""

    iv: float
    method: str  # description of how the IV was derived, for transparency
    near_expiry: str
    far_expiry: Optional[str]
    near_iv: float
    far_iv: Optional[float]
    weight_far: float  # 0 if only near_expiry was used


@dataclass
class RiskFreeRateResult:
    """Result of resolving a risk-free rate for a specific maturity."""

    rate: float
    source: str  # "live" or "manual"
    method: str  # human-readable description of how it was derived
    curve_points: list[tuple[float, float]] = field(default_factory=list)
    # curve_points: list of (maturity_in_years, yield_as_fraction) actually used


@dataclass
class PricingResult:
    """Final output bundle returned to the UI layer."""

    spot_price: float
    strike: float
    option_type: str
    time_to_expiry_years: float
    risk_free_rate: float
    dividend_yield: float
    iv_result: IVInterpolationResult
    rf_result: Optional["RiskFreeRateResult"] = None
    option_price: float = 0.0
    d1: float = float("nan")
    d2: float = float("nan")
    greeks: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 1. Ticker resolution
# --------------------------------------------------------------------------- #


def search_company(query: str) -> list[CompanyMatch]:
    """Resolve free-text (company name or ticker) into candidate tickers.

    Uses yfinance's search endpoint (wraps Yahoo Finance's autocomplete /
    search API). Falls back gracefully and raises informative errors instead
    of letting network exceptions bubble up raw.

    Parameters
    ----------
    query : str
        Company name or ticker symbol, as typed by the user.

    Returns
    -------
    list[CompanyMatch]
        Candidate matches, equities/ETFs only, best-first.

    Raises
    ------
    TickerNotFoundError
        If the query is empty or no matches are found.
    NetworkError
        If the underlying Yahoo Finance request fails outright.
    """
    query = (query or "").strip()
    if not query:
        raise TickerNotFoundError("Please enter a company name or ticker symbol.")

    try:
        searcher = yf.Search(query, max_results=10)
        raw_quotes = searcher.quotes or []
    except Exception as exc:  # network/API failure of any kind
        raise NetworkError(
            f"Could not reach Yahoo Finance to search for '{query}'. "
            f"Please check your connection and try again. (Details: {exc})"
        ) from exc

    matches: list[CompanyMatch] = []
    for q in raw_quotes:
        quote_type = (q.get("quoteType") or "").upper()
        # Keep equities/ETFs; option chains only really make sense for these.
        if quote_type not in ("EQUITY", "ETF"):
            continue
        symbol = q.get("symbol")
        if not symbol:
            continue
        matches.append(
            CompanyMatch(
                symbol=symbol,
                name=q.get("shortname") or q.get("longname") or "",
                exchange=q.get("exchange", ""),
                quote_type=quote_type,
            )
        )

    if not matches:
        # Last-resort fallback: maybe the user already typed an exact, valid
        # ticker that the fuzzy search API just didn't return (rare, but
        # cheap to check).
        if _quick_ticker_sanity_check(query.upper()):
            matches.append(CompanyMatch(symbol=query.upper(), name=""))
        else:
            raise TickerNotFoundError(
                f"No matching company or ticker found for '{query}'. "
                "Please check the spelling and try again."
            )

    return matches


def _quick_ticker_sanity_check(symbol: str) -> bool:
    """Cheap existence check used only as a fallback when search returns nothing."""
    try:
        info = yf.Ticker(symbol).fast_info
        return info is not None and info.get("lastPrice") is not None
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 2. Market data retrieval
# --------------------------------------------------------------------------- #


def get_market_snapshot(ticker: str) -> MarketSnapshot:
    """Fetch spot price, dividend yield, and available expirations for a ticker.

    Raises
    ------
    TickerNotFoundError
        If the ticker is delisted/invalid (no price data at all).
    NetworkError
        If Yahoo Finance cannot be reached.
    NoOptionDataError
        If the ticker is valid but has no listed options.
    """
    try:
        tk = yf.Ticker(ticker)
        fast_info = tk.fast_info
        spot_price = fast_info.get("lastPrice") if fast_info else None

        if spot_price is None:
            # fast_info can be sparse for some tickers; fall back to history.
            hist = tk.history(period="5d")
            if hist.empty:
                raise TickerNotFoundError(
                    f"'{ticker}' does not appear to be a valid, actively traded "
                    "ticker. Please verify the symbol."
                )
            spot_price = float(hist["Close"].iloc[-1])

        spot_price = float(spot_price)

        # Dividend yield: prefer the structured field, fall back to trailing
        # dividends / price if Yahoo only exposes a raw dividend rate.
        info = {}
        try:
            info = tk.get_info()
        except Exception:
            pass  # non-fatal; we just lose dividend precision

        div_yield = info.get("dividendYield")
        if div_yield is None:
            div_yield = 0.0
        else:
            div_yield = float(div_yield)
            # yfinance has, at various versions, returned this as a fraction
            # (0.005) or as a percentage points (0.50 meaning 0.50%). Guard
            # against the unit ambiguity: anything above 0.5 (50%) for a
            # "yield" is almost certainly already a percentage, not a fraction.
            if div_yield > 0.5:
                div_yield = div_yield / 100.0

        try:
            expirations = list(tk.options)
        except Exception as exc:
            raise NetworkError(
                f"Could not retrieve option expirations for '{ticker}'. "
                f"(Details: {exc})"
            ) from exc

        if not expirations:
            raise NoOptionDataError(
                f"'{ticker}' has no listed options on Yahoo Finance."
            )

        company_name = info.get("shortName") or info.get("longName") or ticker

        return MarketSnapshot(
            ticker=ticker,
            spot_price=spot_price,
            dividend_yield=div_yield,
            expirations=expirations,
            company_name=company_name,
        )

    except OptionPricerError:
        raise
    except Exception as exc:
        raise NetworkError(
            f"Unexpected error retrieving market data for '{ticker}': {exc}"
        ) from exc


def get_option_chain(ticker: str, expiration: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the (calls, puts) option chain DataFrames for one expiration.

    Raises
    ------
    NoOptionDataError
        If the chain for this expiration is empty or unavailable.
    NetworkError
        If the request to Yahoo Finance fails.
    """
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiration)
    except Exception as exc:
        raise NetworkError(
            f"Could not retrieve the option chain for '{ticker}' "
            f"expiring {expiration}. (Details: {exc})"
        ) from exc

    calls, puts = chain.calls, chain.puts
    if calls.empty and puts.empty:
        raise NoOptionDataError(
            f"No option data available for '{ticker}' expiring {expiration}."
        )
    return calls, puts


# --------------------------------------------------------------------------- #
# 2b. Risk-free rate: live Treasury yield curve
# --------------------------------------------------------------------------- #

# CBOE/Yahoo Treasury yield index tickers. Each is quoted in percentage
# points (e.g. a history `Close` of 4.85 means 4.85%), and each tracks the
# yield of the on-the-run Treasury security of that approximate maturity.
_TREASURY_TICKERS: list[tuple[float, str]] = [
    (0.25, "^IRX"),  # 13-week T-bill
    (5.0, "^FVX"),   # 5-year note
    (10.0, "^TNX"),  # 10-year note
    (30.0, "^TYX"),  # 30-year bond
]


def fetch_treasury_curve() -> list[tuple[float, float]]:
    """Fetch the latest available yield for each Treasury maturity bucket.

    Returns
    -------
    list[(maturity_in_years, yield_as_fraction)], sorted by maturity,
    containing only the maturities that were successfully retrieved.

    Raises
    ------
    RiskFreeRateError
        If none of the Treasury tickers could be retrieved at all.
    """
    points: list[tuple[float, float]] = []
    last_error: Optional[Exception] = None

    for maturity_years, symbol in _TREASURY_TICKERS:
        try:
            hist = yf.Ticker(symbol).history(period="5d")
            if hist.empty or "Close" not in hist:
                continue
            closes = hist["Close"].dropna()
            if closes.empty:
                continue
            yield_pct = float(closes.iloc[-1])
            if yield_pct <= 0 or yield_pct > 50:  # sanity bound, guards bad ticks
                continue
            points.append((maturity_years, yield_pct / 100.0))
        except Exception as exc:  # keep going; we only need the survivors
            last_error = exc
            continue

    if not points:
        raise RiskFreeRateError(
            "Could not retrieve live Treasury yields from Yahoo Finance "
            f"(tried {', '.join(t for _, t in _TREASURY_TICKERS)}). "
            f"{'Details: ' + str(last_error) if last_error else ''} "
            "You can switch to a manual risk-free rate instead."
        )

    return sorted(points)


def interpolate_risk_free_rate(
    curve: list[tuple[float, float]], time_to_expiry_years: float
) -> RiskFreeRateResult:
    """Interpolate the Treasury curve to the option's exact time-to-expiry.

    Mirrors the IV-interpolation design used elsewhere in this engine:
    linear interpolation between the two bracketing maturities, with flat
    extrapolation beyond the shortest/longest available maturity (e.g. an
    expiry inside the 3-month bucket uses the 3-month bill rate directly;
    an expiry beyond 30 years, if that ever occurs, uses the 30-year rate).

    Raises
    ------
    RiskFreeRateError
        If the curve is empty.
    """
    if not curve:
        raise RiskFreeRateError("No Treasury yield curve points available to interpolate.")

    maturities = np.array([m for m, _ in curve])
    yields = np.array([y for _, y in curve])

    if len(curve) == 1:
        rate = float(yields[0])
        return RiskFreeRateResult(
            rate=rate, source="live",
            method=f"Only one Treasury maturity available ({maturities[0]:.2f}y); used directly.",
            curve_points=curve,
        )

    t = time_to_expiry_years
    if t <= maturities[0]:
        rate = float(yields[0])
        method = (
            f"Expiry ({t:.2f}y) is at or below the shortest available Treasury "
            f"maturity ({maturities[0]:.2f}y) — used that yield directly."
        )
    elif t >= maturities[-1]:
        rate = float(yields[-1])
        method = (
            f"Expiry ({t:.2f}y) is at or beyond the longest available Treasury "
            f"maturity ({maturities[-1]:.2f}y) — used that yield directly."
        )
    else:
        rate = float(np.interp(t, maturities, yields))
        # Identify the bracketing pair for a clear, specific message.
        idx = np.searchsorted(maturities, t)
        lo_m, hi_m = maturities[idx - 1], maturities[idx]
        lo_y, hi_y = yields[idx - 1], yields[idx]
        method = (
            f"Linearly interpolated between the {lo_m:.2f}y Treasury yield "
            f"({lo_y*100:.2f}%) and the {hi_m:.2f}y Treasury yield "
            f"({hi_y*100:.2f}%) to match the option's {t:.2f}y time to expiry."
        )

    return RiskFreeRateResult(rate=rate, source="live", method=method, curve_points=curve)


def get_live_risk_free_rate(time_to_expiry_years: float) -> RiskFreeRateResult:
    """Convenience wrapper: fetch the live curve and interpolate it to a maturity."""
    curve = fetch_treasury_curve()
    return interpolate_risk_free_rate(curve, time_to_expiry_years)


# --------------------------------------------------------------------------- #
# 3. Black-Scholes-Merton core math
# --------------------------------------------------------------------------- #


def bsm_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: Literal["call", "put"],
) -> tuple[float, float, float]:
    """Closed-form Black-Scholes-Merton price for a European option.

    Returns
    -------
    (price, d1, d2)
    """
    if time_to_expiry <= 0:
        # At/after expiry: price collapses to intrinsic value.
        intrinsic = (
            max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        )
        return intrinsic, float("nan"), float("nan")

    if volatility <= 0:
        # Degenerate case: zero-vol forward payoff, discounted.
        forward = spot * np.exp(-dividend_yield * time_to_expiry)
        pv_strike = strike * np.exp(-risk_free_rate * time_to_expiry)
        intrinsic = max(forward - pv_strike, 0.0) if option_type == "call" else max(
            pv_strike - forward, 0.0
        )
        return intrinsic, float("nan"), float("nan")

    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    disc_div = np.exp(-dividend_yield * time_to_expiry)
    disc_rf = np.exp(-risk_free_rate * time_to_expiry)

    if option_type == "call":
        price = spot * disc_div * norm.cdf(d1) - strike * disc_rf * norm.cdf(d2)
    else:
        price = strike * disc_rf * norm.cdf(-d2) - spot * disc_div * norm.cdf(-d1)

    return float(price), float(d1), float(d2)


def bsm_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: Literal["call", "put"],
    d1: float,
    d2: float,
) -> dict:
    """Standard BSM greeks. Returns NaNs gracefully at/after expiry."""
    if time_to_expiry <= 0 or volatility <= 0 or np.isnan(d1):
        return {"delta": float("nan"), "gamma": float("nan"), "vega": float("nan"),
                "theta": float("nan"), "rho": float("nan")}

    sqrt_t = np.sqrt(time_to_expiry)
    disc_div = np.exp(-dividend_yield * time_to_expiry)
    disc_rf = np.exp(-risk_free_rate * time_to_expiry)
    pdf_d1 = norm.pdf(d1)

    gamma = (disc_div * pdf_d1) / (spot * volatility * sqrt_t)
    vega = spot * disc_div * pdf_d1 * sqrt_t / 100.0  # per 1 vol-point (1%)

    if option_type == "call":
        delta = disc_div * norm.cdf(d1)
        theta = (
            -spot * disc_div * pdf_d1 * volatility / (2 * sqrt_t)
            - risk_free_rate * strike * disc_rf * norm.cdf(d2)
            + dividend_yield * spot * disc_div * norm.cdf(d1)
        ) / 365.0
        rho = strike * time_to_expiry * disc_rf * norm.cdf(d2) / 100.0
    else:
        delta = disc_div * (norm.cdf(d1) - 1.0)
        theta = (
            -spot * disc_div * pdf_d1 * volatility / (2 * sqrt_t)
            + risk_free_rate * strike * disc_rf * norm.cdf(-d2)
            - dividend_yield * spot * disc_div * norm.cdf(-d1)
        ) / 365.0
        rho = -strike * time_to_expiry * disc_rf * norm.cdf(-d2) / 100.0

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# --------------------------------------------------------------------------- #
# 3b. Expiry-only knock-out barrier options (closed-form under BSM)
# --------------------------------------------------------------------------- #


class InvalidBarrierError(OptionPricerError):
    """Raised when a barrier level is invalid (e.g. wrong side of spot for
    its direction, or non-positive)."""


def bsm_barrier_price_expiry_only(
    spot: float,
    strike: float,
    barrier: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: Literal["call", "put"],
    barrier_direction: Literal["up", "down"],
) -> float:
    """Closed-form BSM price for a knock-out option where the barrier is
    checked ONLY at expiry (not continuously along the path).

    This is materially simpler than the standard continuously-monitored
    barrier option (which requires the Reiner-Rubinstein formulas and the
    probability of the path ever touching the barrier). Because monitoring
    happens only at expiry, the payoff is just a vanilla call/put payoff
    truncated to the region of the terminal price where the option survives
    -- so the price falls out directly from the same lognormal terminal
    distribution Black-Scholes already assumes, with no path-touching term.

    Payoff at expiry (before discounting):
        up-and-out call:   max(S_T - K, 0)  if S_T <  B,  else 0   (needs B > K to have any value)
        down-and-out call: max(S_T - K, 0)  if S_T >  B,  else 0   (if B <= K, barrier removes
                                                                      nothing -> collapses to vanilla)
        up-and-out put:    max(K - S_T, 0)  if S_T <  B,  else 0   (if B >= K, barrier removes
                                                                      nothing -> collapses to vanilla)
        down-and-out put:  max(K - S_T, 0)  if S_T >  B,  else 0   (needs B < K to have any value)

    Each case is derived as a vanilla option price minus a "truncated tail"
    correction beyond the barrier, evaluated via a second vanilla price
    struck at the barrier itself plus a discounted digital-style term. This
    has been verified against Monte Carlo simulation to within statistical
    noise across multiple barrier placements.

    Parameters
    ----------
    barrier_direction : "up" or "down"
        "up"   -> knocked out if S_T ends up AT OR ABOVE the barrier.
        "down" -> knocked out if S_T ends up AT OR BELOW the barrier.

    Raises
    ------
    InvalidBarrierError
        If the barrier is non-positive, or sits on the wrong side of the
        current spot price for its stated direction (e.g. an "up" barrier
        below spot would already be breached today).
    """
    if barrier <= 0:
        raise InvalidBarrierError("Barrier level must be a positive number.")

    # The "barrier must be on the correct side of spot" check only makes
    # sense BEFORE expiry, when the barrier hasn't been observed yet. At or
    # after expiry, spot effectively IS the terminal observation -- the
    # whole point is to test it against the barrier, not to reject it.
    if time_to_expiry > 0:
        if barrier_direction == "up" and barrier <= spot:
            raise InvalidBarrierError(
                f"An 'up' barrier (${barrier:,.2f}) must be above the current spot "
                f"price (${spot:,.2f}) -- otherwise the option would already be "
                "knocked out today."
            )
        if barrier_direction == "down" and barrier >= spot:
            raise InvalidBarrierError(
                f"A 'down' barrier (${barrier:,.2f}) must be below the current spot "
                f"price (${spot:,.2f}) -- otherwise the option would already be "
                "knocked out today."
            )

    if time_to_expiry <= 0:
        # At/after expiry: just apply the knock-out test directly to spot.
        if barrier_direction == "up" and spot >= barrier:
            return 0.0
        if barrier_direction == "down" and spot <= barrier:
            return 0.0
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)

    def _vanilla(K: float) -> float:
        price, _, _ = bsm_price(
            spot, K, time_to_expiry, risk_free_rate, dividend_yield, volatility, option_type
        )
        return price

    def _d2(K: float) -> float:
        sqrt_t = np.sqrt(time_to_expiry)
        d1 = (
            np.log(spot / K)
            + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry
        ) / (volatility * sqrt_t)
        return d1 - volatility * sqrt_t

    disc = np.exp(-risk_free_rate * time_to_expiry)

    if option_type == "call" and barrier_direction == "up":
        # Up-and-out call: only has value if the barrier sits above the strike.
        if barrier <= strike:
            return 0.0
        return float(
            _vanilla(strike) - _vanilla(barrier)
            - (barrier - strike) * disc * norm.cdf(_d2(barrier))
        )

    if option_type == "call" and barrier_direction == "down":
        # Down-and-out call: if barrier is at/below strike it removes nothing.
        if barrier <= strike:
            return float(_vanilla(strike))
        return float(
            _vanilla(barrier) + (barrier - strike) * disc * norm.cdf(_d2(barrier))
        )

    if option_type == "put" and barrier_direction == "down":
        # Down-and-out put: only has value if the barrier sits below the strike.
        if barrier >= strike:
            return 0.0
        return float(
            _vanilla(strike) - _vanilla(barrier)
            - (strike - barrier) * disc * norm.cdf(-_d2(barrier))
        )

    if option_type == "put" and barrier_direction == "up":
        # Up-and-out put: if barrier is at/above strike it removes nothing.
        if barrier >= strike:
            return float(_vanilla(strike))
        return float(
            _vanilla(barrier) + (strike - barrier) * disc * norm.cdf(-_d2(barrier))
        )

    raise ValueError(f"Unhandled combination: {option_type=}, {barrier_direction=}")


def solve_implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: Literal["call", "put"],
    vol_bounds: tuple[float, float] = (1e-4, 5.0),
) -> Optional[float]:
    """Back out implied volatility from an observed market price via Brent's method.

    Returns None (rather than raising) if no root exists in the bracket or the
    market price is outside arbitrage-free bounds -- callers should simply
    skip that data point rather than fail the whole interpolation.
    """
    if market_price <= 0 or time_to_expiry <= 0:
        return None

    # Arbitrage bounds check: a quote below intrinsic value or above the
    # max possible value (spot, roughly) cannot map to a valid BSM vol.
    intrinsic_disc = (
        max(spot * np.exp(-dividend_yield * time_to_expiry)
            - strike * np.exp(-risk_free_rate * time_to_expiry), 0.0)
        if option_type == "call"
        else max(strike * np.exp(-risk_free_rate * time_to_expiry)
                  - spot * np.exp(-dividend_yield * time_to_expiry), 0.0)
    )
    if market_price < intrinsic_disc - 1e-6:
        return None

    def objective(vol: float) -> float:
        price, _, _ = bsm_price(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, vol, option_type
        )
        return price - market_price

    lo, hi = vol_bounds
    try:
        f_lo, f_hi = objective(lo), objective(hi)
        if f_lo * f_hi > 0:
            return None  # no sign change -> no root in bracket
        return float(brentq(objective, lo, hi, xtol=1e-6, maxiter=200))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 4. Implied volatility surface construction & interpolation
# --------------------------------------------------------------------------- #


def _mid_price(row: pd.Series) -> Optional[float]:
    """Robust mid-price from a yfinance option chain row.

    Prefers bid/ask mid; falls back to lastPrice if the quote is stale/zero.
    """
    bid, ask = row.get("bid", np.nan), row.get("ask", np.nan)
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0
    last = row.get("lastPrice", np.nan)
    if pd.notna(last) and last > 0:
        return float(last)
    return None


def build_iv_smile(
    ticker: str,
    expiration: str,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: Literal["call", "put"],
    valuation_date: Optional[dt.date] = None,
) -> pd.DataFrame:
    """Build a strike -> implied-volatility smile for a single expiration.

    Implied vols are solved per-strike from observed market mid-prices using
    Brent's method against the BSM formula (yfinance's own `impliedVolatility`
    column is deliberately ignored, per spec).

    Returns
    -------
    pd.DataFrame with columns ['strike', 'iv'], sorted by strike, containing
    only strikes where a valid IV could be solved.
    """
    calls, puts = get_option_chain(ticker, expiration)
    chain = calls if option_type == "call" else puts

    if chain.empty:
        raise NoOptionDataError(
            f"No {option_type} options available for '{ticker}' expiring {expiration}."
        )

    valuation_date = valuation_date or dt.date.today()
    exp_date = dt.datetime.strptime(expiration, "%Y-%m-%d").date()
    tau = max((exp_date - valuation_date).days, 0) / 365.0

    records = []
    for _, row in chain.iterrows():
        strike = row.get("strike")
        if pd.isna(strike):
            continue
        mid = _mid_price(row)
        if mid is None:
            continue
        # Skip illiquid/zero-volume-and-zero-OI rows that tend to carry stale,
        # unreliable quotes which distort the smile.
        vol_oi = (row.get("volume") or 0) + (row.get("openInterest") or 0)
        if vol_oi <= 0 and pd.isna(row.get("bid")):
            continue

        iv = solve_implied_volatility(
            market_price=mid,
            spot=spot,
            strike=float(strike),
            time_to_expiry=tau if tau > 0 else 1e-6,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
        if iv is not None and 0.01 <= iv <= 4.0:  # sane vol bounds, drop outliers
            records.append({"strike": float(strike), "iv": iv})

    smile = pd.DataFrame.from_records(records).drop_duplicates(subset="strike")
    if smile.empty:
        raise ImpliedVolatilityError(
            f"Could not solve implied volatility for any {option_type} strikes "
            f"on '{ticker}' expiring {expiration} (insufficient liquid quotes)."
        )
    return smile.sort_values("strike").reset_index(drop=True)


def interpolate_iv_for_strike(smile: pd.DataFrame, strike: float) -> float:
    """Linearly interpolate (and flat-extrapolate at the ends) IV across strikes.

    Using linear interpolation in strike space is a standard, robust choice
    for a single-expiration smile; flat extrapolation avoids nonsensical
    vol explosions just outside the observed strike range.
    """
    strikes = smile["strike"].to_numpy()
    ivs = smile["iv"].to_numpy()

    if len(strikes) == 1:
        return float(ivs[0])

    if strike <= strikes[0]:
        return float(ivs[0])
    if strike >= strikes[-1]:
        return float(ivs[-1])

    return float(np.interp(strike, strikes, ivs))


def get_interpolated_iv(
    ticker: str,
    spot: float,
    strike: float,
    requested_expiration: dt.date,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: Literal["call", "put"],
    available_expirations: list[str],
    valuation_date: Optional[dt.date] = None,
) -> IVInterpolationResult:
    """Full pipeline: find bracketing listed expirations, build smiles for each,
    interpolate each smile to the requested strike, then interpolate across
    time-to-expiry between the two (or use the single nearest expiry if the
    request falls outside the listed range or matches one exactly).

    This implements the spec's requirement that IV depend on BOTH the strike
    AND the listed expirations closest to the requested expiration, via a
    bilinear-style interpolation: strike interpolation within each expiry's
    smile, then linear interpolation in time-to-expiry across the two smiles.
    """
    valuation_date = valuation_date or dt.date.today()
    exp_dates = sorted(
        dt.datetime.strptime(e, "%Y-%m-%d").date() for e in available_expirations
    )

    if not exp_dates:
        raise NoOptionDataError(f"No listed expirations available for '{ticker}'.")

    # Locate the bracketing pair (near, far) around the requested date.
    near_date, far_date = _bracket_expirations(exp_dates, requested_expiration)

    near_str = near_date.strftime("%Y-%m-%d")
    near_smile = build_iv_smile(
        ticker, near_str, spot, risk_free_rate, dividend_yield, option_type, valuation_date
    )
    near_iv = interpolate_iv_for_strike(near_smile, strike)

    if far_date is None or far_date == near_date:
        return IVInterpolationResult(
            iv=near_iv,
            method="Single nearest listed expiration (strike-interpolated smile)",
            near_expiry=near_str,
            far_expiry=None,
            near_iv=near_iv,
            far_iv=None,
            weight_far=0.0,
        )

    far_str = far_date.strftime("%Y-%m-%d")
    far_smile = build_iv_smile(
        ticker, far_str, spot, risk_free_rate, dividend_yield, option_type, valuation_date
    )
    far_iv = interpolate_iv_for_strike(far_smile, strike)

    # Linear interpolation across calendar time between the two expiries,
    # evaluated at the requested expiration date (variance-time interpolation
    # would be more "correct" for term-structure work, but linear-in-time on
    # vol is the standard, numerically stable approach for this use case and
    # behaves well close to either anchor).
    total_days = (far_date - near_date).days
    target_days = (requested_expiration - near_date).days
    weight_far = 0.0 if total_days == 0 else np.clip(target_days / total_days, 0.0, 1.0)

    blended_iv = (1 - weight_far) * near_iv + weight_far * far_iv

    return IVInterpolationResult(
        iv=float(blended_iv),
        method="Strike-interpolated smile, time-interpolated between nearest two expirations",
        near_expiry=near_str,
        far_expiry=far_str,
        near_iv=near_iv,
        far_iv=far_iv,
        weight_far=float(weight_far),
    )


def _bracket_expirations(
    exp_dates: list[dt.date], target: dt.date
) -> tuple[dt.date, Optional[dt.date]]:
    """Find the listed expiration(s) that bracket the target date.

    - If target is before the first listed expiry -> (first, second-or-None).
    - If target is after the last listed expiry -> (last, second-to-last-or-None),
      effectively flat-extrapolating using the two longest-dated expiries.
    - If target falls between two listed expiries -> (lower, upper).
    - If target matches a listed expiry exactly -> (that_date, None).
    """
    if target in exp_dates:
        return target, None

    if target < exp_dates[0]:
        return exp_dates[0], (exp_dates[1] if len(exp_dates) > 1 else None)

    if target > exp_dates[-1]:
        return (
            (exp_dates[-2], exp_dates[-1]) if len(exp_dates) > 1 else (exp_dates[-1], None)
        )

    for i in range(len(exp_dates) - 1):
        if exp_dates[i] < target < exp_dates[i + 1]:
            return exp_dates[i], exp_dates[i + 1]

    # Should be unreachable given the checks above.
    return exp_dates[0], None


# --------------------------------------------------------------------------- #
# 5. Top-level orchestration
# --------------------------------------------------------------------------- #


def price_option(
    ticker: str,
    spot: float,
    dividend_yield: float,
    strike: float,
    expiration: dt.date,
    option_type: Literal["call", "put"],
    available_expirations: list[str],
    risk_free_rate: Optional[float] = None,
    use_live_rate: bool = True,
    valuation_date: Optional[dt.date] = None,
) -> PricingResult:
    """End-to-end: validate inputs, resolve risk-free rate, interpolate IV,
    price via BSM, compute greeks.

    Parameters
    ----------
    risk_free_rate : float, optional
        Manual annualized rate (as a fraction, e.g. 0.04). Required if
        `use_live_rate` is False; ignored (informationally) if True.
    use_live_rate : bool
        If True (default), fetches the live Treasury yield curve and
        interpolates it to this option's exact time-to-expiry, overriding
        `risk_free_rate`. If True but the live curve cannot be fetched,
        raises `RiskFreeRateError` rather than silently falling back, so the
        caller/UI can decide how to handle it (e.g. prompt for a manual rate).

    Raises
    ------
    InvalidStrikeError, InvalidExpirationError, ImpliedVolatilityError,
    NoOptionDataError, NetworkError, RiskFreeRateError
    """
    valuation_date = valuation_date or dt.date.today()

    if strike is None or strike <= 0:
        raise InvalidStrikeError("Strike price must be a positive number.")
    if strike > spot * 10 or strike < spot * 0.1:
        raise InvalidStrikeError(
            f"Strike price {strike:,.2f} is implausibly far from the current "
            f"spot price ({spot:,.2f}). Please double-check the value."
        )
    if expiration <= valuation_date:
        raise InvalidExpirationError(
            f"Expiration date {expiration.isoformat()} is not in the future. "
            "Please choose an expiration after today."
        )

    time_to_expiry = (expiration - valuation_date).days / 365.0

    if use_live_rate:
        rf_result = get_live_risk_free_rate(time_to_expiry)
    else:
        if risk_free_rate is None:
            raise RiskFreeRateError(
                "A manual risk-free rate is required when live rates are disabled."
            )
        rf_result = RiskFreeRateResult(
            rate=risk_free_rate, source="manual",
            method="User-supplied flat rate (live Treasury curve disabled).",
        )

    rfr = rf_result.rate

    iv_result = get_interpolated_iv(
        ticker=ticker,
        spot=spot,
        strike=strike,
        requested_expiration=expiration,
        risk_free_rate=rfr,
        dividend_yield=dividend_yield,
        option_type=option_type,
        available_expirations=available_expirations,
        valuation_date=valuation_date,
    )

    price, d1, d2 = bsm_price(
        spot, strike, time_to_expiry, rfr, dividend_yield, iv_result.iv, option_type
    )
    greeks = bsm_greeks(
        spot, strike, time_to_expiry, rfr, dividend_yield, iv_result.iv,
        option_type, d1, d2,
    )

    return PricingResult(
        spot_price=spot,
        strike=strike,
        option_type=option_type,
        time_to_expiry_years=time_to_expiry,
        risk_free_rate=rfr,
        dividend_yield=dividend_yield,
        iv_result=iv_result,
        rf_result=rf_result,
        option_price=price,
        d1=d1,
        d2=d2,
        greeks=greeks,
    )

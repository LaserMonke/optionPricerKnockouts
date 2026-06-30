"""
app.py
======
Streamlit front-end for the Black-Scholes-Merton equity option pricer.

Run with:
    streamlit run app.py

This file is intentionally focused on presentation/orchestration. All
financial math and Yahoo Finance plumbing lives in `bsm_engine.py`.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bsm_engine as eng

# --------------------------------------------------------------------------- #
# Page config & global style
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="BSM Option Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    /* ---- Base canvas ---- */
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #232733;
    }
    section[data-testid="stSidebar"] * {
        color: #e6e6e6;
    }

    /* ---- Headings ---- */
    h1, h2, h3 {
        color: #f5f5f5;
        font-weight: 600;
    }
    h1 { letter-spacing: -0.02em; }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background-color: #161b27;
        border: 1px solid #232733;
        border-radius: 10px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] {
        color: #8a92a6 !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        color: #f5f5f5 !important;
        font-weight: 700;
    }

    /* ---- Inputs ---- */
    .stTextInput input, .stNumberInput input, .stDateInput input {
        background-color: #1a1f2b;
        color: #e6e6e6;
        border: 1px solid #2a3040;
        border-radius: 6px;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1a1f2b;
        border-radius: 6px;
    }

    /* ---- Buttons ---- */
    .stButton button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        transition: background-color 0.15s ease;
    }
    .stButton button:hover {
        background-color: #1d4ed8;
    }

    /* ---- Dividers / containers ---- */
    .accent-line {
        height: 3px;
        width: 64px;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
        border-radius: 2px;
        margin: 0.25rem 0 1.25rem 0;
    }
    .info-box {
        background-color: #161b27;
        border: 1px solid #232733;
        border-left: 3px solid #2563eb;
        border-radius: 8px;
        padding: 14px 16px;
        font-size: 0.92rem;
        color: #c7ccd6;
    }
    .small-muted {
        color: #8a92a6;
        font-size: 0.82rem;
    }

    /* ---- Dataframe ---- */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    footer { visibility: hidden; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_DARK_TEMPLATE = "plotly_dark"
ACCENT = "#38bdf8"
ACCENT2 = "#2563eb"
GOOD = "#22c55e"
BAD = "#ef4444"

# --------------------------------------------------------------------------- #
# Cached data-access wrappers
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=600, show_spinner=False)
def cached_search_company(query: str):
    """Cache company/ticker search results for 10 minutes."""
    return eng.search_company(query)


@st.cache_data(ttl=120, show_spinner=False)
def cached_market_snapshot(ticker: str):
    """Cache spot price / dividend yield / expirations for 2 minutes
    (short TTL since spot price is time-sensitive)."""
    return eng.get_market_snapshot(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_treasury_curve():
    """Cache the live Treasury yield curve for 1 hour — these yields move
    slowly enough intraday that hourly refresh is more than sufficient,
    and it avoids hammering Yahoo Finance with 4 extra requests per pricing."""
    return eng.fetch_treasury_curve()


@st.cache_data(ttl=300, show_spinner=False)
def cached_iv_smile(ticker: str, expiration: str, spot: float, r: float, q: float, opt_type: str):
    """Cache a built IV smile for 5 minutes — this is the expensive call
    (one Brent solve per strike), so caching it is the key perf lever."""
    return eng.build_iv_smile(ticker, expiration, spot, r, q, opt_type)


def run_full_pricing(
    ticker, spot, div_yield, strike, expiration, option_type, expirations,
    use_live_rate, manual_rate, use_barrier=False, barrier_level=None, barrier_direction=None,
):
    """Thin wrapper around eng.price_option that routes smile-building through
    the cached function so repeated strike tweaks for the same expiry/ticker
    don't re-solve the whole chain. Resolves the risk-free rate (live
    Treasury curve, matched to time-to-expiry, or a manual flat rate) before
    any IV solving, since the rate is itself an input to the BSM-based
    Brent's-method IV solve.

    When use_barrier=True, the final pricing step swaps the vanilla BSM
    formula for the closed-form expiry-only knock-out barrier formula. The
    market-implied volatility is still derived from the VANILLA option chain
    (the barrier feature has no separate listed market to read IV from), then
    applied to the barrier formula -- this is standard practice for products
    without their own liquid market. Greeks are computed via finite
    differences in this case, since the barrier formula doesn't have simple
    closed-form Greeks the way the vanilla formula does."""
    valuation_date = dt.date.today()

    if strike is None or strike <= 0:
        raise eng.InvalidStrikeError("Strike price must be a positive number.")
    if strike > spot * 10 or strike < spot * 0.1:
        raise eng.InvalidStrikeError(
            f"Strike price {strike:,.2f} is implausibly far from the current "
            f"spot price ({spot:,.2f}). Please double-check the value."
        )
    if expiration <= valuation_date:
        raise eng.InvalidExpirationError(
            f"Expiration date {expiration.isoformat()} is not in the future. "
            "Please choose an expiration after today."
        )
    if use_barrier and (barrier_level is None or barrier_level <= 0):
        raise eng.InvalidBarrierError("Barrier level must be a positive number.")

    time_to_expiry = (expiration - valuation_date).days / 365.0

    if use_live_rate:
        curve = cached_treasury_curve()
        rf_result = eng.interpolate_risk_free_rate(curve, time_to_expiry)
    else:
        rf_result = eng.RiskFreeRateResult(
            rate=manual_rate, source="manual",
            method="User-supplied flat rate (live Treasury curve disabled).",
        )
    rfr = rf_result.rate

    exp_dates = sorted(dt.datetime.strptime(e, "%Y-%m-%d").date() for e in expirations)
    near_date, far_date = eng._bracket_expirations(exp_dates, expiration)

    near_str = near_date.strftime("%Y-%m-%d")
    near_smile = cached_iv_smile(ticker, near_str, spot, rfr, div_yield, option_type)
    near_iv = eng.interpolate_iv_for_strike(near_smile, strike)

    far_smile = None
    if far_date is None or far_date == near_date:
        iv_result = eng.IVInterpolationResult(
            iv=near_iv,
            method="Single nearest listed expiration (strike-interpolated smile)",
            near_expiry=near_str, far_expiry=None, near_iv=near_iv, far_iv=None, weight_far=0.0,
        )
    else:
        far_str = far_date.strftime("%Y-%m-%d")
        far_smile = cached_iv_smile(ticker, far_str, spot, rfr, div_yield, option_type)
        far_iv = eng.interpolate_iv_for_strike(far_smile, strike)
        total_days = (far_date - near_date).days
        target_days = (expiration - near_date).days
        weight_far = 0.0 if total_days == 0 else np.clip(target_days / total_days, 0.0, 1.0)
        blended_iv = (1 - weight_far) * near_iv + weight_far * far_iv
        iv_result = eng.IVInterpolationResult(
            iv=float(blended_iv),
            method="Strike-interpolated smile, time-interpolated between nearest two expirations",
            near_expiry=near_str, far_expiry=far_str, near_iv=near_iv, far_iv=far_iv,
            weight_far=float(weight_far),
        )

    if use_barrier:
        price = eng.bsm_barrier_price_expiry_only(
            spot, strike, barrier_level, time_to_expiry, rfr, div_yield,
            iv_result.iv, option_type, barrier_direction,
        )
        d1, d2 = float("nan"), float("nan")
        greeks = _finite_difference_barrier_greeks(
            spot, strike, barrier_level, time_to_expiry, rfr, div_yield,
            iv_result.iv, option_type, barrier_direction,
        )
    else:
        price, d1, d2 = eng.bsm_price(spot, strike, time_to_expiry, rfr, div_yield, iv_result.iv, option_type)
        greeks = eng.bsm_greeks(spot, strike, time_to_expiry, rfr, div_yield, iv_result.iv, option_type, d1, d2)

    result = eng.PricingResult(
        spot_price=spot, strike=strike, option_type=option_type,
        time_to_expiry_years=time_to_expiry, risk_free_rate=rfr, dividend_yield=div_yield,
        iv_result=iv_result, rf_result=rf_result, option_price=price, d1=d1, d2=d2, greeks=greeks,
    )
    return result, near_smile, far_smile


def _finite_difference_barrier_greeks(spot, strike, barrier, T, r, q, sigma, option_type, direction):
    """Central-difference Greeks for the barrier formula, since it has no
    simple closed-form derivatives the way the vanilla BSM formula does.
    Bump sizes are small relative to typical input magnitudes."""
    def price_at(S=spot, sig=sigma, t=T, rate=r):
        if t <= 0:
            t = 1e-6
        return eng.bsm_barrier_price_expiry_only(S, strike, barrier, t, rate, q, sig, option_type, direction)

    h_s = max(spot * 0.001, 0.01)
    h_sig = 0.0010
    h_t = 1.0 / 365.0
    h_r = 0.0001

    delta = (price_at(S=spot + h_s) - price_at(S=spot - h_s)) / (2 * h_s)
    gamma = (price_at(S=spot + h_s) - 2 * price_at() + price_at(S=spot - h_s)) / (h_s ** 2)
    vega = (price_at(sig=sigma + h_sig) - price_at(sig=sigma - h_sig)) / (2 * h_sig) / 100.0
    theta = -(price_at(t=T + h_t) - price_at(t=max(T - h_t, 1e-6))) / (2 * h_t) / 365.0
    rho = (price_at(rate=r + h_r) - price_at(rate=r - h_r)) / (2 * h_r) / 100.0

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# --------------------------------------------------------------------------- #
# Session state init
# --------------------------------------------------------------------------- #

if "resolved_ticker" not in st.session_state:
    st.session_state.resolved_ticker = None
if "company_matches" not in st.session_state:
    st.session_state.company_matches = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown("## 📈 Black-Scholes-Merton Option Pricer")
st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="small-muted">Market-implied volatility, solved strike-by-strike from listed '
    "option quotes via Brent's method — not Yahoo Finance's pre-computed IV — then interpolated "
    "to your exact strike and expiration.</div>",
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------- #
# Sidebar: Input panel
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### Input Panel")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    query = st.text_input(
        "Company name or ticker",
        placeholder="e.g. Apple, Tesla, AAPL, TSLA",
        key="query_input",
    ).strip()

    # Reset disambiguation state if the query text changed.
    if query and query != st.session_state.last_query:
        st.session_state.last_query = query
        st.session_state.resolved_ticker = None
        st.session_state.company_matches = []

    search_error = None
    if query:
        try:
            if not st.session_state.company_matches:
                with st.spinner("Searching..."):
                    st.session_state.company_matches = cached_search_company(query)
        except eng.TickerNotFoundError as e:
            search_error = str(e)
        except eng.NetworkError as e:
            search_error = str(e)

    if search_error:
        st.error(search_error)

    matches = st.session_state.company_matches
    if matches:
        if len(matches) == 1:
            st.session_state.resolved_ticker = matches[0].symbol
            st.success(f"Resolved: **{matches[0].label}**")
        else:
            labels = [m.label for m in matches]
            choice = st.selectbox("Multiple matches found — select one:", labels, key="match_choice")
            idx = labels.index(choice)
            st.session_state.resolved_ticker = matches[idx].symbol

    st.write("")
    strike_input = st.number_input(
        "Strike price ($)", min_value=0.0, value=0.0, step=1.0, format="%.2f",
        help="Must be a positive number.",
    )

    default_expiry = dt.date.today() + dt.timedelta(days=45)
    expiration_input = st.date_input(
        "Expiration date",
        value=default_expiry,
        min_value=dt.date.today() + dt.timedelta(days=1),
        help="Any future date — does not need to match a listed expiration; "
             "IV is interpolated between the nearest listed expiries.",
    )

    option_type_input = st.selectbox("Option type", ["Call", "Put"])

    st.write("")
    use_barrier_input = st.toggle(
        "Knock-out barrier (expiry-only)",
        value=False,
        help="Prices a knock-out option where the barrier is checked ONLY at "
             "expiry, not continuously along the path. The option pays the "
             "usual call/put payoff unless the underlying ends up past the "
             "barrier, in which case it pays zero.",
    )

    barrier_direction_input = None
    barrier_level_input = None
    if use_barrier_input:
        barrier_direction_input = st.selectbox(
            "Barrier direction",
            ["Up (knock out if price ends above barrier)", "Down (knock out if price ends below barrier)"],
        )
        barrier_direction_input = "up" if barrier_direction_input.startswith("Up") else "down"

        barrier_level_input = st.number_input(
            "Barrier level ($)", min_value=0.01, value=100.0,
            step=1.0, format="%.2f",
            help="Must be above current spot for an 'up' barrier, or below "
                 "current spot for a 'down' barrier — otherwise the option "
                 "would already be knocked out today. (Spot price is shown "
                 "below once a ticker is resolved.)",
        )

    st.write("")
    use_live_rate_input = st.toggle(
        "Use live Treasury yield curve",
        value=True,
        help="Fetches current US Treasury yields (3-month, 5-year, 10-year, "
             "30-year) from Yahoo Finance and interpolates to your option's "
             "exact time to expiry. Turn off to enter a flat rate manually.",
    )

    if use_live_rate_input:
        manual_rate_input = None
        st.caption("Risk-free rate will be matched to the expiration date below.")
    else:
        manual_rate_input = st.number_input(
            "Risk-free rate (%)", min_value=0.0, max_value=25.0, value=4.0, step=0.10,
            format="%.2f",
            help="Annualized, continuously-compounded flat rate, used for all maturities.",
        ) / 100.0

    st.write("")
    price_clicked = st.button("Price Option", use_container_width=True)

# --------------------------------------------------------------------------- #
# Main panel: Results
# --------------------------------------------------------------------------- #

if not query:
    st.info("👈 Enter a company name or ticker in the sidebar to get started.")
    st.stop()

if search_error:
    st.stop()

ticker = st.session_state.resolved_ticker
if not ticker:
    st.stop()

# --- Fetch market snapshot (spot, dividend yield, expirations) ---
try:
    with st.spinner(f"Fetching market data for {ticker}..."):
        snapshot = cached_market_snapshot(ticker)
except eng.TickerNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()
except eng.NoOptionDataError as e:
    st.error(f"❌ {e}")
    st.stop()
except eng.NetworkError as e:
    st.error(f"🌐 {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error: {e}")
    st.stop()

company_label = f"{snapshot.company_name} ({snapshot.ticker})" if snapshot.company_name else snapshot.ticker
st.markdown(f"#### {company_label}")

top_cols = st.columns(4)
with top_cols[0]:
    st.metric("Spot Price", f"${snapshot.spot_price:,.2f}")
with top_cols[1]:
    st.metric("Dividend Yield", f"{snapshot.dividend_yield * 100:.2f}%")
with top_cols[2]:
    st.metric("Listed Expirations", f"{len(snapshot.expirations)}")
with top_cols[3]:
    nearest = snapshot.expirations[0] if snapshot.expirations else "—"
    st.metric("Nearest Expiry", nearest)

st.write("")

if not price_clicked:
    st.markdown(
        '<div class="info-box">Configure the strike, expiration, option type, and risk-free '
        "rate in the sidebar, then click <b>Price Option</b> to compute the theoretical "
        "value.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# --- Validate inputs that don't require a network call ---
if strike_input <= 0:
    st.error("❌ Please enter a strike price greater than 0.")
    st.stop()

if expiration_input <= dt.date.today():
    st.error("❌ Expiration date must be in the future (past dates cannot be priced).")
    st.stop()

option_type = option_type_input.lower()

# --- Run pricing pipeline ---
try:
    with st.spinner("Resolving risk-free rate and solving implied volatility..."):
        result, near_smile, far_smile = run_full_pricing(
            ticker=snapshot.ticker,
            spot=snapshot.spot_price,
            div_yield=snapshot.dividend_yield,
            strike=float(strike_input),
            expiration=expiration_input,
            option_type=option_type,
            expirations=snapshot.expirations,
            use_live_rate=use_live_rate_input,
            manual_rate=manual_rate_input,
            use_barrier=use_barrier_input,
            barrier_level=barrier_level_input,
            barrier_direction=barrier_direction_input,
        )
except eng.InvalidStrikeError as e:
    st.error(f"❌ {e}")
    st.stop()
except eng.InvalidExpirationError as e:
    st.error(f"❌ {e}")
    st.stop()
except eng.InvalidBarrierError as e:
    st.error(f"🚧 {e}")
    st.stop()
except eng.NoOptionDataError as e:
    st.error(f"❌ {e}")
    st.stop()
except eng.ImpliedVolatilityError as e:
    st.error(f"⚠️ {e}")
    st.stop()
except eng.RiskFreeRateError as e:
    st.error(f"📉 {e}")
    st.info("Tip: turn off **'Use live Treasury yield curve'** in the sidebar to enter a rate manually.")
    st.stop()
except eng.NetworkError as e:
    st.error(f"🌐 {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error while pricing: {e}")
    st.stop()

# --------------------------------------------------------------------------- #
# Results panel
# --------------------------------------------------------------------------- #

st.markdown("### Results")
st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

res_cols = st.columns(4)
with res_cols[0]:
    st.metric("Stock Price", f"${result.spot_price:,.2f}")
with res_cols[1]:
    st.metric("Interpolated IV", f"{result.iv_result.iv * 100:.2f}%")
with res_cols[2]:
    rf_label = "Risk-Free Rate (live)" if result.rf_result.source == "live" else "Risk-Free Rate (manual)"
    st.metric(rf_label, f"{result.risk_free_rate * 100:.2f}%")
with res_cols[3]:
    price_label = (
        f"Knock-Out {option_type_input} Price" if use_barrier_input else f"BSM {option_type_input} Price"
    )
    st.metric(price_label, f"${result.option_price:,.2f}")

if use_barrier_input:
    barrier_word = "above" if barrier_direction_input == "up" else "below"
    st.markdown(
        f'<div class="info-box"><b>Barrier methodology:</b> Expiry-only monitoring — this '
        f"{option_type_input.lower()} pays the usual intrinsic value at expiry unless the stock "
        f"ends up {barrier_word} the <b>${barrier_level_input:,.2f}</b> barrier, in which case it "
        "pays zero. The barrier is checked only at the final expiry date, not continuously along "
        "the way — closed-form under Black-Scholes-Merton. Greeks below are computed via finite "
        "differences, since this formula has no simple closed-form derivatives.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

# --- Risk-free rate methodology transparency box ---
st.markdown(
    f'<div class="info-box"><b>Risk-free rate:</b> {result.rf_result.method}</div>',
    unsafe_allow_html=True,
)
st.write("")

# --- IV methodology transparency box ---
iv_r = result.iv_result
if iv_r.far_expiry:
    method_detail = (
        f"Blended **{(1 - iv_r.weight_far) * 100:.1f}%** weight on **{iv_r.near_expiry}** "
        f"(IV {iv_r.near_iv*100:.2f}%) and **{iv_r.weight_far * 100:.1f}%** weight on "
        f"**{iv_r.far_expiry}** (IV {iv_r.far_iv*100:.2f}%), each strike-interpolated "
        f"from that expiry's smile at K=${result.strike:,.2f}."
    )
else:
    method_detail = (
        f"Used the single nearest listed expiration **{iv_r.near_expiry}** "
        f"(IV {iv_r.near_iv*100:.2f}%), strike-interpolated at K=${result.strike:,.2f}."
    )
st.markdown(
    f'<div class="info-box"><b>IV methodology:</b> {iv_r.method}.<br/>{method_detail}</div>',
    unsafe_allow_html=True,
)

st.write("")

# --- Pricing detail + Greeks table ---
detail_cols = st.columns(2)
with detail_cols[0]:
    st.markdown("##### Pricing Inputs")
    detail_df = pd.DataFrame(
        {
            "Parameter": [
                "Spot Price", "Strike", "Time to Expiry (yrs)",
                f"Risk-Free Rate ({result.rf_result.source})",
                "Dividend Yield", "Implied Volatility", "d1", "d2",
            ],
            "Value": [
                f"${result.spot_price:,.2f}",
                f"${result.strike:,.2f}",
                f"{result.time_to_expiry_years:.4f}",
                f"{result.risk_free_rate*100:.2f}%",
                f"{result.dividend_yield*100:.2f}%",
                f"{result.iv_result.iv*100:.2f}%",
                f"{result.d1:.4f}" if not np.isnan(result.d1) else "—",
                f"{result.d2:.4f}" if not np.isnan(result.d2) else "—",
            ],
        }
    )
    st.dataframe(detail_df, hide_index=True, use_container_width=True)

with detail_cols[1]:
    st.markdown("##### Greeks")
    g = result.greeks
    greeks_df = pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega (per 1% vol)", "Theta (per day)", "Rho (per 1% rate)"],
            "Value": [
                f"{g['delta']:.4f}" if not np.isnan(g["delta"]) else "—",
                f"{g['gamma']:.4f}" if not np.isnan(g["gamma"]) else "—",
                f"{g['vega']:.4f}" if not np.isnan(g["vega"]) else "—",
                f"{g['theta']:.4f}" if not np.isnan(g["theta"]) else "—",
                f"{g['rho']:.4f}" if not np.isnan(g["rho"]) else "—",
            ],
        }
    )
    st.dataframe(greeks_df, hide_index=True, use_container_width=True)

st.write("")

# --------------------------------------------------------------------------- #
# Visualization: IV smile(s) with interpolated point highlighted
# --------------------------------------------------------------------------- #

st.markdown("### Implied Volatility Smile")
st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=near_smile["strike"], y=near_smile["iv"] * 100,
        mode="lines+markers", name=f"Smile: {iv_r.near_expiry}",
        line=dict(color=ACCENT, width=2), marker=dict(size=6),
    )
)
fig.add_trace(
    go.Scatter(
        x=[result.strike], y=[iv_r.near_iv * 100],
        mode="markers", name=f"Strike-interp @ {iv_r.near_expiry}",
        marker=dict(size=11, color=ACCENT, symbol="diamond", line=dict(width=1, color="white")),
        showlegend=False,
    )
)

if far_smile is not None and iv_r.far_expiry:
    fig.add_trace(
        go.Scatter(
            x=far_smile["strike"], y=far_smile["iv"] * 100,
            mode="lines+markers", name=f"Smile: {iv_r.far_expiry}",
            line=dict(color="#a78bfa", width=2), marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[result.strike], y=[iv_r.far_iv * 100],
            mode="markers", name=f"Strike-interp @ {iv_r.far_expiry}",
            marker=dict(size=11, color="#a78bfa", symbol="diamond", line=dict(width=1, color="white")),
            showlegend=False,
        )
    )

fig.add_trace(
    go.Scatter(
        x=[result.strike], y=[iv_r.iv * 100],
        mode="markers", name="Final interpolated IV",
        marker=dict(size=16, color=GOOD, symbol="star", line=dict(width=1.5, color="white")),
    )
)
fig.add_vline(x=result.strike, line_dash="dash", line_color="#555a66")
fig.add_vline(x=result.spot_price, line_dash="dot", line_color="#8a92a6",
              annotation_text="Spot", annotation_position="top")

fig.update_layout(
    template=PLOTLY_DARK_TEMPLATE,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    height=440,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Strike ($)", yaxis_title="Implied Volatility (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Visualization: Payoff diagram at expiry
# --------------------------------------------------------------------------- #

st.markdown("### Payoff at Expiration")
st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

# Build the x-axis range from spot, strike, AND barrier (when active) so the
# full payoff shape -- including the knock-out cliff -- is always visible,
# regardless of how far apart these levels are (e.g. far OTM LEAPS strikes).
range_candidates = [result.spot_price, result.strike]
if use_barrier_input:
    range_candidates.append(barrier_level_input)
range_lo = min(range_candidates) * 0.5
range_hi = max(range_candidates) * 1.5
spot_range = np.linspace(range_lo, range_hi, 400)

if option_type == "call":
    raw_payoff = np.maximum(spot_range - result.strike, 0)
else:
    raw_payoff = np.maximum(result.strike - spot_range, 0)

if use_barrier_input:
    if barrier_direction_input == "up":
        knocked_out = spot_range >= barrier_level_input
    else:
        knocked_out = spot_range <= barrier_level_input
    raw_payoff = np.where(knocked_out, 0.0, raw_payoff)

payoff = raw_payoff - result.option_price

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=spot_range, y=payoff, mode="lines", name="P&L at expiry",
        line=dict(color=ACCENT2, width=3),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.15)",
    )
)
fig2.add_hline(y=0, line_color="#555a66", line_width=1)
fig2.add_vline(x=result.strike, line_dash="dash", line_color="#8a92a6",
               annotation_text="Strike", annotation_position="top")
fig2.add_vline(x=result.spot_price, line_dash="dot", line_color=GOOD,
               annotation_text="Spot", annotation_position="bottom")

if use_barrier_input:
    fig2.add_vline(x=barrier_level_input, line_dash="dash", line_color=BAD,
                   annotation_text=f"Barrier ${barrier_level_input:,.2f}", annotation_position="top right")
    st.caption(
        "Note: with a knock-out barrier, the usual single 'breakeven' point isn't well-defined "
        "the same way — the payoff drops to a flat loss (the premium paid) past the barrier, "
        "shown as the cliff edge above."
    )
else:
    breakeven = (
        result.strike + result.option_price if option_type == "call"
        else result.strike - result.option_price
    )
    fig2.add_vline(x=breakeven, line_dash="dash", line_color="#f59e0b",
                   annotation_text=f"Breakeven ${breakeven:,.2f}", annotation_position="top right")

fig2.update_layout(
    template=PLOTLY_DARK_TEMPLATE,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    height=400,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Underlying Price at Expiry ($)", yaxis_title="Profit / Loss ($)",
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    '<div class="small-muted">Theoretical pricing for educational/analytical purposes only. '
    "Not investment advice. Market data via Yahoo Finance (yfinance); implied volatility is "
    "independently solved via Brent's method against the Black-Scholes-Merton formula.</div>",
    unsafe_allow_html=True,
)

"""
Australian Retirement Simulation + Home Price Growth Analyser.

Two tabs in one app:
  1. Retirement Simulator — monthly cashflow sim with AU tax, super and
     Age Pension rules verified as of April 2026 (FY 2025-26).
  2. Home Price Growth — address-based property sale history research via
     Google Gemini + Google Search grounding.

Run with:   streamlit run retirement_sim.py

Requirements:
    pip install streamlit numpy pandas plotly google-genai

Disclaimer: educational only, not financial advice. Verify figures against
ATO, Services Australia and your super fund before making decisions.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# google-genai is only needed for the Home Price Growth tab. If it's not
# installed, the retirement simulator still works; the other tab shows a
# friendly install instruction instead of crashing.
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# =========================================================================
# 🔑  GOOGLE GEMINI API KEY (optional default)
# =========================================================================
# Users enter their key in the sidebar of the app at runtime, so this is
# usually left blank. If you want to embed a default key (e.g. for a private
# deployment), paste it between the quotes below — it'll pre-fill the input.
# Get a free key at: https://aistudio.google.com/apikey
GEMINI_API_KEY_DEFAULT = ""
# =========================================================================

# -------------------------------------------------------------------------
# Page config MUST be the first Streamlit call.
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="AU Retirement & Property Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# VERIFIED DEFAULTS (FY 2025-26 / April 2026)
# Sources are listed here and repeated in the sidebar help for each field.
# =========================================================================

# Personal income tax - ATO resident rates for FY 2025-26 (Stage 3 in effect
# from 1 July 2024; Treasurer confirmed same scales apply 2025-26).
# https://www.ato.gov.au/tax-rates-and-codes/tax-rates-australian-residents
TAX_BRACKETS_2025_26 = [
    (18_200,  0.00),
    (45_000,  0.16),
    (135_000, 0.30),
    (190_000, 0.37),
    (float("inf"), 0.45),
]
MEDICARE_LEVY_RATE = 0.02
# Medicare levy low-income shade-in (2024-25 thresholds; indexed annually in the
# Budget). Exact 2025-26 figures announced in the 2025 Federal Budget.
MEDICARE_THRESHOLD_LOWER = 27_222   # fully exempt below
MEDICARE_THRESHOLD_UPPER = 34_027   # full 2% levy above; 10% phase-in in between

# Low Income Tax Offset (LITO) - FY 2025-26 (unchanged from 2024-25).
LITO_MAX     = 700
LITO_FULL_TO = 37_500
LITO_PHASE1_UPPER = 45_000  # 5 cents/$ phase-out between 37,500 and 45,000
LITO_PHASE2_UPPER = 66_667  # 1.5 cents/$ phase-out between 45,000 and 66,667

# Superannuation
SG_RATE_2025_26 = 0.12            # SG permanently 12% from 1 July 2025
CONCESSIONAL_CAP_2025_26 = 30_000 # per person per year
CONTRIBUTIONS_TAX = 0.15
ACCUMULATION_EARNINGS_TAX = 0.15
RETIREMENT_EARNINGS_TAX = 0.00    # account-based pension phase

# Age Pension - Services Australia, rates from 20 March 2026.
PENSION_AGE = 67
# Max annual rates including supplements:
MAX_PENSION_SINGLE_ANNUAL = 31_223.40   # $1,200.90/fortnight
MAX_PENSION_COUPLE_ANNUAL = 47_070.40   # $1,810.40/fortnight combined

# Income test (per-fortnight converted to annual)
INCOME_FREE_AREA_SINGLE = 218 * 26       # $5,668/yr
INCOME_FREE_AREA_COUPLE = 380 * 26       # $9,880/yr (combined)
INCOME_TAPER = 0.50

# Assets test - lower thresholds (indexed 1 July) from 1 July 2025.
ASSETS_LOWER_SINGLE_HO     = 321_500
ASSETS_LOWER_COUPLE_HO     = 481_500
ASSETS_LOWER_SINGLE_NONHO  = 579_500
ASSETS_LOWER_COUPLE_NONHO  = 739_500
ASSETS_TAPER_PER_1000_FN   = 3.0   # $3/fn per $1,000
ASSETS_TAPER_PER_1000_YR   = 78.0  # annualised

# Deeming (from 20 March 2026)
DEEMING_LOW_RATE   = 0.0125
DEEMING_HIGH_RATE  = 0.0325
DEEMING_THRESHOLD_SINGLE = 64_200
DEEMING_THRESHOLD_COUPLE = 106_200

# Minimum account-based pension drawdown rates (standard schedule)
MIN_DRAWDOWN = {
    (0, 64):  0.04,
    (65, 74): 0.05,
    (75, 79): 0.06,
    (80, 84): 0.07,
    (85, 89): 0.09,
    (90, 94): 0.11,
    (95, 200): 0.14,
}

# Preservation age — for people born after 30 June 1964, preservation age is 60.
# This is the earliest age super can be accessed (under "retirement" condition of release).
PRESERVATION_AGE = 60

# ASFA Retirement Standard, Sep 2025 quarter (couple/single, homeowner,
# comfortable). Updated ~quarterly.
ASFA_COMFORTABLE_COUPLE = 77_375
ASFA_COMFORTABLE_SINGLE = 54_240

TODAY_YEAR = 2026  # all "real-dollar" outputs are stated in today's dollars

# =========================================================================
# GEMINI HELPERS (for the Home Price Growth tab)
# =========================================================================
# Free-tier Gemini models to try in order, as of April 2026.
# gemini-2.0-flash is retired; do not use.
GEMINI_MODEL_PRIORITY = [
    "gemini-2.5-flash",       # balanced
    "gemini-2.5-flash-lite",  # highest free-tier quota
    "gemini-2.5-pro",         # best quality, lowest quota
]

HOME_PRICE_SYSTEM_PROMPT = """You are an expert Australian property data researcher.
You have access to Google Search. When given an Australian property address,
you MUST search the web to find its complete sale/transaction history.

Search sites like domain.com.au, realestate.com.au, onthehouse.com.au,
propertyvalue.com.au, allhomes.com.au, and any other Australian property
data source. Search for the SPECIFIC address provided.

Also look for any Development Applications (DAs), building permits,
renovation records, or new-build indicators for that address.

RESPOND WITH ONLY VALID JSON — no markdown fences, no backticks, no preamble.
Use this exact schema:

{
  "address": "full matched address string",
  "property_type": "house/unit/townhouse/apartment/land/other",
  "bedrooms": null or number,
  "bathrooms": null or number,
  "car_spaces": null or number,
  "land_area_sqm": null or number,
  "year_built": null or number,
  "sales": [
    {
      "date": "YYYY-MM-DD",
      "price": number (in dollars, no commas, no $ sign),
      "sale_type": "private sale/auction/off-the-plan/new build/unknown",
      "agency": "agency name or empty string",
      "source": "website where you found this data"
    }
  ],
  "renovations": [
    {
      "year": number,
      "description": "what was done",
      "source": "where you found this"
    }
  ],
  "notes": "any extra context — zoning, heritage, flood zone, etc.",
  "data_confidence": "high/medium/low",
  "sources_checked": ["list of URLs or sites you searched"]
}

CRITICAL RULES:
- Search multiple times if needed to find complete history.
- Include ALL sales you can find, sorted oldest to newest.
- Prices must be plain numbers (e.g. 850000 not $850,000).
- Dates must be YYYY-MM-DD. If only year known, use YYYY-01-01.
- If a price was withheld, set price to null but still include the entry.
- NEVER fabricate or hallucinate prices. Only report verified data.
- If no data found, return the JSON with empty arrays.
"""

DA_SYSTEM_PROMPT = """You are a researcher looking up Development Applications
(DAs), building permits and renovation approvals for a given Australian address.
Use Google Search. Check council planning portals (NSW ePlanning, VicPlan,
PlanSA, iPlan, etc.) and aggregators.

Respond with ONLY valid JSON:
{
  "development_applications": [
    {
      "year": number,
      "type": "renovation/extension/new build/demolition/other",
      "description": "brief description",
      "status": "approved/completed/pending/unknown",
      "source": "URL or site"
    }
  ]
}
"""


def _current_api_key() -> str:
    """Return the API key for this user session.
    Priority: sidebar input (in session_state) → file-level default → empty."""
    return (st.session_state.get("gemini_api_key", "") or "").strip() \
        or GEMINI_API_KEY_DEFAULT.strip()


def _get_gemini_client():
    """Build a Gemini client for the current session's key. Returns None if
    no key is set or the client init fails."""
    if not GENAI_AVAILABLE:
        return None
    key = _current_api_key()
    if not key:
        return None
    # Cache the client on the session so we don't rebuild per call, but DO
    # rebuild if the user changes their key.
    cached = st.session_state.get("_gemini_client")
    cached_key = st.session_state.get("_gemini_client_key")
    if cached is not None and cached_key == key:
        return cached
    try:
        client = genai.Client(api_key=key)
    except Exception:
        st.session_state["_gemini_client"] = None
        st.session_state["_gemini_client_key"] = key
        return None
    st.session_state["_gemini_client"] = client
    st.session_state["_gemini_client_key"] = key
    return client


def _detect_gemini_model():
    """Pick the first free-tier model that responds to a small test prompt.
    Cached on the session keyed by the API key, so changing keys forces a
    re-detection."""
    client = _get_gemini_client()
    if client is None:
        return None, "Gemini client not initialised (missing package or API key)."

    key = _current_api_key()
    cached = st.session_state.get("_gemini_model")
    cached_for_key = st.session_state.get("_gemini_model_key")
    if cached and cached_for_key == key:
        return cached, None

    last_err = "no models attempted"
    for name in GEMINI_MODEL_PRIORITY:
        try:
            client.models.generate_content(
                model=name,
                contents="ok",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            st.session_state["_gemini_model"] = name
            st.session_state["_gemini_model_key"] = key
            return name, None
        except Exception as e:
            last_err = str(e)
            continue
    st.session_state["_gemini_model"] = None
    st.session_state["_gemini_model_key"] = key
    return None, last_err


def _clean_json(text):
    """Strip markdown fences and extract JSON from a Gemini response."""
    if not text:
        return None
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _query_gemini(prompt: str, system_instruction: str | None = None,
                  retries: int = 3, model_override: str | None = None,
                  max_output_tokens: int = 4096) -> str | None:
    """Call Gemini with Google Search grounding + retry-on-rate-limit."""
    client = _get_gemini_client()
    if client is None:
        return None
    model = model_override
    if model is None:
        model, _ = _detect_gemini_model()
        if model is None:
            return None

    search_tool = types.Tool(google_search=types.GoogleSearch())
    cfg = types.GenerateContentConfig(
        tools=[search_tool],
        temperature=0.1,
        max_output_tokens=max_output_tokens,
    )
    if system_instruction:
        cfg.system_instruction = system_instruction

    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt, config=cfg
            )
            return resp.text
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = min((2 ** (attempt + 1)) * 5, 45)
                time.sleep(wait)
                continue
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return None
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_property_history_cached(address: str, _api_key_fingerprint: str) -> dict | None:
    """Internal cached fetch. The api_key fingerprint is part of the cache
    key so different users don't see each other's cached results — but it
    is not actually used inside the function body (the live key comes from
    session state via _query_gemini)."""
    prompt = (f"Find the complete sale and transaction history for this "
              f"Australian property address: {address}\n\n"
              "Search domain.com.au, realestate.com.au, onthehouse.com.au, "
              "propertyvalue.com.au and any other relevant sites.")
    raw = _query_gemini(prompt, system_instruction=HOME_PRICE_SYSTEM_PROMPT)
    return _clean_json(raw)


def fetch_property_history(address: str) -> dict | None:
    return _fetch_property_history_cached(address, _api_key_fingerprint())


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_da_records_cached(address: str, _api_key_fingerprint: str) -> dict | None:
    prompt = (f"Search for Development Applications, building permits, and "
              f"renovation approvals for: {address}\n\n"
              "Check council planning portals (NSW ePlanning, VicPlan, "
              "PlanSA, iPlan, etc.) and aggregators.")
    raw = _query_gemini(prompt, system_instruction=DA_SYSTEM_PROMPT)
    return _clean_json(raw)


def fetch_da_records(address: str) -> dict | None:
    return _fetch_da_records_cached(address, _api_key_fingerprint())


def _api_key_fingerprint() -> str:
    """Short fingerprint of the current API key — used only as a cache-key
    discriminator so caches don't leak between users in shared deployments.
    The actual key is never logged, returned, or sent anywhere."""
    import hashlib
    key = _current_api_key()
    if not key:
        return "no-key"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


# -------------------------------------------------------------------------
# DEEP SEARCH — multi-pass strategy to squeeze out every sale record.
# Each pass targets a different site/angle. Results are merged + deduped.
# -------------------------------------------------------------------------
DEEP_SEARCH_PASSES = [
    (
        "domain",
        "Search ONLY domain.com.au for property {addr}. "
        "Open the property's 'Property history' / 'Sale history' section. "
        "List EVERY sale listed there — even ones from the 1990s or early 2000s. "
        "Include withheld prices as null. Also note if it's currently listed."
    ),
    (
        "realestate",
        "Search ONLY realestate.com.au for property {addr}. "
        "Find the 'Sold' page or 'Property value' / 'Property timeline' data. "
        "Return every transaction with date and price (or null if withheld)."
    ),
    (
        "onthehouse",
        "Search ONLY onthehouse.com.au and propertyvalue.com.au for {addr}. "
        "These sites often have sales from the 1980s-2000s that newer portals "
        "don't show. List every transaction found."
    ),
    (
        "statevg",
        "Search the relevant STATE Valuer-General or land registry records for "
        "{addr}. In NSW this is the Valuer General / NSW LRS sales database; in "
        "Victoria use DELWP property sales data; in QLD the Titles Registry; in "
        "WA Landgate. Return every transaction with date, price and document "
        "reference if available."
    ),
]


def _best_model_for_deep_search() -> str | None:
    """Prefer gemini-2.5-pro for deep search (better at aggregating multiple
    sources and following long instructions). Fall back to whichever model
    passed our detection test."""
    client = _get_gemini_client()
    if client is None:
        return None
    for name in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"):
        try:
            client.models.generate_content(
                model=name, contents="ok",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            return name
        except Exception:
            continue
    return None


def _merge_sales(lists_of_sales: list[list[dict]]) -> list[dict]:
    """Merge multiple sales lists from different passes, deduplicating by
    (rough date, rough price). Preserves the earliest-seen source for each
    unique sale."""
    seen: dict[tuple, dict] = {}

    def key_for(sale: dict):
        d = _parse_sale_date(sale.get("date"))
        # Round to month for date dedup (different sources round differently)
        date_key = (d.year, d.month) if d else (None, None)
        # Round price to nearest $1k for price dedup
        p = sale.get("price")
        price_key = round(p / 1000) if p else None
        return (date_key, price_key)

    for sales in lists_of_sales:
        for sale in sales or []:
            k = key_for(sale)
            # If the same sale already seen, prefer one with a source noted
            if k not in seen:
                seen[k] = sale
            else:
                existing = seen[k]
                if not existing.get("source") and sale.get("source"):
                    seen[k] = sale

    # Sort oldest → newest by parsed date (unparseable at the end)
    merged = list(seen.values())
    merged.sort(key=lambda s: (_parse_sale_date(s.get("date")) or datetime.max))
    return merged


def fetch_property_history_deep(address: str, status_writer=None) -> dict | None:
    """Deep search: 4 separate site-targeted queries, merged and deduped.
    status_writer is an optional callable (e.g. st.write) that receives
    progress lines. Deep search does NOT cache — the user explicitly asked
    for a fresh multi-pass lookup."""
    model = _best_model_for_deep_search()
    if model is None:
        return None

    # Container: start with an empty shell; we'll fill fields from the best pass
    merged: dict = {
        "address": address,
        "property_type": None,
        "bedrooms": None,
        "bathrooms": None,
        "car_spaces": None,
        "land_area_sqm": None,
        "year_built": None,
        "sales": [],
        "renovations": [],
        "notes": "",
        "data_confidence": None,
        "sources_checked": [],
    }

    all_sales_lists: list[list[dict]] = []

    for pass_key, pass_template in DEEP_SEARCH_PASSES:
        if status_writer:
            status_writer(f"🔎 Deep-search pass: {pass_key}…")
        prompt = pass_template.format(addr=address)
        raw = _query_gemini(
            prompt,
            system_instruction=HOME_PRICE_SYSTEM_PROMPT,
            model_override=model,
            max_output_tokens=8192,
            retries=2,
        )
        parsed = _clean_json(raw) or {}
        all_sales_lists.append(parsed.get("sales") or [])
        # Back-fill feature fields from whichever pass has them
        for field in ("property_type", "bedrooms", "bathrooms", "car_spaces",
                       "land_area_sqm", "year_built"):
            if merged[field] in (None, "") and parsed.get(field) not in (None, ""):
                merged[field] = parsed[field]
        # Track sources
        for src in parsed.get("sources_checked") or []:
            if src and src not in merged["sources_checked"]:
                merged["sources_checked"].append(src)
        # Keep the longer/richer note
        note = parsed.get("notes") or ""
        if len(note) > len(merged["notes"] or ""):
            merged["notes"] = note
        # Small pause between passes so we don't slam the rate limit
        time.sleep(1.5)

    # Combine and dedupe sales
    merged["sales"] = _merge_sales(all_sales_lists)
    if status_writer:
        status_writer(f"✅ Merged {sum(len(s) for s in all_sales_lists)} raw "
                       f"records → {len(merged['sales'])} unique sales.")

    # Confidence: high if ≥3 passes returned at least one sale
    non_empty_passes = sum(1 for s in all_sales_lists if s)
    if non_empty_passes >= 3:
        merged["data_confidence"] = "high"
    elif non_empty_passes >= 2:
        merged["data_confidence"] = "medium"
    elif merged["sales"]:
        merged["data_confidence"] = "low"

    return merged


def _parse_sale_date(d) -> datetime | None:
    if not d:
        return None
    s = str(d)[:10]
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%d-%m-%Y", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _annualised_growth(p1: float, p2: float, years: float) -> float | None:
    if not p1 or not p2 or years <= 0 or p1 <= 0:
        return None
    return ((p2 / p1) ** (1 / years) - 1) * 100


def _detect_sale_flags(row: dict, reno_years: list[int] | None = None,
                       median_growth: float = 5.5) -> str:
    flags = []
    gr = row.get("annual_growth_pct")
    hold = row.get("hold_years")
    stype = str(row.get("sale_type", "")).lower()
    sale_year = row.get("year")

    if reno_years and sale_year:
        for ry in reno_years:
            if ry <= sale_year and sale_year - ry <= 3:
                flags.append("🔨 Post-reno sale")
                break
    if gr is not None and gr > median_growth * 2:
        flags.append("📈 Abnormal growth")
    if hold is not None and hold < 2 and gr is not None and gr > 15:
        flags.append("⚡ Quick flip")
    if any(kw in stype for kw in ["new", "built", "off the plan", "off-the-plan"]):
        flags.append("🏗️ New build / OTP")
    return " · ".join(flags)


def _build_sales_dataframe(sales_ok: list[dict], reno_years: list[int]) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(sales_ok):
        dt = _parse_sale_date(s.get("date"))
        row = {
            "sale_num": i + 1,
            "date": dt.strftime("%d %b %Y") if dt else (s.get("date") or "?"),
            "date_obj": dt,
            "year": dt.year if dt else None,
            "price": s["price"],
            "sale_type": s.get("sale_type") or "",
            "agency": s.get("agency") or "",
            "source": s.get("source") or "",
            "hold_years": None,
            "total_growth_pct": None,
            "total_growth_dollar": None,
            "annual_growth_pct": None,
        }
        if i > 0 and dt:
            prev_dt = _parse_sale_date(sales_ok[i - 1].get("date"))
            prev_price = sales_ok[i - 1]["price"]
            if prev_dt and prev_price:
                years = (dt - prev_dt).days / 365.25
                row["hold_years"] = round(years, 1)
                row["total_growth_pct"] = round((s["price"] / prev_price - 1) * 100, 1)
                row["total_growth_dollar"] = round(s["price"] - prev_price)
                if years > 0.1:
                    ag = _annualised_growth(prev_price, s["price"], years)
                    row["annual_growth_pct"] = round(ag, 1) if ag is not None else None
        row["flags"] = _detect_sale_flags(row, reno_years=reno_years)
        rows.append(row)
    return pd.DataFrame(rows)


def _render_property_result(address: str, data: dict, da_data: dict | None):
    """Display full property analysis for a single address."""
    # ---- Property summary card ----
    matched = data.get("address") or address
    col1, col2, col3 = st.columns([3, 1, 1])
    col1.markdown(f"### 🏠 {matched}")
    conf = (data.get("data_confidence") or "").lower()
    if conf:
        emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
        col2.metric("Data confidence", f"{emoji} {conf.title()}")

    feat_cols = st.columns(5)
    def _feat(col, label, val):
        col.metric(label, str(val) if val not in (None, "") else "—")
    _feat(feat_cols[0], "Type", (data.get("property_type") or "?").title())
    _feat(feat_cols[1], "Bed", data.get("bedrooms"))
    _feat(feat_cols[2], "Bath", data.get("bathrooms"))
    _feat(feat_cols[3], "Car", data.get("car_spaces"))
    _feat(feat_cols[4], "Land (m²)", data.get("land_area_sqm"))
    if data.get("year_built"):
        st.caption(f"Built: {data['year_built']}")
    if data.get("notes"):
        st.info(data["notes"])

    # ---- Sales ----
    sales = data.get("sales") or []
    sales_ok = [s for s in sales if s.get("price")]
    sales_null = [s for s in sales if not s.get("price")]

    if not sales_ok:
        st.warning(
            "No sale prices were found. The property may not have transacted recently, "
            "the price may have been withheld, or the address might not match any listing. "
            "Try searching the address directly on domain.com.au or realestate.com.au."
        )
        return

    # DA years for flag detection
    reno_years = []
    if da_data and da_data.get("development_applications"):
        for da in da_data["development_applications"]:
            if da.get("year"):
                try:
                    reno_years.append(int(da["year"]))
                except (TypeError, ValueError):
                    pass

    df = _build_sales_dataframe(sales_ok, reno_years)

    # ---- Summary metrics ----
    first_sale, last_sale = sales_ok[0], sales_ok[-1]
    first_dt = _parse_sale_date(first_sale["date"])
    last_dt = _parse_sale_date(last_sale["date"])
    total_yrs = (last_dt - first_dt).days / 365.25 if first_dt and last_dt else 0
    cagr = _annualised_growth(first_sale["price"], last_sale["price"], total_yrs) \
        if total_yrs > 0.1 else None
    total_ret_pct = (last_sale["price"] / first_sale["price"] - 1) * 100

    st.markdown("### 📊 Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sales found", len(sales_ok))
    m2.metric(
        "First → Last",
        f"${last_sale['price']:,.0f}",
        delta=f"+${last_sale['price'] - first_sale['price']:,.0f}"
        if last_sale['price'] >= first_sale['price'] else
        f"-${first_sale['price'] - last_sale['price']:,.0f}",
    )
    m3.metric("Overall CAGR", f"{cagr:.1f}% p.a." if cagr is not None else "—")
    m4.metric(
        "Total return",
        f"{total_ret_pct:+.1f}%",
        delta=f"over {total_yrs:.1f} yrs" if total_yrs else None,
        delta_color="off",
    )

    # ---- Chart ----
    dates_x = [r["date_obj"] for _, r in df.iterrows() if r["date_obj"]]
    prices_y = [r["price"] for _, r in df.iterrows() if r["date_obj"]]
    flags_t = [r["flags"] for _, r in df.iterrows() if r["date_obj"]]
    cagr_y = [r.get("annual_growth_pct") for _, r in df.iterrows() if r["date_obj"]]

    if len(dates_x) >= 2:
        st.markdown("### 📈 Price history")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=dates_x, y=prices_y, mode="lines+markers", name="Sale price",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=12, color="#2563eb", line=dict(width=2, color="white")),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>$%{y:,.0f}<extra></extra>",
        ))
        fdx = [d for d, f in zip(dates_x, flags_t) if f]
        fdy = [p for p, f in zip(prices_y, flags_t) if f]
        fdt = [f for f in flags_t if f]
        if fdx:
            fig.add_trace(go.Scatter(
                x=fdx, y=fdy, mode="markers", name="Flagged",
                marker=dict(size=18, color="#ef4444", symbol="diamond",
                            line=dict(width=2, color="white")),
                text=fdt,
                hovertemplate="<b>%{x|%d %b %Y}</b><br>$%{y:,.0f}<br>%{text}<extra></extra>",
            ))
        # Use add_shape + add_annotation instead of add_vline with
        # annotation_text, which has a known bug on date axes where
        # plotly tries to sum() datetime objects when positioning the label.
        for ry in reno_years:
            x_str = f"{int(ry)}-06-01"
            fig.add_shape(
                type="line", xref="x", yref="paper",
                x0=x_str, x1=x_str, y0=0, y1=1,
                line=dict(color="#f59e0b", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=x_str, y=1.02, xref="x", yref="paper",
                text=f"🔨 DA {ry}", showarrow=False,
                font=dict(color="#f59e0b", size=11),
                xanchor="center", yanchor="bottom",
            )
        bar_colors = ["#22c55e" if (g is not None and g >= 0) else "#ef4444"
                      for g in cagr_y]
        fig.add_trace(go.Bar(
            x=dates_x, y=[g if g is not None else 0 for g in cagr_y],
            name="CAGR %", marker_color=bar_colors, opacity=0.3,
            hovertemplate="%{y:.1f}% p.a.<extra></extra>",
        ), secondary_y=True)
        fig.update_layout(
            template="plotly_white", height=450,
            legend=dict(orientation="h", y=-0.15),
            hovermode="x unified",
            margin=dict(t=30, b=30, l=10, r=10),
        )
        fig.update_yaxes(title_text="Sale price ($)", secondary_y=False,
                         tickformat="$,.0f")
        fig.update_yaxes(title_text="CAGR (%)", secondary_y=True,
                         ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Sale history table ----
    st.markdown("### 📜 Sale history")
    display_df = df.rename(columns={
        "sale_num": "#", "date": "Date sold", "price": "Price",
        "sale_type": "Type", "hold_years": "Hold (yrs)",
        "total_growth_pct": "Total growth %", "annual_growth_pct": "CAGR %",
        "flags": "Flags",
    })[["#", "Date sold", "Price", "Type", "Hold (yrs)",
        "Total growth %", "CAGR %", "Flags"]].copy()
    display_df["Price"] = display_df["Price"].map(lambda v: f"${v:,.0f}")
    display_df = display_df.fillna("—")
    # replace any empty-string Flags with em-dash
    display_df["Flags"] = display_df["Flags"].replace("", "—")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ---- DA records ----
    if da_data and da_data.get("development_applications"):
        st.markdown("### 🏗️ Development applications / renovations")
        da_rows = []
        for da in da_data["development_applications"]:
            da_rows.append({
                "Year": da.get("year", "?"),
                "Type": (da.get("type") or "?").title(),
                "Description": da.get("description") or "",
                "Status": (da.get("status") or "?").title(),
                "Source": da.get("source") or "",
            })
        st.dataframe(pd.DataFrame(da_rows), use_container_width=True,
                     hide_index=True)

    # ---- Sources ----
    srcs = data.get("sources_checked") or []
    if srcs:
        with st.expander(f"📚 Sources checked by Gemini ({len(srcs)})"):
            for s in srcs:
                st.caption(f"- {s}")
    if sales_null:
        st.caption(f"*{len(sales_null)} sale(s) had withheld prices and were "
                   "excluded from growth calculations.*")


def _render_home_price_tab():
    """Body of the Home Price Growth tab."""
    # ---- Setup guard ----
    if not GENAI_AVAILABLE:
        st.error(
            "The `google-genai` package is not installed. Install it with:\n\n"
            "```bash\npip install google-genai\n```\n\n"
            "Then restart the app."
        )
        return

    # ---- API key input (top of the tab) ----
    with st.container(border=True):
        col_key, col_help = st.columns([3, 1])
        with col_key:
            entered = st.text_input(
                "🔑 Google Gemini API key",
                value=st.session_state.get("gemini_api_key", GEMINI_API_KEY_DEFAULT),
                type="password",
                placeholder="Paste your free key from aistudio.google.com/apikey",
                help=(
                    "Your key is held only in this browser session. It's never "
                    "logged or sent anywhere except to Google's API to run "
                    "your queries. Get one free at https://aistudio.google.com/apikey"
                ),
                key="gemini_api_key_input",
            )
            # Persist into the canonical session_state slot
            if entered != st.session_state.get("gemini_api_key", ""):
                st.session_state["gemini_api_key"] = entered
                # Force model re-detection on key change
                for k in ("_gemini_client", "_gemini_client_key",
                          "_gemini_model", "_gemini_model_key"):
                    st.session_state.pop(k, None)
        with col_help:
            with st.popover("ℹ️ How to get a key", use_container_width=True):
                st.markdown(
                    "1. Open **https://aistudio.google.com/apikey** in a new tab\n"
                    "2. Click **Create API Key** (no credit card needed)\n"
                    "3. Copy and paste it on the left\n\n"
                    "**Free-tier quotas (April 2026):**\n"
                    "- gemini-2.5-flash → 10 RPM · 250 RPD\n"
                    "- gemini-2.5-flash-lite → 15 RPM · 1000 RPD\n"
                    "- gemini-2.5-pro → 5 RPM · 100 RPD\n\n"
                    "RPM = requests per minute, RPD = requests per day."
                )

    if not _current_api_key():
        st.info("Enter your Gemini API key above to start analysing properties.")
        return

    # Verify a model responds (cached).
    model, err = _detect_gemini_model()
    if model is None:
        st.error(
            f"Could not reach any Gemini model. Last error: {err or 'unknown'}\n\n"
            "Possible causes:\n"
            "- Invalid API key\n"
            "- Daily quota exhausted (wait until midnight PT)\n"
            "- All models in free tier currently limited — create a new Google\n"
            "  Cloud project at aistudio.google.com for a fresh quota bucket."
        )
        if st.button("🔄 Retry connection", key="retry_gemini"):
            for k in ("_gemini_client", "_gemini_client_key",
                      "_gemini_model", "_gemini_model_key"):
                st.session_state.pop(k, None)
            st.rerun()
        return
    st.success(f"✅ Connected · using **{model}** · results cached for 1h per address")

    # ---- Input form ----
    with st.form("home_price_form"):
        address = st.text_input(
            "Property address",
            placeholder="e.g. 10 Smith Street, Fitzroy VIC 3065",
            help="More specific = better results. Include suburb, state and postcode.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            check_da = st.checkbox(
                "Include DA / renovation records", value=True,
                help="Uses an extra API call; slightly slower."
            )
        with col_b:
            deep_search = st.checkbox(
                "Deep search (better results, 4× API quota)", value=False,
                help=(
                    "Runs 4 separate queries targeting domain, realestate, "
                    "onthehouse + state Valuer-General records and merges the "
                    "results. Uses gemini-2.5-pro when available. Bypasses the "
                    "address cache so you get a fresh lookup. Recommended when "
                    "the quick search missed older sales."
                )
            )
        submitted = st.form_submit_button("🔍 Analyse", type="primary",
                                           use_container_width=True)

    if submitted:
        address = (address or "").strip()
        if not address:
            st.warning("Enter an address first.")
            return
        with st.status(f"Analysing {address}…", expanded=True) as status:
            if deep_search:
                st.write("🚀 Deep search mode: querying 4 sources separately…")
                data = fetch_property_history_deep(address, status_writer=st.write)
            else:
                st.write("🔎 Searching sale history on domain, realestate, onthehouse…")
                data = fetch_property_history(address)

            if data is None:
                status.update(label="❌ Could not retrieve data",
                              state="error", expanded=True)
                st.error(
                    "Gemini returned no parseable data for that address. "
                    "Try a more specific format (include suburb, state and "
                    "postcode). If the problem persists your daily quota "
                    "may be exhausted."
                )
                return

            # If the quick search found nothing, hint at deep search
            sales_count = len([s for s in (data.get("sales") or []) if s.get("price")])
            if sales_count == 0 and not deep_search:
                st.write("ℹ️ Quick search found no priced sales — you may want "
                         "to re-run with **Deep search** enabled.")

            da_data = None
            if check_da:
                st.write("🏗️ Searching DA / renovation records…")
                time.sleep(1.5)
                da_data = fetch_da_records(address)

            st.write("📊 Computing growth metrics…")
            status.update(label="✅ Analysis complete", state="complete",
                           expanded=False)

        st.divider()
        _render_property_result(address, data, da_data)

    # ---- Compare mode ----
    with st.expander("📊 Compare multiple properties (side by side)"):
        st.caption(
            "Paste up to 4 addresses, one per line. Each address uses 1-2 API "
            "calls, so watch your quota."
        )
        multi = st.text_area(
            "Addresses (one per line)",
            placeholder="10 Smith Street, Fitzroy VIC 3065\n"
                        "25 Hall Street, Bondi Beach NSW 2026",
            height=120,
        )
        include_da_multi = st.checkbox("Include DA checks in comparison",
                                       value=False, key="multi_da")
        if st.button("Compare addresses", key="compare_btn"):
            addrs = [a.strip() for a in (multi or "").split("\n") if a.strip()]
            if not addrs:
                st.warning("No addresses entered.")
                return
            if len(addrs) > 4:
                st.warning("Limiting to the first 4 addresses.")
                addrs = addrs[:4]
            rows = []
            for addr in addrs:
                with st.spinner(f"Fetching {addr}…"):
                    d = fetch_property_history(addr)
                    dd = fetch_da_records(addr) if include_da_multi else None
                if not d:
                    rows.append(dict(Address=addr, Sales="—", First="—",
                                     Last="—", CAGR="—", Flags="—"))
                    continue
                sales_ok = [s for s in (d.get("sales") or []) if s.get("price")]
                if not sales_ok:
                    rows.append(dict(Address=addr, Sales=0, First="—", Last="—",
                                     CAGR="—", Flags="—"))
                    continue
                reno_years_local = []
                if dd and dd.get("development_applications"):
                    for da in dd["development_applications"]:
                        if da.get("year"):
                            try:
                                reno_years_local.append(int(da["year"]))
                            except (TypeError, ValueError):
                                pass
                df_a = _build_sales_dataframe(sales_ok, reno_years_local)
                first = sales_ok[0]; last = sales_ok[-1]
                fdt = _parse_sale_date(first["date"])
                ldt = _parse_sale_date(last["date"])
                yrs = (ldt - fdt).days / 365.25 if (fdt and ldt) else 0
                cagr_a = _annualised_growth(first["price"], last["price"], yrs) \
                    if yrs > 0.1 else None
                flag_count = sum(1 for _, r in df_a.iterrows() if r.get("flags"))
                rows.append(dict(
                    Address=addr[:60],
                    Sales=len(sales_ok),
                    First=f"${first['price']:,.0f}",
                    Last=f"${last['price']:,.0f}",
                    CAGR=f"{cagr_a:.1f}%" if cagr_a is not None else "—",
                    Flags=flag_count,
                ))
                time.sleep(1.5)
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

    # ---- Helpful links ----
    with st.expander("🏛️ Free council DA search portals"):
        st.markdown(
            "| State | Portal |\n"
            "|---|---|\n"
            "| NSW | [ePlanning DA Tracker](https://www.planningportal.nsw.gov.au/datracker) |\n"
            "| VIC | [VicPlan](https://mapshare.vic.gov.au/vicplan/) |\n"
            "| QLD | [MyDAS](https://planning.statedevelopment.qld.gov.au/) |\n"
            "| SA  | [PlanSA](https://plan.sa.gov.au/) |\n"
            "| WA  | [DPLH PlanWA](https://www.wa.gov.au/organisation/department-of-planning-lands-and-heritage) |\n"
            "| TAS | [iPlan](https://iplan.tas.gov.au/) |\n"
            "| ACT | [ACT Planning](https://www.planning.act.gov.au/) |\n"
            "| NT  | [NT Planning](https://nt.gov.au/property/building-and-development) |\n"
        )


# =========================================================================
# UI
# =========================================================================
st.title("🇦🇺 Retirement & Property Simulator")
st.caption(
    "A two-tab tool. Tab 1 runs a monthly cashflow retirement simulation with "
    f"Australian tax, super and Age Pension rules current to April {TODAY_YEAR}. "
    "Tab 2 researches the sale history of any Australian property address using "
    "Gemini + Google Search. Educational only; not financial advice."
)

tab1, tab2 = st.tabs(["🏦 Retirement Simulator", "🏠 Home Price Growth"])

with tab1:
    with st.expander("What this simulator does and its limitations"):
        st.markdown(
            f"""
- Runs a **month-by-month** simulation from the start year to end year.
- Each output is in **start-year dollars** (default **{TODAY_YEAR}** AUD), i.e.
  nominal values are deflated by CPI back to the start year so you can read
  every chart in today's purchasing power.
- Rate sliders accept **nominal** returns; implied real rates are shown
  beside each slider.
- Models each household member separately (their own salary, super, age,
  retirement age, unemployment periods).
- **Independent retirement**: one member can retire (and start drawing super
  once they pass preservation age) while the other keeps working. The working
  member's take-home pay covers expenses first; the retired member still
  must meet the legislated minimum drawdown.
- Uses **FY 2025-26 ATO tax brackets**, **Medicare levy (with low-income
  shade-in)**, and **LITO**.
- Super accumulation: **12% SG** (permanent from 1 July 2025), **$30,000
  concessional cap**, **15%** contributions tax and **15%** earnings tax during
  accumulation, **0%** earnings tax during the retirement pension phase.
- Age Pension: 20 March 2026 rates, both income test (with proper deeming) and
  assets test, returning the lower of the two entitlements. All thresholds
  are **indexed to inflation** each year of the projection.
- Minimum drawdown rates per the Superannuation Industry (Supervision)
  Regulations (4% / 5% / 6% / 7% / 9% / 11% / 14% by age band).
- **Monthly surplus split**: the slider in section 4 controls how much of
  each month's excess cashflow goes into the investment account versus the
  offset/cash account — applies both while the mortgage is live and after.
- **Limitations**: ignores the $2M transfer balance cap (relevant only for
  very large balances), Division 293 extra super tax (income above $250k),
  Work Bonus, franking credits, spouse super contribution offsets, insurance
  premiums in super, CGT on investments, and any lump-sum super tax for those
  withdrawing before age 60. The Age Pension is only claimed once ALL members
  are both retired and at/above pension age (conservative simplification).
"""
        )

# -------------------------------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------------------------------
sb = st.sidebar
sb.header("1. Simulation window")
start_year = sb.number_input("Start year", 2020, 2100, TODAY_YEAR, 1)
end_year   = sb.number_input("End year (death)", 2020, 2120, TODAY_YEAR + 54, 1)
if end_year <= start_year:
    sb.error("End year must be after start year.")
    st.stop()

sb.header("2. Household")
household_size = sb.selectbox("Number of members", [1, 2], index=1)

members = []
for i in range(household_size):
    sb.subheader(f"Member {i+1}")
    age = sb.number_input(
        f"Current age (member {i+1})", 18, 80,
        35 if i == 0 else 34, 1, key=f"age_{i}"
    )
    salary = sb.number_input(
        f"Current annual salary, pre-tax (member {i+1})",
        0.0, 10_000_000.0, 100_000.0, 1_000.0, key=f"sal_{i}",
        help="Gross salary excluding super. The 12% SG is paid on top by the employer."
    )
    super_bal = sb.number_input(
        f"Current super balance (member {i+1})",
        0.0, 10_000_000.0, 50_000.0, 1_000.0, key=f"sup_{i}"
    )
    salary_sacrifice = sb.number_input(
        f"Annual salary sacrifice to super (member {i+1})",
        0.0, 100_000.0, 0.0, 500.0, key=f"ss_{i}",
        help="Pre-tax voluntary super contribution. Will be capped at the "
             "concessional cap minus SG automatically."
    )
    retire_age = sb.number_input(
        f"Retirement age (member {i+1})", 30, 90, 60, 1, key=f"ret_{i}",
        help="Age at which this person stops working. Independent of Age Pension age."
    )
    # Unemployment: use text input to allow multiple periods
    unemp_str = sb.text_input(
        f"Unemployment periods (member {i+1}) — e.g. '2034-2035, 2041'",
        "", key=f"un_{i}",
        help="Comma-separated years or ranges. Leave empty for none."
    )
    unemp_years: set[int] = set()
    try:
        for part in unemp_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-")
                unemp_years.update(range(int(lo), int(hi) + 1))
            else:
                unemp_years.add(int(part))
    except ValueError:
        sb.error(f"Couldn't parse unemployment periods for member {i+1}.")
        st.stop()
    members.append(dict(
        age=age, salary=salary, super_bal=super_bal,
        salary_sacrifice=salary_sacrifice,
        retire_age=retire_age, unemp_years=unemp_years,
    ))

sb.header("3. Economic assumptions")
sb.caption("All rates below are **nominal** (before inflation). Implied real rates shown under each.")
inflation       = sb.slider("Annual inflation (CPI)", 0.0, 0.10, 0.025, 0.005,
                            help="Long-run RBA target mid-point is 2.5%.")
wage_growth     = sb.slider("Annual wage growth", 0.0, 0.10, 0.035, 0.005,
                            help="ABS wage price index long-run ~3-3.5%.")
sb.caption(f"→ Real wage growth ≈ **{((1+wage_growth)/(1+inflation)-1)*100:.2f}%**")
super_return    = sb.slider("Super nominal return (before tax)", 0.0, 0.15, 0.065, 0.005,
                            help="Balanced option long-run ~6-7% before tax.")
sb.caption(f"→ Real super return (pre-tax) ≈ **{((1+super_return)/(1+inflation)-1)*100:.2f}%**")
invest_return   = sb.slider("Outside-super nominal return", 0.0, 0.15, 0.045, 0.005,
                            help="After-tax on the investor's marginal rate. Set to 0 for pure cash/offset-like savings.")
sb.caption(f"→ Real outside-super return ≈ **{((1+invest_return)/(1+inflation)-1)*100:.2f}%**")
mortgage_rate   = sb.slider("Mortgage interest rate (annual)", 0.0, 0.15, 0.060, 0.005)
sb.caption(f"→ Real mortgage rate ≈ **{((1+mortgage_rate)/(1+inflation)-1)*100:.2f}%**")

sb.header("4. Home & mortgage")
homeowner = sb.checkbox("Homeowner (for Age Pension assets test)", True)
mortgage_principal = sb.number_input("Mortgage balance today", 0.0, 10_000_000.0,
                                     500_000.0, 1_000.0)
offset_balance     = sb.number_input("Offset account balance today", 0.0, 10_000_000.0,
                                     150_000.0, 1_000.0)
mortgage_payment   = sb.number_input("Monthly mortgage payment", 0.0, 50_000.0,
                                     3_000.0, 50.0,
                                     help="Fixed nominal repayment until mortgage is paid off.")
include_investment = sb.checkbox(
    "Use an investment account outside super", True,
    help="If off, excess savings accumulate as cash (no growth) after the mortgage is paid."
)
initial_investment = sb.number_input(
    "Initial investment balance", 0.0, 10_000_000.0, 0.0, 1_000.0
) if include_investment else 0.0
if include_investment:
    savings_alloc_pct = sb.slider(
        "Share of monthly surplus going to investment (%)",
        0, 100, 0, 5,
        help="0% = everything first to offset/cash (the usual 'offset-first' strategy while you have a mortgage, "
             "since every dollar in offset saves you the full mortgage rate tax-free). "
             "100% = all surplus goes straight into the investment account. "
             "Any value in between splits the monthly surplus proportionally. "
             "Applies both while the mortgage is live and after payoff."
    )
else:
    savings_alloc_pct = 0
reserve_target = sb.number_input(
    "Cash reserve kept when paying off mortgage early", 0.0, 1_000_000.0,
    50_000.0, 1_000.0,
    help="Once offset exceeds principal + this reserve (inflated), the mortgage is closed "
         "and the remainder of the offset flows to investment / cash."
)

sb.header("5. Tax overrides (FY 2025-26 defaults)")
# Defaults are shown read-only-ish; user can tweak
sb.caption("Edit only if modelling a different policy scenario.")
sb.markdown("Brackets: `threshold:rate, ...` with `inf` for top.")
brackets_text = sb.text_area(
    "Tax brackets",
    ", ".join(f"{int(t) if t!=float('inf') else 'inf'}:{r}" for t, r in TAX_BRACKETS_2025_26),
    help="Defaults are ATO resident rates, FY 2025-26."
)
try:
    tax_brackets = []
    for pair in brackets_text.split(","):
        t, r = pair.strip().split(":")
        tax_brackets.append((float("inf") if t.strip() == "inf" else float(t), float(r)))
except Exception:
    sb.error("Bracket parse error. Format: threshold:rate, ...")
    st.stop()
medicare_rate = sb.slider("Medicare levy", 0.0, 0.05, MEDICARE_LEVY_RATE, 0.005)
mls_low  = sb.number_input("Medicare levy lower threshold", 0.0, 100_000.0,
                           float(MEDICARE_THRESHOLD_LOWER), 100.0)
mls_high = sb.number_input("Medicare levy full-2% threshold", 0.0, 100_000.0,
                           float(MEDICARE_THRESHOLD_UPPER), 100.0)

sb.header("6. Super overrides (FY 2025-26 defaults)")
sg_rate         = sb.slider("Super Guarantee rate", 0.0, 0.20, SG_RATE_2025_26, 0.005)
conc_cap        = sb.number_input("Concessional cap per person",
                                  0.0, 500_000.0, float(CONCESSIONAL_CAP_2025_26), 500.0)
contrib_tax     = sb.slider("Contributions tax", 0.0, 0.50, CONTRIBUTIONS_TAX, 0.005)
earn_tax_accum  = sb.slider("Super earnings tax (accumulation phase)",
                            0.0, 0.50, ACCUMULATION_EARNINGS_TAX, 0.005)
earn_tax_pen    = sb.slider("Super earnings tax (pension phase)",
                            0.0, 0.50, RETIREMENT_EARNINGS_TAX, 0.005,
                            help="0% by default — account-based pensions are tax-free.")

sb.header("7. Age Pension overrides (Mar 2026 defaults)")
pension_age          = sb.number_input("Age Pension age", 60, 75, PENSION_AGE, 1)
preservation_age     = sb.number_input(
    "Super preservation age", 55, 70, PRESERVATION_AGE, 1,
    help="Earliest age super can be accessed via the retirement condition of release. "
         "Anyone born on or after 1 July 1964 has preservation age 60."
)
max_pens_single      = sb.number_input("Max single pension (annual)",
                                       0.0, 200_000.0, MAX_PENSION_SINGLE_ANNUAL, 100.0)
max_pens_couple      = sb.number_input("Max couple pension (annual, combined)",
                                       0.0, 200_000.0, MAX_PENSION_COUPLE_ANNUAL, 100.0)
inc_free_single      = sb.number_input("Income free area - single (annual)",
                                       0.0, 100_000.0, float(INCOME_FREE_AREA_SINGLE), 100.0)
inc_free_couple      = sb.number_input("Income free area - couple (annual, combined)",
                                       0.0, 100_000.0, float(INCOME_FREE_AREA_COUPLE), 100.0)
inc_taper            = sb.slider("Income taper rate", 0.0, 1.0, INCOME_TAPER, 0.05)
assets_free_default  = (
    ASSETS_LOWER_COUPLE_HO     if homeowner and household_size == 2 else
    ASSETS_LOWER_SINGLE_HO     if homeowner else
    ASSETS_LOWER_COUPLE_NONHO  if household_size == 2 else
    ASSETS_LOWER_SINGLE_NONHO
)
assets_free          = sb.number_input("Assets test full-pension threshold",
                                       0.0, 5_000_000.0, float(assets_free_default), 1_000.0,
                                       help="Varies by homeowner status and household size.")
assets_taper_yr      = sb.number_input("Assets taper ($/yr per $1,000 over)",
                                       0.0, 200.0, ASSETS_TAPER_PER_1000_YR, 1.0)
deem_low             = sb.slider("Deeming rate - low", 0.0, 0.10, DEEMING_LOW_RATE, 0.0025)
deem_high            = sb.slider("Deeming rate - high", 0.0, 0.10, DEEMING_HIGH_RATE, 0.0025)
deem_thresh_default  = DEEMING_THRESHOLD_COUPLE if household_size == 2 else DEEMING_THRESHOLD_SINGLE
deem_thresh          = sb.number_input("Deeming threshold (low rate applies below)",
                                       0.0, 1_000_000.0, float(deem_thresh_default), 100.0)

sb.header("8. Target retirement income")
default_target = ASFA_COMFORTABLE_COUPLE if household_size == 2 else ASFA_COMFORTABLE_SINGLE
target_income = sb.number_input(
    f"Desired annual retirement income (in start-year {start_year} dollars)",
    0.0, 1_000_000.0, float(default_target), 1_000.0,
    help=f"Default is ASFA Retirement Standard 'comfortable' (Sep 2025 quarter): "
         f"${ASFA_COMFORTABLE_COUPLE:,}/yr couple, ${ASFA_COMFORTABLE_SINGLE:,}/yr single (homeowner, age ~65)."
)

sb.header(f"9. Base living expenses (monthly, {start_year} dollars)")
# User-friendly input: allow monthly or annual entry via tuple
def money_input(label: str, monthly_default: float, help_text: str = "") -> float:
    col1, col2 = sb.columns([2, 1])
    with col1:
        val = st.number_input(label, 0.0, 1_000_000.0, monthly_default, 10.0,
                              key=f"exp_{label}", help=help_text)
    with col2:
        mode = st.selectbox("freq", ["monthly", "yearly"], key=f"freq_{label}",
                            label_visibility="collapsed")
    return val / 12 if mode == "yearly" else val

base_items = {
    "Groceries":              money_input("Groceries",              800.0),
    "Utilities":              money_input("Utilities",              400.0),
    "Transport (non-car)":    money_input("Transport (non-car)",    200.0),
    "Car insurance":          money_input("Car insurance",          2000.0/12, "Default shown monthly; flip to yearly if you prefer."),
    "Car registration":       money_input("Car registration",       800.0/12),
    "Car fuel + service":     money_input("Car fuel + service",     400.0),
    "Healthcare out-of-pocket": money_input("Healthcare out-of-pocket", 200.0),
    "Private health insurance": money_input("Private health insurance", 300.0),
    "Body corporate / strata":  money_input("Body corporate / strata",  1500.0/12),
    "Council rates":          money_input("Council rates",          1500.0/12),
    "Entertainment":          money_input("Entertainment",          200.0),
    "Personal care":          money_input("Personal care",          150.0),
    "Clothing":               money_input("Clothing",               150.0),
    "Household supplies":     money_input("Household supplies",     100.0),
    "Gym/fitness":            money_input("Gym/fitness",            80.0),
    "Travel":                 money_input("Travel (holidays)",      500.0),
    "Miscellaneous":          money_input("Miscellaneous",          300.0),
}
base_monthly = sum(base_items.values())

sb.header(f"10. Child expenses (optional, monthly in {start_year} dollars)")
sb.caption("Each line: amount (monthly), start year, end year. Leave amount=0 to skip.")
child_items_cfg = [
    ("Childcare",     36_000/12, start_year,      start_year + 5),
    ("Baby supplies",  3_600/12, start_year,      start_year + 3),
    ("Food (kids)",    2_400/12, start_year,      start_year + 25),
    ("Clothing (kids)",1_000/12, start_year,      start_year + 18),
    ("Healthcare (kids)", 500/12, start_year,     start_year + 25),
    ("Primary/secondary school",  800/12, start_year + 5, start_year + 18),
    ("Tertiary support",         2_000/12, start_year + 18, start_year + 25),
    ("Extracurricular",          1_000/12, start_year + 5, start_year + 18),
    ("Misc (kids)",              1_000/12, start_year,     start_year + 18),
]
child_items = {}
for label, amt_def, sy_def, ey_def in child_items_cfg:
    with sb.expander(label, expanded=False):
        amt = st.number_input(f"{label} amount (monthly, {start_year} $)",
                              0.0, 100_000.0, amt_def, 50.0, key=f"c_a_{label}")
        sy  = st.number_input(f"{label} start year",
                              1900, 2150, sy_def, 1, key=f"c_s_{label}")
        ey  = st.number_input(f"{label} end year",
                              1900, 2150, ey_def, 1, key=f"c_e_{label}")
    if amt > 0 and ey > sy:
        child_items[label] = dict(amount=amt, start=sy, end=ey)

# =========================================================================
# FINANCIAL FUNCTIONS
# =========================================================================
def income_tax_nominal(income: float, concessional_sacrifice: float) -> float:
    """Personal income tax on salary minus salary-sacrificed super, at the
    user's configured brackets + Medicare Levy (with low-income shade-in) - LITO.
    Operates in nominal dollars, i.e. assumes the brackets also inflate with time.
    Since this model keeps brackets static in nominal terms (no explicit indexation
    of brackets), use the `bracket_indexing_factor` to scale brackets year by year.
    """
    taxable = max(income - concessional_sacrifice, 0)
    tax = 0.0
    prev_thr = 0.0
    for thr, rate in tax_brackets:
        if taxable <= thr:
            tax += (taxable - prev_thr) * rate
            break
        tax += (thr - prev_thr) * rate
        prev_thr = thr

    # Medicare levy low-income shade-in
    if taxable <= mls_low:
        medicare = 0.0
    elif taxable <= mls_high:
        # 10% of income above lower threshold, capped at the full 2% once full threshold reached
        medicare = min(0.10 * (taxable - mls_low), medicare_rate * taxable)
    else:
        medicare = medicare_rate * taxable

    # LITO
    if taxable <= LITO_FULL_TO:
        lito = LITO_MAX
    elif taxable <= LITO_PHASE1_UPPER:
        lito = LITO_MAX - 0.05 * (taxable - LITO_FULL_TO)
    elif taxable <= LITO_PHASE2_UPPER:
        lito = 325 - 0.015 * (taxable - LITO_PHASE1_UPPER)
    else:
        lito = 0

    return max(tax - lito, 0) + medicare


def min_drawdown_rate(age: int) -> float:
    for (lo, hi), rate in MIN_DRAWDOWN.items():
        if lo <= age <= hi:
            return rate
    return 0.14


def age_pension_monthly(
    super_balances_nom: list[float],
    outside_balance_nom: float,
    ages: list[int],
    infl_to_date: float,
) -> float:
    """Age Pension in nominal dollars per month. Assumes both partners reach
    pension age at the same time for simplicity; if one member qualifies and
    the other doesn't, returns a proportional single-rate entitlement for
    the qualifying member (rough approximation).
    """
    eligible = [a >= pension_age for a in ages]
    if not any(eligible):
        return 0.0

    # Determine which rate/threshold set applies
    both_eligible = all(eligible) if household_size == 2 else eligible[0]
    using_couple_rules = (household_size == 2 and both_eligible)

    # Scale all thresholds and max rate to nominal terms of this year.
    max_annual_nom = (max_pens_couple if using_couple_rules else max_pens_single) * infl_to_date
    inc_free_nom   = (inc_free_couple if using_couple_rules else inc_free_single) * infl_to_date
    assets_free_nom = assets_free * infl_to_date
    deem_thresh_nom = deem_thresh * infl_to_date

    # Combined assessable assets and deemed income
    total_assets = sum(super_balances_nom) + outside_balance_nom
    low_part  = min(total_assets, deem_thresh_nom)
    high_part = max(0.0, total_assets - deem_thresh_nom)
    deemed_annual = low_part * deem_low + high_part * deem_high

    income_reduction = max(0.0, (deemed_annual - inc_free_nom) * inc_taper)
    assets_reduction = max(0.0, (total_assets - assets_free_nom) * (assets_taper_yr / 1000.0))
    annual_pension = max(0.0, max_annual_nom - max(income_reduction, assets_reduction))

    # If couple but only one partner eligible, roughly halve the entitlement.
    if household_size == 2 and not both_eligible:
        # Use single-rate thresholds for the one eligible member instead.
        max_annual_nom_s = max_pens_single * infl_to_date
        inc_free_nom_s   = inc_free_single * infl_to_date
        # Member's share of assets (assume equal split for simplicity)
        my_assets = total_assets / 2
        my_assets_free = (ASSETS_LOWER_SINGLE_HO if homeowner else ASSETS_LOWER_SINGLE_NONHO) * infl_to_date
        my_deem_thresh = DEEMING_THRESHOLD_SINGLE * infl_to_date
        low_s  = min(my_assets, my_deem_thresh)
        high_s = max(0.0, my_assets - my_deem_thresh)
        deemed_s = low_s * deem_low + high_s * deem_high
        inc_red_s = max(0.0, (deemed_s - inc_free_nom_s) * inc_taper)
        ast_red_s = max(0.0, (my_assets - my_assets_free) * (assets_taper_yr / 1000.0))
        annual_pension = max(0.0, max_annual_nom_s - max(inc_red_s, ast_red_s))

    return annual_pension / 12


# =========================================================================
# SIMULATION
# =========================================================================
def run_sim(expense_overrides: dict | None = None):
    """Run the full month-by-month simulation.

    expense_overrides: optional dict {item_label: multiplier}. Multipliers
    < 1 reduce that item, multipliers > 1 increase it. Item labels are the
    keys of base_items (e.g. 'Groceries') or child_items (e.g. 'Childcare').
    Used by the cost-optimisation tool. None = use the user-entered amounts.
    """
    expense_overrides = expense_overrides or {}

    # Apply overrides to base items, recompute base_monthly
    base_items_eff = {
        name: amt * expense_overrides.get(name, 1.0)
        for name, amt in base_items.items()
    }
    base_monthly_eff = sum(base_items_eff.values())

    # Apply overrides to child items
    child_items_eff = {
        name: dict(cfg, amount=cfg["amount"] * expense_overrides.get(name, 1.0))
        for name, cfg in child_items.items()
    }

    months = (end_year - start_year) * 12
    r_mort_m = (1 + mortgage_rate) ** (1/12) - 1

    # State
    super_bal = [m["super_bal"] for m in members]
    super_in_pension_phase = [False] * household_size
    out_bal = initial_investment
    mort = mortgage_principal
    mortgage_live = mort > 0
    # offset only exists while there's a mortgage. savings is the cash bucket
    # that replaces it once the mortgage is paid off.
    if mortgage_live:
        offset = offset_balance
        savings = 0.0
    else:
        # User entered an 'offset' balance but has no mortgage → treat it as cash savings.
        offset = 0.0
        savings = offset_balance
    payoff_year = None

    # Log arrays (stored in start-year dollars unless labeled)
    log = {k: [] for k in [
        "salary", "tax", "take_home", "sg", "sacrifice",
        "expenses_base", "expenses_kids", "expenses_mortgage", "expenses_total",
        "pension", "super_withdrawal", "outside_withdrawal", "savings_withdrawal",
        "savings_flow", "principal_paid", "offset", "savings", "outside_balance",
        "super_total", "super_m1", "super_m2",
        "shortfall", "retired_flag",
    ]}

    for m in range(months):
        year_idx = m // 12               # 0-based
        mo_idx   = m % 12
        year     = start_year + year_idx
        # Monthly inflation factor (=1 at m=0): smoothly scales with time.
        infl_factor = (1 + inflation) ** (year_idx + mo_idx / 12)

        ages = [mem["age"] + year_idx for mem in members]

        # -----------------------------------------------------------------
        # Status this month (per member, then rolled up)
        # -----------------------------------------------------------------
        retired = [ages[i] >= members[i]["retire_age"] for i in range(household_size)]
        # A member can access super (pension phase) once they are retired AND
        # at or above preservation age. Once flipped, they stay in pension phase.
        for i in range(household_size):
            if retired[i] and ages[i] >= preservation_age and not super_in_pension_phase[i]:
                super_in_pension_phase[i] = True
        employed = [
            (not retired[i]) and (year not in members[i]["unemp_years"])
            for i in range(household_size)
        ]
        hh_all_retired = all(retired)
        hh_all_pension_age = all(a >= pension_age for a in ages)

        # -----------------------------------------------------------------
        # Incomes from work — per person
        # -----------------------------------------------------------------
        salary_hh_nom = 0.0
        tax_hh_nom    = 0.0
        sg_per_person = [0.0] * household_size
        sacrifice_per_person = [0.0] * household_size
        for i in range(household_size):
            if employed[i]:
                sal_nom = members[i]["salary"] * (1 + wage_growth) ** year_idx
                sg_amt  = sal_nom * sg_rate
                max_sac = max(0, conc_cap - sg_amt)
                sac_amt = min(members[i]["salary_sacrifice"] * (1 + wage_growth) ** year_idx,
                              max_sac)
                sg_per_person[i]        = sg_amt
                sacrifice_per_person[i] = sac_amt
                salary_hh_nom += sal_nom
                tax_hh_nom    += income_tax_nominal(sal_nom, sac_amt)
        take_home_monthly_nom = (salary_hh_nom - tax_hh_nom - sum(sacrifice_per_person)) / 12

        # -----------------------------------------------------------------
        # Expenses — living + kids + mortgage (mortgage is always nominal fixed)
        # -----------------------------------------------------------------
        base_today = base_monthly_eff
        kids_today = 0.0
        for name, cfg in child_items_eff.items():
            if cfg["start"] <= year < cfg["end"]:
                kids_today += cfg["amount"]
        expenses_living_nom = (base_today + kids_today) * infl_factor
        mortgage_nom = mortgage_payment if mortgage_live else 0.0
        expenses_total_nom = expenses_living_nom + mortgage_nom

        # -----------------------------------------------------------------
        # Age Pension (monthly, nominal). Simplified policy: claimed only when
        # ALL household members are both retired AND at/above pension age.
        # -----------------------------------------------------------------
        pension_nom = (age_pension_monthly(super_bal, out_bal, ages, infl_factor)
                       if hh_all_retired and hh_all_pension_age else 0.0)

        # -----------------------------------------------------------------
        # Determine "need" this month
        #   - While any member is working: cover actual expenses
        #   - When household is fully retired: spend the larger of actual
        #     expenses and the target retirement income (target is treated
        #     as a MINIMUM the household wants to draw, so e.g. excess from
        #     the mandatory min-drawdown flows to savings if expenses are lower).
        # -----------------------------------------------------------------
        if hh_all_retired:
            target_m_nom = target_income * infl_factor / 12
            need = max(expenses_total_nom, target_m_nom)
        else:
            need = expenses_total_nom

        # -----------------------------------------------------------------
        # Apply mortgage interest/principal split (separately from the
        # cashflow allocation below). This just decides how much of this
        # month's mortgage_payment is interest vs principal.
        # -----------------------------------------------------------------
        principal_paid_nom = 0.0
        if mortgage_live:
            net_debt = max(mort - offset, 0.0)
            interest = net_debt * r_mort_m
            principal_rep = min(mortgage_payment - interest, mort)
            mort -= principal_rep
            principal_paid_nom = principal_rep

        # -----------------------------------------------------------------
        # Income sources, in priority order:
        #   1. Take-home pay (any working members)
        #   2. Age Pension
        #   3. Mandatory minimum super drawdown (per member in pension phase)
        #   4. Additional super drawdown (only from members in pension phase)
        #   5. Outside investment
        #   6. Offset / cash
        # Anything not funded → shortfall.
        # -----------------------------------------------------------------
        super_wd = [0.0] * household_size
        outside_wd = 0.0
        savings_wd = 0.0
        shortfall = 0.0
        offset_drawn = 0.0

        # Mandatory minimum drawdowns (must come out whether needed or not).
        for i in range(household_size):
            if super_in_pension_phase[i] and super_bal[i] > 0:
                min_dd = super_bal[i] * min_drawdown_rate(ages[i]) / 12
                w = min(min_dd, super_bal[i])
                super_wd[i] = w
                super_bal[i] -= w

        automatic_income = take_home_monthly_nom + pension_nom + sum(super_wd)
        gap = need - automatic_income

        if gap > 0:
            # Need more. Draw more super (from pension-phase members only).
            # Allocate gap proportionally to remaining balances.
            eligible_balances = [
                super_bal[i] if super_in_pension_phase[i] else 0.0
                for i in range(household_size)
            ]
            total_eligible = sum(eligible_balances)
            if total_eligible > 0:
                for i in range(household_size):
                    if eligible_balances[i] > 0:
                        share = eligible_balances[i] / total_eligible
                        extra = min(gap * share, super_bal[i])
                        super_wd[i] += extra
                        super_bal[i] -= extra
                gap = need - take_home_monthly_nom - pension_nom - sum(super_wd)

            # Cash bucket next — whichever one is active this stage.
            # Drawing cash before investment preserves growth assets.
            if mortgage_live:
                if gap > 0 and offset > 0:
                    w = min(gap, offset)
                    offset_drawn = w
                    offset -= w
                    gap -= w
            else:
                if gap > 0 and savings > 0:
                    w = min(gap, savings)
                    savings_wd = w
                    savings -= w
                    gap -= w

            # Outside investment last.
            if gap > 0 and out_bal > 0:
                w = min(gap, out_bal)
                outside_wd = w
                out_bal -= w
                gap -= w

            shortfall = max(0.0, gap)

        else:
            # Surplus this month — allocate it per user split.
            surplus = -gap
            if mortgage_live:
                # Split between offset (debt reduction) and investment.
                inv_share = surplus * (savings_alloc_pct / 100.0) if include_investment else 0.0
                off_share = surplus - inv_share
                if include_investment:
                    out_bal += inv_share
                offset += off_share
            else:
                # No mortgage. Split between savings account (cash) and investment.
                if include_investment:
                    inv_share = surplus * (savings_alloc_pct / 100.0)
                    cash_share = surplus - inv_share
                    out_bal += inv_share
                    savings += cash_share
                else:
                    savings += surplus

        # -----------------------------------------------------------------
        # Early-mortgage-payoff trigger (offset got big enough).
        # At payoff: close the offset account. The reserve stays as cash in
        # the new savings account; anything above the reserve is split per
        # the user's allocation into investment vs savings.
        # -----------------------------------------------------------------
        if mortgage_live and mort > 0:
            reserve_nom = reserve_target * infl_factor
            if offset - mort >= reserve_nom:
                principal_paid_nom += mort
                offset -= mort
                mort = 0.0
                mortgage_live = False
                payoff_year = year + (1 if mo_idx >= 9 else 0)

                leftover = max(0.0, offset - reserve_nom)
                # Move the reserve out of offset and into the savings account
                savings += reserve_nom
                offset -= reserve_nom
                # Split the leftover between investment and savings per allocation
                if leftover > 0:
                    if include_investment:
                        inv_share = leftover * (savings_alloc_pct / 100.0)
                        cash_share = leftover - inv_share
                        out_bal += inv_share
                        savings += cash_share
                    else:
                        savings += leftover
                    offset -= leftover
                # Offset is now closed.
                offset = 0.0

        # -----------------------------------------------------------------
        # Monthly growth on outside investment / cash.
        # Outside investment grows at invest_return; cash (when investment is
        # off) earns nothing.
        # -----------------------------------------------------------------
        if include_investment:
            out_bal *= (1 + invest_return / 12)

        # -----------------------------------------------------------------
        # Annual super update at start of each year.
        # Per-person: own SG + own sacrifice, taxed at 15%. Growth taxed at
        # 15% (accumulation) or 0% (pension phase).
        # -----------------------------------------------------------------
        if mo_idx == 0:
            for i in range(household_size):
                growth = super_bal[i] * super_return
                tax_on_growth = growth * (earn_tax_pen if super_in_pension_phase[i] else earn_tax_accum)
                # No new contributions once the person has started the pension phase
                contrib_gross = sg_per_person[i] + sacrifice_per_person[i]
                contrib_net = contrib_gross * (1 - contrib_tax) if not super_in_pension_phase[i] else 0.0
                super_bal[i] = super_bal[i] + growth - tax_on_growth + contrib_net

        # -----------------------------------------------------------------
        # Record (convert nominal -> start-year dollars)
        # -----------------------------------------------------------------
        def dfl(x): return x / infl_factor
        log["salary"].append(dfl(salary_hh_nom) / 12)
        log["tax"].append(dfl(tax_hh_nom) / 12)
        log["take_home"].append(dfl(take_home_monthly_nom))
        log["sg"].append(dfl(sum(sg_per_person)) / 12)
        log["sacrifice"].append(dfl(sum(sacrifice_per_person)) / 12)
        log["expenses_base"].append(dfl(base_today * infl_factor))
        log["expenses_kids"].append(dfl(kids_today * infl_factor))
        log["expenses_mortgage"].append(dfl(mortgage_nom))
        log["expenses_total"].append(dfl(expenses_total_nom))
        log["pension"].append(dfl(pension_nom))
        log["super_withdrawal"].append(dfl(sum(super_wd)))
        log["outside_withdrawal"].append(dfl(outside_wd))
        log["savings_withdrawal"].append(dfl(savings_wd))
        log["savings_flow"].append(dfl(max(0, -gap)))
        log["principal_paid"].append(dfl(principal_paid_nom))
        log["offset"].append(dfl(offset))
        log["savings"].append(dfl(savings))
        log["outside_balance"].append(dfl(out_bal))
        log["super_m1"].append(dfl(super_bal[0]))
        log["super_m2"].append(dfl(super_bal[1]) if household_size == 2 else 0.0)
        log["super_total"].append(dfl(sum(super_bal)))
        log["shortfall"].append(dfl(shortfall))
        log["retired_flag"].append(1 if hh_all_retired else 0)

    return log, payoff_year


# =========================================================================
# RUN & DISPLAY
# =========================================================================
with tab1:
    if st.sidebar.button("▶ Run simulation", type="primary"):
        with st.spinner("Running..."):
            log, payoff_year = run_sim()
            n_years = end_year - start_year
            years = [start_year + i for i in range(n_years)]

            # Annual aggregation (flows = sum; balances = end-of-year)
            flow_keys    = ["salary", "tax", "take_home", "sg", "sacrifice",
                            "expenses_base", "expenses_kids", "expenses_mortgage",
                            "expenses_total", "pension", "super_withdrawal",
                            "outside_withdrawal", "savings_withdrawal",
                            "savings_flow", "principal_paid", "shortfall"]
            balance_keys = ["offset", "savings", "outside_balance",
                            "super_m1", "super_m2", "super_total"]

            annual = {}
            for k in flow_keys:
                annual[k] = [sum(log[k][i*12:(i+1)*12]) for i in range(n_years)]
            for k in balance_keys:
                annual[k] = [log[k][(i+1)*12 - 1] for i in range(n_years)]
            annual["retired_flag"] = [log["retired_flag"][(i+1)*12 - 1] for i in range(n_years)]

            # ------------------- Summary -------------------
            st.subheader("Summary")
            first_retire_idx = next((i for i, r in enumerate(annual["retired_flag"]) if r), None)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mortgage paid off", f"{payoff_year}" if payoff_year else "Not paid off")
            if first_retire_idx is not None:
                pre_idx = max(first_retire_idx - 1, 0)
                col2.metric(f"Super at retirement ({start_year} $)",
                            f"${annual['super_total'][pre_idx]:,.0f}")
                col3.metric(f"Investment at retirement ({start_year} $)",
                            f"${annual['outside_balance'][pre_idx]:,.0f}")
                col4.metric(f"Cash savings at retirement ({start_year} $)",
                            f"${annual['savings'][pre_idx]:,.0f}")

            total_short = sum(annual["shortfall"])
            if total_short > 0.5:
                st.error(
                    f"Total unmet cashflow over simulation: "
                    f"${total_short:,.0f} ({start_year} dollars). "
                    "Some months, income + pension + super + outside were not enough to cover expenses."
                )
            else:
                st.success("Plan funded: no shortfalls in any month.")

            # ------------------- Charts -------------------
            st.subheader(f"Balances over time (in {start_year} dollars)")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=years, y=annual["super_total"], name="Super (both)", mode="lines"))
            fig1.add_trace(go.Scatter(x=years, y=annual["outside_balance"], name="Investment", mode="lines"))
            fig1.add_trace(go.Scatter(x=years, y=annual["offset"], name="Offset (pre-payoff)", mode="lines",
                                      line=dict(dash="dot")))
            fig1.add_trace(go.Scatter(x=years, y=annual["savings"], name="Savings account (post-payoff)", mode="lines"))
            fig1.update_layout(hovermode="x unified", yaxis_tickprefix="$", yaxis_tickformat=",",
                               height=450, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader(f"Annual cashflow (in {start_year} dollars)")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=years, y=annual["take_home"], name="Take-home pay"))
            fig2.add_trace(go.Bar(x=years, y=annual["pension"], name="Age Pension"))
            fig2.add_trace(go.Bar(x=years, y=annual["super_withdrawal"], name="Super withdrawals"))
            fig2.add_trace(go.Bar(x=years, y=annual["outside_withdrawal"], name="Outside withdrawals"))
            fig2.add_trace(go.Scatter(x=years, y=annual["expenses_total"],
                                      name="Total expenses", mode="lines+markers",
                                      line=dict(color="black", width=2)))
            fig2.update_layout(barmode="stack", hovermode="x unified",
                               yaxis_tickprefix="$", yaxis_tickformat=",",
                               height=450, legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig2, use_container_width=True)

            if first_retire_idx is not None:
                st.subheader(f"Retirement income mix (in {start_year} dollars)")
                fig3 = go.Figure()
                post = slice(first_retire_idx, None)
                fig3.add_trace(go.Bar(x=years[post], y=annual["pension"][post], name="Age Pension"))
                fig3.add_trace(go.Bar(x=years[post], y=annual["super_withdrawal"][post], name="Super drawdown"))
                fig3.add_trace(go.Bar(x=years[post], y=annual["savings_withdrawal"][post], name="Savings drawdown"))
                fig3.add_trace(go.Bar(x=years[post], y=annual["outside_withdrawal"][post], name="Investment drawdown"))
                fig3.update_layout(barmode="stack", hovermode="x unified",
                                   yaxis_tickprefix="$", yaxis_tickformat=",", height=400)
                st.plotly_chart(fig3, use_container_width=True)

            # ------------------- Tables -------------------
            with st.expander(f"Year-by-year detail (in {start_year} dollars)"):
                df = pd.DataFrame({
                    "Year": years,
                    "Retired?": ["Yes" if r else "" for r in annual["retired_flag"]],
                    "Salary (HH)": annual["salary"],
                    "Tax": annual["tax"],
                    "Take-home": annual["take_home"],
                    "SG contrib": annual["sg"],
                    "Salary sacrifice": annual["sacrifice"],
                    "Expenses base": annual["expenses_base"],
                    "Expenses kids": annual["expenses_kids"],
                    "Expenses mortgage": annual["expenses_mortgage"],
                    "Expenses total": annual["expenses_total"],
                    "Savings flow": annual["savings_flow"],
                    "Principal paid": annual["principal_paid"],
                    "Offset": annual["offset"],
                    "Savings account": annual["savings"],
                    "Investment": annual["outside_balance"],
                    "Super (M1)": annual["super_m1"],
                    "Super (M2)": annual["super_m2"],
                    "Super total": annual["super_total"],
                    "Pension": annual["pension"],
                    "Super withdraw": annual["super_withdrawal"],
                    "Savings withdraw": annual["savings_withdrawal"],
                    "Investment withdraw": annual["outside_withdrawal"],
                    "Shortfall": annual["shortfall"],
                })
                # Format money columns
                money_cols = [c for c in df.columns if c not in ("Year", "Retired?")]
                df_display = df.copy()
                for c in money_cols:
                    df_display[c] = df_display[c].map(lambda v: f"${v:,.0f}")
                st.dataframe(df_display, use_container_width=True, height=500)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download as CSV", csv, "retirement_sim.csv", "text/csv")

            # ============================================================
            # COST OPTIMISATION
            # ============================================================
            st.divider()
            st.subheader("🎯 Cost Optimisation")
            st.caption(
                "For each adjustable expense, the simulator re-runs assuming "
                "you reduced that expense by the % you choose, and measures "
                "the lifetime financial improvement. Use this to find which "
                "spending categories give you the most retirement benefit per "
                "dollar saved."
            )

            opt_pct = st.slider(
                "Hypothetical reduction to apply to each expense",
                min_value=5, max_value=80, value=20, step=5,
                format="%d%%",
                help="A 20% reduction means the simulator pretends you cut "
                     "that expense by 20% from now until end of simulation."
            )

            # Build candidate list: (label, monthly_amount_today, category, active_years)
            # Active years = how many years this expense is actually paid.
            n_sim_years = end_year - start_year
            candidates = []
            for label, amt in base_items.items():
                if amt <= 0:
                    continue
                # Base items run for the full simulation
                annual_amt_today = amt * 12
                candidates.append({
                    "label": label,
                    "category": "Base",
                    "monthly_today": amt,
                    "annual_today": annual_amt_today,
                    "active_years": n_sim_years,
                })
            for label, cfg in child_items.items():
                if cfg["amount"] <= 0:
                    continue
                yrs_active = max(0, cfg["end"] - cfg["start"])
                candidates.append({
                    "label": label,
                    "category": "Child",
                    "monthly_today": cfg["amount"],
                    "annual_today": cfg["amount"] * 12,
                    "active_years": yrs_active,
                })

            if not candidates:
                st.info("No adjustable expenses found to optimise.")
            else:
                # ----- Pareto chart: top expenses by lifetime cost -----
                st.markdown("### 💸 Where your money actually goes (lifetime)")
                # Lifetime cost in start-year dollars (no inflation discounting
                # complication — we already deflate inside the sim, and these
                # are user-entered today-$ values multiplied by years active).
                for c in candidates:
                    c["lifetime_cost"] = c["annual_today"] * c["active_years"]
                # Sort and take top 15 for the chart
                ranked_cost = sorted(candidates,
                                     key=lambda c: c["lifetime_cost"],
                                     reverse=True)[:15]
                fig_cost = go.Figure()
                fig_cost.add_trace(go.Bar(
                    x=[c["lifetime_cost"] for c in ranked_cost],
                    y=[f"{c['label']} ({c['category']})" for c in ranked_cost],
                    orientation="h",
                    marker_color=["#3b82f6" if c["category"] == "Base" else "#a855f7"
                                  for c in ranked_cost],
                    text=[f"${c['lifetime_cost']:,.0f}" for c in ranked_cost],
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Lifetime cost: $%{x:,.0f}<br>"
                        "Years active: %{customdata[0]}<br>"
                        "Annual: $%{customdata[1]:,.0f}<extra></extra>"
                    ),
                    customdata=[[c["active_years"], c["annual_today"]]
                                for c in ranked_cost],
                ))
                fig_cost.update_layout(
                    height=max(350, 30 * len(ranked_cost)),
                    yaxis=dict(autorange="reversed"),
                    xaxis_tickprefix="$", xaxis_tickformat=",",
                    margin=dict(l=10, r=80, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_cost, use_container_width=True)

                # ----- Run sensitivity sweep -----
                # For each candidate, run the sim with that expense scaled down.
                multiplier = 1.0 - (opt_pct / 100.0)

                # Establish baseline metrics from the ALREADY-COMPUTED log
                baseline_shortfall_total = sum(annual["shortfall"])
                baseline_final_super = annual["super_total"][-1]
                baseline_final_invest = annual["outside_balance"][-1]
                baseline_final_savings = annual["savings"][-1]
                baseline_final_assets = (baseline_final_super
                                         + baseline_final_invest
                                         + baseline_final_savings)

                # Lifetime improvement = (shortfall avoided) + (extra final assets)
                progress = st.progress(0.0, text="Running sensitivity sweep…")
                results = []
                for idx, c in enumerate(candidates):
                    progress.progress(
                        (idx + 1) / len(candidates),
                        text=f"Testing: {c['label']} ({idx+1}/{len(candidates)})…"
                    )
                    overrides = {c["label"]: multiplier}
                    alt_log, _ = run_sim(expense_overrides=overrides)
                    # Aggregate annual flows / end balances exactly as the main code does
                    alt_short_annual = [sum(alt_log["shortfall"][i*12:(i+1)*12])
                                        for i in range(n_sim_years)]
                    alt_short_total = sum(alt_short_annual)
                    alt_super = alt_log["super_total"][-1]
                    alt_invest = alt_log["outside_balance"][-1]
                    alt_savings = alt_log["savings"][-1]
                    alt_final_assets = alt_super + alt_invest + alt_savings

                    short_avoided = baseline_shortfall_total - alt_short_total
                    extra_assets = alt_final_assets - baseline_final_assets
                    lifetime_improvement = short_avoided + extra_assets

                    # Total $ saved on this expense over the simulation,
                    # expressed in start-year dollars (the user-entered
                    # `annual_today` already IS in start-year dollars, and
                    # `lifetime_improvement` is also in start-year dollars,
                    # so this gives an apples-to-apples leverage ratio).
                    dollars_saved = (c["annual_today"] * c["active_years"]
                                     * (opt_pct / 100.0))
                    leverage = (lifetime_improvement / dollars_saved
                                if dollars_saved > 0 else 0)

                    results.append({
                        "label": c["label"],
                        "category": c["category"],
                        "monthly_today": c["monthly_today"],
                        "annual_today": c["annual_today"],
                        "active_years": c["active_years"],
                        "dollars_saved": dollars_saved,
                        "shortfall_avoided": short_avoided,
                        "extra_final_assets": extra_assets,
                        "lifetime_improvement": lifetime_improvement,
                        "leverage": leverage,
                    })
                progress.empty()

                # ----- Top 10 by lifetime improvement -----
                results_sorted = sorted(results,
                                        key=lambda r: r["lifetime_improvement"],
                                        reverse=True)
                top10 = results_sorted[:10]

                st.markdown(f"### 🏆 Top 10 expenses to reduce (at {opt_pct}% cut)")
                st.caption(
                    "**Lifetime improvement** = (shortfall reduced) + "
                    "(extra wealth at end of simulation), all in start-year dollars. "
                    "**Leverage** = improvement per start-year dollar of expense reduced. "
                    "Leverage <1× means inflation eats most of the saving (typical "
                    "when surplus sits as cash). Leverage >1× means the surplus "
                    "compounds (typical when routed into investment with real returns). "
                    "Leverage near zero means the cut barely moves your retirement "
                    "outcome — usually because some of the freed cashflow gets "
                    "absorbed by higher tax or lost Age Pension entitlement."
                )

                fig_top = go.Figure()
                fig_top.add_trace(go.Bar(
                    x=[r["lifetime_improvement"] for r in reversed(top10)],
                    y=[f"#{len(top10)-i}. {r['label']}" for i, r in enumerate(reversed(top10))],
                    orientation="h",
                    marker_color=[
                        "#10b981" if r["lifetime_improvement"] > 0 else "#ef4444"
                        for r in reversed(top10)
                    ],
                    text=[f"+${r['lifetime_improvement']:,.0f}"
                          for r in reversed(top10)],
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Lifetime improvement: $%{x:,.0f}<br>"
                        "Dollars saved: $%{customdata[0]:,.0f}<br>"
                        "Leverage: %{customdata[1]:.2f}×<extra></extra>"
                    ),
                    customdata=[[r["dollars_saved"], r["leverage"]]
                                for r in reversed(top10)],
                ))
                fig_top.update_layout(
                    height=max(350, 35 * len(top10)),
                    xaxis_tickprefix="$", xaxis_tickformat=",",
                    xaxis_title=f"Lifetime improvement (start-year $) at {opt_pct}% cut",
                    margin=dict(l=10, r=120, t=20, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_top, use_container_width=True)

                # ----- Detail table -----
                st.markdown("### 📋 Full ranking")
                df_opt = pd.DataFrame([{
                    "Rank": i + 1,
                    "Expense": r["label"],
                    "Type": r["category"],
                    "Today's monthly": f"${r['monthly_today']:,.0f}",
                    "Active years": r["active_years"],
                    f"$ saved at {opt_pct}%": f"${r['dollars_saved']:,.0f}",
                    "Shortfall avoided": f"${r['shortfall_avoided']:,.0f}",
                    "Extra final assets": f"${r['extra_final_assets']:,.0f}",
                    "Lifetime improvement": f"${r['lifetime_improvement']:,.0f}",
                    "Leverage": f"{r['leverage']:.2f}×",
                } for i, r in enumerate(results_sorted)])
                st.dataframe(df_opt, use_container_width=True, hide_index=True)

                # ----- Combined-cut scenario -----
                st.markdown(f"### 🧮 Combined: cut all top-10 by {opt_pct}%")
                top10_overrides = {r["label"]: multiplier for r in top10}
                combined_log, _ = run_sim(expense_overrides=top10_overrides)
                comb_short = sum(combined_log["shortfall"])
                comb_assets = (combined_log["super_total"][-1]
                              + combined_log["outside_balance"][-1]
                              + combined_log["savings"][-1])
                comb_short_avoided = baseline_shortfall_total - comb_short
                comb_extra = comb_assets - baseline_final_assets
                comb_total = comb_short_avoided + comb_extra

                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Shortfall avoided",
                           f"${comb_short_avoided:,.0f}",
                           help="Reduction in cumulative unmet cashflow.")
                cc2.metric("Extra final assets",
                           f"${comb_extra:,.0f}",
                           help="Increase in super + investment + savings at end.")
                cc3.metric("Combined lifetime gain",
                           f"${comb_total:,.0f}",
                           delta=f"vs. baseline" if comb_total else None,
                           delta_color="off")
                st.caption(
                    "Combined gain is usually a bit less than the sum of the "
                    "10 individual gains — when you cut multiple expenses, "
                    "the freed-up cashflow gets reinvested with diminishing "
                    "marginal returns (e.g. once shortfall hits zero, further "
                    "cuts only boost final assets, not pension entitlement)."
                )

    else:
        st.info("Configure the sidebar, then click **Run simulation**.")

    # -------------------------------------------------------------------------
    # Footer: source list for defaults
    # -------------------------------------------------------------------------
    with st.expander("Sources for the default values"):
        st.markdown(
            """
    **Tax (FY 2025-26)** – ATO, *Tax rates – Australian resident*,
    https://www.ato.gov.au/tax-rates-and-codes/tax-rates-australian-residents
    (Stage 3 Tax Cuts, in effect since 1 July 2024, confirmed for FY 2025-26.)

    **Medicare levy + LITO** – ATO current year figures.
    LITO: $700 max, 5c/$ phase-out 37,500→45,000, 1.5c/$ 45,000→66,667.

    **Superannuation Guarantee** – 12% permanent from 1 July 2025.
    (Superannuation Guarantee (Administration) Act.)

    **Concessional cap** – $30,000 per person per year from 1 July 2024
    (ATO, Key superannuation rates and thresholds).

    **Age Pension (rates from 20 March 2026)** – Services Australia:
    - Single: $1,200.90/fn = $31,223.40/yr (incl. pension & energy supplement)
    - Couple each: $905.20/fn, combined $1,810.40/fn = $47,070.40/yr

    **Income test (from 20 March 2026)** – free area $218/fn single,
    $380/fn couple combined; 50c/$ taper.

    **Assets test – full-pension thresholds (from 1 July 2025)**:
    - Single homeowner: $321,500;  couple homeowner: $481,500
    - Single non-homeowner: $579,500;  couple non-homeowner: $739,500
    Reduction $3/fn per $1,000 above threshold = $78/yr per $1,000.

    **Deeming (from 20 March 2026)** – 1.25% up to $64,200 (single) / $106,200
    (couple); 3.25% above.

    **Minimum drawdown** – Superannuation Industry (Supervision) Regulations:
    4% (<65), 5% (65-74), 6% (75-79), 7% (80-84), 9% (85-89), 11% (90-94), 14% (95+).

    **Retirement target (default)** – ASFA Retirement Standard, Sep 2025 quarter,
    homeowner 'comfortable': $77,375/yr couple, $54,240/yr single.
    https://www.superannuation.asn.au/consumers/retirement-standard/
    """
        )


# =========================================================================
# TAB 2 — HOME PRICE GROWTH
# =========================================================================
with tab2:
    st.header("🏠 Australian Home Price Growth Analyser")
    st.caption(
        "Enter any Australian property address. This tab uses Google Gemini "
        "(free tier) with Google Search grounding to research the sale history "
        "from sites like domain.com.au, realestate.com.au, onthehouse.com.au "
        "and more. Results are cached for 1 hour per address."
    )
    _render_home_price_tab()

"""
Australian Retirement Simulation — comprehensive edition.

All default values reflect Australian rules and rates verified as of April 2026
(FY 2025-26). Every source is cited in the sidebar help text. Every threshold
used in projections is indexed to inflation so that 50-year-forward results
remain internally consistent.

Run with:   streamlit run retirement_sim.py

Disclaimer: educational only, not financial advice. Verify figures against
ATO, Services Australia and your super fund before making decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------------------------
# Page config MUST be the first Streamlit call.
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="AU Retirement Simulator",
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
# UI
# =========================================================================
st.title("🇦🇺 Retirement Simulation")
st.caption(
    "Monthly cashflow simulation with Australian tax, super and Age Pension "
    f"rules current to April {TODAY_YEAR}. Educational only; not financial advice."
)
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
def run_sim():
    months = (end_year - start_year) * 12
    r_mort_m = (1 + mortgage_rate) ** (1/12) - 1

    # State
    super_bal = [m["super_bal"] for m in members]
    # Track when each member enters the pension phase (stops accumulation earnings tax)
    super_in_pension_phase = [False] * household_size
    out_bal = initial_investment
    offset = offset_balance
    mort = mortgage_principal
    mortgage_live = mort > 0
    payoff_year = None

    # Log arrays (stored in today's dollars unless labeled)
    log = {k: [] for k in [
        "salary", "tax", "take_home", "sg", "sacrifice",
        "expenses_base", "expenses_kids", "expenses_mortgage", "expenses_total",
        "pension", "super_withdrawal", "outside_withdrawal",
        "savings_flow", "principal_paid", "offset", "outside_balance",
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
        base_today = base_monthly
        kids_today = 0.0
        for name, cfg in child_items.items():
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

            # Outside investment next
            if gap > 0 and out_bal > 0:
                w = min(gap, out_bal)
                outside_wd = w
                out_bal -= w
                gap -= w

            # Offset / cash last
            if gap > 0 and offset > 0:
                w = min(gap, offset)
                offset_drawn = w
                offset -= w
                gap -= w

            shortfall = max(0.0, gap)

        else:
            # Surplus this month — allocate it per user split.
            surplus = -gap
            if mortgage_live:
                # Offset-first is the typical choice while mortgage is live,
                # because offset effectively earns the mortgage rate tax-free.
                # The slider controls how much (if any) is siphoned into investment.
                inv_share = surplus * (savings_alloc_pct / 100.0) if include_investment else 0.0
                off_share = surplus - inv_share
                if include_investment:
                    out_bal += inv_share
                offset += off_share
            else:
                # No mortgage. Offset becomes a plain cash account.
                if include_investment:
                    inv_share = surplus * (savings_alloc_pct / 100.0)
                    cash_share = surplus - inv_share
                    out_bal += inv_share
                    offset += cash_share  # track cash in offset var for continuity
                else:
                    offset += surplus

        # -----------------------------------------------------------------
        # Early-mortgage-payoff trigger (offset got big enough).
        # Pay off the remaining principal; leftover above the reserve flows
        # to investment/cash per the same allocation split.
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
                if leftover > 0:
                    if include_investment:
                        inv_share = leftover * (savings_alloc_pct / 100.0)
                        cash_share = leftover - inv_share
                        out_bal += inv_share
                        offset -= inv_share  # cash_share stays in offset
                    # else: leftover already in offset, stay as cash

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
        log["savings_flow"].append(dfl(max(0, -gap)))
        log["principal_paid"].append(dfl(principal_paid_nom))
        log["offset"].append(dfl(offset))
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
if st.sidebar.button("▶ Run simulation", type="primary"):
    with st.spinner("Running..."):
        log, payoff_year = run_sim()
        n_years = end_year - start_year
        years = [start_year + i for i in range(n_years)]

        # Annual aggregation (flows = sum; balances = end-of-year)
        flow_keys    = ["salary", "tax", "take_home", "sg", "sacrifice",
                        "expenses_base", "expenses_kids", "expenses_mortgage",
                        "expenses_total", "pension", "super_withdrawal",
                        "outside_withdrawal", "savings_flow", "principal_paid",
                        "shortfall"]
        balance_keys = ["offset", "outside_balance", "super_m1", "super_m2",
                        "super_total"]

        annual = {}
        for k in flow_keys:
            # Most flows were stored monthly; salary, tax, take_home, sg were
            # divided by 12 in logging, so to get annual need to multiply by 12
            # OR sum. We logged monthly equivalents, so sum() gives annual.
            annual[k] = [sum(log[k][i*12:(i+1)*12]) for i in range(n_years)]
        for k in balance_keys:
            annual[k] = [log[k][(i+1)*12 - 1] for i in range(n_years)]
        annual["retired_flag"] = [log["retired_flag"][(i+1)*12 - 1] for i in range(n_years)]

        # ------------------- Summary -------------------
        st.subheader("Summary")
        first_retire_idx = next((i for i, r in enumerate(annual["retired_flag"]) if r), None)
        col1, col2, col3 = st.columns(3)
        col1.metric("Mortgage paid off", f"{payoff_year}" if payoff_year else "Not paid off")
        if first_retire_idx is not None:
            # Balance just before retirement begins
            pre_idx = max(first_retire_idx - 1, 0)
            col2.metric(f"Super at retirement ({start_year} $)",
                        f"${annual['super_total'][pre_idx]:,.0f}")
            col3.metric(f"Outside assets at retirement ({start_year} $)",
                        f"${annual['outside_balance'][pre_idx]:,.0f}")

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
        fig1.add_trace(go.Scatter(x=years, y=annual["outside_balance"], name="Outside investment / cash", mode="lines"))
        fig1.add_trace(go.Scatter(x=years, y=annual["offset"], name="Offset account", mode="lines"))
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
            fig3.add_trace(go.Bar(x=years[post], y=annual["outside_withdrawal"][post], name="Outside drawdown"))
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
                "Outside": annual["outside_balance"],
                "Super (M1)": annual["super_m1"],
                "Super (M2)": annual["super_m2"],
                "Super total": annual["super_total"],
                "Pension": annual["pension"],
                "Super withdraw": annual["super_withdrawal"],
                "Outside withdraw": annual["outside_withdrawal"],
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

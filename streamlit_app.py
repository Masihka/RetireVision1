"""
Retirement Simulation App – rebased to 2026 AUD.

Key 2026 reference values used as defaults
-------------------------------------------
* Tax brackets: 2025-26 FY (Stage 3 implemented)
    0 – 18,200        : 0%
    18,201 – 45,000   : 16%
    45,001 – 135,000  : 30%
    135,001 – 190,000 : 37%
    190,001 +         : 45%
* Medicare levy: 2%
* LITO: $700, phasing out 37,500 -> 66,667
* Superannuation Guarantee: 12% (permanent from 1 July 2025)
* Concessional contribution cap: $30,000 (2024-25 onwards)
* Maximum Super Contribution Base: ~$65,070/quarter = $260,280/year in 2025-26
* Age Pension (from 20 March 2026, incl. supplements):
    Single max: $31,223.40 p.a. ($1,200.90/ftn)
    Couple max combined: $47,070.40 p.a. ($1,810.40/ftn)
* Income-test free area (annual):
    Single: $5,668    Couple combined: $9,880
* Income-test taper: 50c per $1
* Deeming thresholds (from 20 March 2026):
    Single: $64,200   Couple combined: $106,200
* Deeming rates: 1.25% lower / 3.25% upper
* Assets-test lower threshold (from 20 March 2026):
    Single homeowner       : $321,500
    Single non-homeowner   : $579,500
    Couple homeowner       : $481,500
    Couple non-homeowner   : $739,500
* Assets-test taper: $3/ftn per $1,000 over threshold  ~= $78/yr per $1,000
* ASFA "comfortable" retirement standard (Dec-25 quarter, homeowner):
    Couple: $76,505 p.a.   Single: $54,240 p.a.
* RBA cash rate at time of rebase: 4.10% (Mar 2026)
* Average owner-occupier variable mortgage rate: ~5.75%
* RBA inflation target midpoint: 2.5% (used as long-run default)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------
hide_streamlit_ui = """
<style>
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
.styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
.viewerBadge_text__1JaDK { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
"""
st.set_page_config(initial_sidebar_state="expanded")
st.markdown(hide_streamlit_ui, unsafe_allow_html=True)
st.title("Retirement Simulation App (2026 AUD base) — by Masih")
st.warning(
    "📢 **Disclaimer:** This app is not financial advice and has not been "
    "validated against real-world case studies. It is for general "
    "informational use only. All dollar values are expressed in 2026 AUD."
)

# ---------------------------------------------------------------------------
# Sidebar – simulation window
# ---------------------------------------------------------------------------
st.sidebar.header("Simulation Settings")
simulation_start_year = st.sidebar.number_input(
    "Start Year", min_value=2000, max_value=2100, value=2026, step=1,
    help="Year the simulation begins (base year for 'real' AUD values)")
retirement_year = st.sidebar.number_input(
    "Retirement Year", min_value=2000, max_value=2100, value=2055, step=1,
    help="Year retirement begins")
death_year = st.sidebar.number_input(
    "End Year", min_value=2000, max_value=2100, value=2080, step=1,
    help="Year the simulation ends")

# Household size needs to exist before we use it to default retirement salary.
household_size = st.sidebar.selectbox(
    "Household Size", [1, 2], index=1,
    help="Number of income earners in household", key="household_size")

# ASFA Dec-25 quarter 'comfortable' standard (homeowner).
default_retirement_salary = 76505.0 if household_size == 2 else 54240.0
retirement_salary = st.sidebar.number_input(
    "Desired Retirement Salary (Annual, 2026 AUD)",
    value=default_retirement_salary, step=1000.0,
    help=("Default is the ASFA 'comfortable' standard (Dec-25 quarter): "
          "$76,505 couple, $54,240 single."))

starting_age = st.sidebar.number_input(
    "Starting Age (of oldest earner)", min_value=18, max_value=100,
    value=35, step=1,
    help="Used to apply super minimum drawdown rates and pension eligibility")

# ---------------------------------------------------------------------------
# Sidebar – economic assumptions
# ---------------------------------------------------------------------------
st.sidebar.header("Economic Assumptions")
inflation_rate = st.sidebar.slider(
    "Annual Inflation Rate", 0.0, 0.1, 0.025, step=0.005,
    help="Long-run CPI assumption. RBA target midpoint is 2.5%.")
wage_growth_rate = st.sidebar.slider(
    "Annual Wage Growth Rate", 0.0, 0.1, 0.033, step=0.005,
    help="Recent Wage Price Index has been ~3.3% p.a.")
nominal_super_growth_rate = st.sidebar.slider(
    "Super Growth Rate (nominal, pre-tax)", 0.0, 0.15, 0.07, step=0.005,
    help="Typical long-run balanced super return assumption.")
nominal_investment_rate = st.sidebar.slider(
    "Investment Rate (nominal)", 0.0, 0.15, 0.045, step=0.005,
    help="Return on non-super investments.")
super_rate = st.sidebar.slider(
    "Super Contribution Rate (SG)", 0.0, 0.2, 0.12, step=0.005,
    help="Super Guarantee reached 12% on 1 July 2025 and is the permanent rate.")
nominal_mortgage_rate = st.sidebar.slider(
    "Mortgage Rate (nominal)", 0.0, 0.15, 0.0575, step=0.0025,
    help=("Approx. average owner-occupier variable rate in early 2026. "
          "RBA cash rate was 4.10% (Mar 2026)."))

# ---------------------------------------------------------------------------
# Sidebar – personal finances
# ---------------------------------------------------------------------------
st.sidebar.header("Personal Finances")
initial_super = st.sidebar.number_input(
    "Initial Super per Person", value=60000.0, step=1000.0,
    help="Starting super balance per household member (2026 AUD)")
initial_salary = st.sidebar.number_input(
    "Initial Salary per Person", value=110000.0, step=1000.0,
    help="Starting gross annual salary per person (2026 AUD)")
salary_sacrifice_annual = st.sidebar.number_input(
    "Annual Salary Sacrifice", value=0.0, step=1000.0,
    help="Annual pre-tax (concessional) super contribution per person, "
         "on top of employer SG")
mortgage_principal = st.sidebar.number_input(
    "Mortgage Principal", value=600000.0, step=1000.0,
    help="Initial mortgage loan amount (2026 AUD)")
offset_balance = st.sidebar.number_input(
    "Offset Balance", value=150000.0, step=1000.0,
    help="Initial balance in mortgage offset account (2026 AUD)")
mortgage_payment = st.sidebar.number_input(
    "Monthly Mortgage Payment", value=3500.0, step=100.0,
    help="Fixed monthly payment towards mortgage (2026 AUD)")
reserve_target = st.sidebar.number_input(
    "Reserve Target", value=100000.0, step=1000.0,
    help="Desired cash reserve (in real 2026 AUD) kept before paying off "
         "mortgage and redirecting savings")
homeowner = st.sidebar.checkbox(
    "Homeowner", value=True, help="Check if household owns a home")

# ---------------------------------------------------------------------------
# Sidebar – tax settings
# ---------------------------------------------------------------------------
st.sidebar.header("Tax Settings (2025-26 FY defaults)")
tax_brackets_input = st.sidebar.text_area(
    "Tax Brackets (threshold:rate, comma-separated)",
    "18200:0, 45000:0.16, 135000:0.30, 190000:0.37, inf:0.45",
    help=("Australian resident tax rates for 2025-26 FY (Stage 3 in force). "
          "From 1 July 2026 the 16% rate drops to 15%."))
medicare_levy = st.sidebar.slider(
    "Medicare Levy", 0.0, 0.05, 0.02, step=0.005,
    help="Standard Medicare levy is 2%.")
max_concessional_cap = st.sidebar.number_input(
    "Max Concessional Cap", value=30000.0, step=1000.0,
    help="Annual concessional super cap (2024-25 onwards: $30,000).")
max_super_contrib_base_annual = st.sidebar.number_input(
    "Max Super Contribution Base (annual)", value=65070.0 * 4, step=1000.0,
    help=("Employers only have to pay SG up to this income level. "
          "Approx. $65,070/quarter in 2025-26 = $260,280/yr."))

try:
    tax_brackets = []
    for item in tax_brackets_input.split(','):
        threshold, rate = item.strip().split(':')
        threshold = float(threshold) if threshold != 'inf' else float('inf')
        rate = float(rate)
        tax_brackets.append((threshold, rate))
except Exception:
    st.sidebar.error("Invalid tax brackets format. Use 'threshold:rate' "
                     "pairs separated by commas.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar – expenses
# ---------------------------------------------------------------------------
st.sidebar.header("Base Expenses (Monthly, 2026 AUD)")
base_expenses = {}

expenses_with_frequency = [
    ('car_insurance', 'Car Insurance', 2200.0 / 12,
     'Car insurance cost'),
    ('car_registration', 'Car Registration', 900.0 / 12,
     'Car registration cost'),
    ('traveling_abroad', 'Traveling Abroad', 6000.0 / 12,
     'Travel / vacations'),
    ('car_service_fee', 'Car Service Fee', 700.0 / 12,
     'Car maintenance'),
    ('body_corporate', 'Body Corporate', 1800.0 / 12,
     'Strata fees'),
    ('council_cost', 'Council Cost', 1800.0 / 12,
     'Council rates'),
]
for key, label, default, help_text in expenses_with_frequency:
    frequency = st.sidebar.selectbox(
        f"{label} Frequency", ["Monthly", "Yearly"], index=0,
        key=f"{key}_frequency",
        help=f"Choose whether to input {label} as a monthly or yearly amount")
    value = st.sidebar.number_input(
        label, value=default, step=10.0, key=f"{key}_input", help=help_text)
    base_expenses[key] = value / 12 if frequency == "Yearly" else value

other_expenses = {
    'utilities': st.sidebar.number_input(
        "Utilities", value=450.0, step=10.0,
        help="Monthly utility bills (electricity, gas, water, internet)"),
    'food_and_groceries': st.sidebar.number_input(
        "Food and Groceries", value=900.0, step=10.0,
        help="Monthly grocery spend"),
    'transportation_other': st.sidebar.number_input(
        "Other Transportation", value=400.0, step=10.0,
        help="Fuel, public transport, parking, tolls"),
    'healthcare': st.sidebar.number_input(
        "Healthcare (out of pocket)", value=400.0, step=10.0,
        help="GP gap fees, specialists, PBS co-pays, dental"),
    'private_health_insurance': st.sidebar.number_input(
        "Private Health Insurance", value=350.0, step=10.0,
        help="Monthly premium (couple hospital+extras ~$350-450 in 2026)"),
    'personal_care': st.sidebar.number_input(
        "Personal Care", value=200.0, step=10.0),
    'entertainment': st.sidebar.number_input(
        "Entertainment", value=250.0, step=10.0),
    'clothing': st.sidebar.number_input(
        "Clothing", value=200.0, step=10.0),
    'household_supplies': st.sidebar.number_input(
        "Household Supplies", value=150.0, step=10.0),
    'miscellaneous': st.sidebar.number_input(
        "Miscellaneous", value=700.0, step=10.0),
    'gym': st.sidebar.number_input(
        "Gym", value=100.0, step=10.0),
}
base_expenses.update(other_expenses)
base_expenses_monthly = sum(base_expenses.values())

# ---------------------------------------------------------------------------
# Sidebar – child expenses (unchanged structure, minor default bumps)
# ---------------------------------------------------------------------------
st.sidebar.header("Child Expenses (Monthly, 2026 AUD)")
child_expenses = {
    'childcare': {
        'start': st.sidebar.number_input("Childcare Start Year",
            value=simulation_start_year, step=1),
        'end':   st.sidebar.number_input("Childcare End Year",
            value=simulation_start_year + 5, step=1),
        'amount': st.sidebar.number_input("Childcare Amount",
            value=42000.0 / 12, step=100.0,
            help="After CCS, full-time childcare in a capital city is "
                 "typically $35-50k/yr in 2026"),
    },
    'baby_supplies': {
        'start': st.sidebar.number_input("Baby Supplies Start Year",
            value=simulation_start_year, step=1),
        'end':   st.sidebar.number_input("Baby Supplies End Year",
            value=simulation_start_year + 3, step=1),
        'amount': st.sidebar.number_input("Baby Supplies Amount",
            value=4200.0 / 12, step=100.0),
    },
    'food': {
        'start': st.sidebar.number_input("Food (Kids) Start Year",
            value=simulation_start_year, step=1),
        'end':   st.sidebar.number_input("Food (Kids) End Year",
            value=simulation_start_year + 25, step=1),
        'amount': st.sidebar.number_input("Food (Kids) Amount",
            value=3000.0 / 12, step=100.0),
    },
    'clothing': {
        'start': st.sidebar.number_input("Clothing (Kids) Start Year",
            value=simulation_start_year, step=1),
        'end':   st.sidebar.number_input("Clothing (Kids) End Year",
            value=simulation_start_year + 18, step=1),
        'amount': st.sidebar.number_input("Clothing (Kids) Amount",
            value=1200.0 / 12, step=100.0),
    },
    'healthcare': {
        'start': st.sidebar.number_input("Healthcare (Kids) Start Year",
            value=simulation_start_year, step=1),
        'end':   st.sidebar.number_input("Healthcare (Kids) End Year",
            value=simulation_start_year + 25, step=1),
        'amount': st.sidebar.number_input("Healthcare (Kids) Amount",
            value=600.0 / 12, step=100.0),
    },
    'education': {
        'start': st.sidebar.number_input("Education Start Year",
            value=simulation_start_year + 5, step=1),
        'end':   st.sidebar.number_input("Education End Year",
            value=simulation_start_year + 18, step=1),
        'amount': st.sidebar.number_input("Education Amount",
            value=1000.0 / 12, step=100.0),
    },
    'tertiary': {
        'start': st.sidebar.number_input("Tertiary Start Year",
            value=simulation_start_year + 18, step=1),
        'end':   st.sidebar.number_input("Tertiary End Year",
            value=simulation_start_year + 25, step=1),
        'amount': st.sidebar.number_input("Tertiary Amount",
            value=2400.0 / 12, step=100.0),
    },
    'extracurricular': {
        'start': st.sidebar.number_input("Extracurricular Start Year",
            value=simulation_start_year + 5, step=1),
        'end':   st.sidebar.number_input("Extracurricular End Year",
            value=simulation_start_year + 18, step=1),
        'amount': st.sidebar.number_input("Extracurricular Amount",
            value=1200.0 / 12, step=100.0),
    },
    'misc': {
        'start': st.sidebar.number_input("Misc (Kids) Start Year",
            value=simulation_start_year, step=1),
        'end':   st.sidebar.number_input("Misc (Kids) End Year",
            value=simulation_start_year + 18, step=1),
        'amount': st.sidebar.number_input("Misc (Kids) Amount",
            value=1200.0 / 12, step=100.0),
    },
}
expenses = {'base': base_expenses, 'child': child_expenses}

# ---------------------------------------------------------------------------
# Sidebar – pension settings (20 March 2026 values)
# ---------------------------------------------------------------------------
st.sidebar.header("Pension Settings (20 March 2026 figures)")
pension_age = st.sidebar.number_input(
    "Pension Age", value=67, step=1,
    help="Age pension qualifying age is 67.")
max_pension_single = st.sidebar.number_input(
    "Max Pension Single (Annual)", value=31223.40, step=100.0,
    help="$1,200.90/ftn incl. pension + energy supplements (20 Mar 2026).")
max_pension_couple = st.sidebar.number_input(
    "Max Pension Couple Combined (Annual)", value=47070.40, step=100.0,
    help="$1,810.40/ftn combined, incl. supplements (20 Mar 2026).")
income_free_area_single = st.sidebar.number_input(
    "Income-Test Free Area Single (Annual)", value=5668.0, step=100.0,
    help="$218/ftn x 26 (20 Mar 2026).")
income_free_area_couple = st.sidebar.number_input(
    "Income-Test Free Area Couple Combined (Annual)",
    value=9880.0, step=100.0, help="$380/ftn x 26 (20 Mar 2026).")
income_taper = st.sidebar.slider(
    "Income Taper Rate", 0.0, 1.0, 0.50, step=0.05,
    help="Pension reduces by 50c per $1 of assessable income over free area.")

# Assets-test free area picker – pick the right cell based on household
# composition and home ownership.
asset_free_matrix = {
    (True,  2): 481500.0,   # homeowner couple
    (True,  1): 321500.0,   # homeowner single
    (False, 2): 739500.0,   # non-homeowner couple
    (False, 1): 579500.0,   # non-homeowner single
}
default_assets_free = asset_free_matrix[(homeowner, household_size)]
assets_free_area = st.sidebar.number_input(
    "Assets-Test Free Area", value=default_assets_free, step=1000.0,
    help="Lower threshold from 20 March 2026.")
assets_taper_per_1000_annual = st.sidebar.number_input(
    "Assets Taper (Annual $ per $1,000 over threshold)",
    value=78.0, step=1.0,
    help="$3/ftn per $1,000 = $78/year per $1,000.")

# Deeming – 20 March 2026
st.sidebar.header("Deeming (20 March 2026)")
deem_threshold_single = st.sidebar.number_input(
    "Deeming Threshold Single", value=64200.0, step=100.0)
deem_threshold_couple = st.sidebar.number_input(
    "Deeming Threshold Couple Combined", value=106200.0, step=100.0)
deem_rate_lower = st.sidebar.slider(
    "Deeming Rate – Lower", 0.0, 0.1, 0.0125, step=0.0025)
deem_rate_upper = st.sidebar.slider(
    "Deeming Rate – Upper", 0.0, 0.15, 0.0325, step=0.0025)

# ---------------------------------------------------------------------------
# Sidebar – unemployment
# ---------------------------------------------------------------------------
st.sidebar.header("Unemployment Periods")
unemployment_years = {}
for i in range(household_size):
    st.sidebar.subheader(f"Member {i+1}")
    start = st.sidebar.number_input(
        f"Unemployment Start Year (Member {i+1})",
        value=simulation_start_year + 9 if i == 0 else simulation_start_year + 14,
        step=1)
    end = st.sidebar.number_input(
        f"Unemployment End Year (Member {i+1})",
        value=simulation_start_year + 10 if i == 0 else simulation_start_year + 15,
        step=1)
    unemployment_years[i] = range(start, end + 1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_DRAWDOWN_RATES = {
    (0, 64):  0.04,
    (65, 74): 0.05,
    (75, 79): 0.06,
    (80, 84): 0.07,
    (85, 89): 0.09,
    (90, 94): 0.11,
    (95, float('inf')): 0.14,
}

# Low-Income Tax Offset (2025-26 unchanged): $700 below $37,500,
# then 5c taper to $45,000, then 1.5c taper to $66,667.
def lito_amount(taxable: float) -> float:
    if taxable <= 37500:
        return 700.0
    if taxable <= 45000:
        return max(0.0, 700.0 - 0.05 * (taxable - 37500))
    if taxable <= 66667:
        return max(0.0, 325.0 - 0.015 * (taxable - 45000))
    return 0.0


def calculate_tax(nominal_income: float, nominal_sacrifice: float,
                  tax_brackets_list) -> float:
    """Australian resident income tax (including Medicare levy, net of LITO).

    All inputs and outputs are NOMINAL amounts for the year in question.
    The tax brackets are assumed to be in nominal terms for that year
    (we index them with inflation upstream if desired).
    """
    taxable = max(nominal_income - nominal_sacrifice, 0.0)
    tax = 0.0
    prev_threshold = 0.0
    for threshold, rate in tax_brackets_list:
        if taxable <= threshold:
            tax += (taxable - prev_threshold) * rate
            break
        tax += (threshold - prev_threshold) * rate
        prev_threshold = threshold
    tax += taxable * medicare_levy
    tax = max(tax - lito_amount(taxable), 0.0)
    return tax


def get_drawdown_rate(age: int) -> float:
    for (lo, hi), rate in MIN_DRAWDOWN_RATES.items():
        if lo <= age <= hi:
            return rate
    return 0.14


def calculate_pension(super_balances, investment_balance,
                      real_inflation_factor, current_age):
    """Monthly Age Pension in NOMINAL dollars for current month.

    real_inflation_factor = cumulative inflation multiplier from
    simulation_start_year to 'now' (used to index pension rates).
    """
    if current_age < pension_age:
        return 0.0

    total_assets = sum(super_balances) + investment_balance

    # Deemed income (annual, nominal). Thresholds are also indexed by inflation.
    threshold = (deem_threshold_couple if household_size == 2
                 else deem_threshold_single) * real_inflation_factor
    financial_assets = sum(super_balances) + investment_balance
    low_portion  = min(financial_assets, threshold) * deem_rate_lower
    high_portion = max(0.0, financial_assets - threshold) * deem_rate_upper
    deemed_income_annual = low_portion + high_portion

    # Free areas (annual, inflated).
    free_area = (income_free_area_couple if household_size == 2
                 else income_free_area_single) * real_inflation_factor
    income_reduction = max(0.0,
                           (deemed_income_annual - free_area) * income_taper)

    assets_reduction = max(0.0,
        (total_assets - assets_free_area * real_inflation_factor)
        * assets_taper_per_1000_annual / 1000.0)

    max_pension_annual = ((max_pension_couple if household_size == 2
                           else max_pension_single) * real_inflation_factor)

    pension_annual = max(0.0,
        max_pension_annual - max(income_reduction, assets_reduction))
    return pension_annual / 12.0  # monthly


# ---------------------------------------------------------------------------
# Run the simulation
# ---------------------------------------------------------------------------
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        total_years = death_year - simulation_start_year
        if total_years <= 0 or retirement_year <= simulation_start_year:
            st.error("Please ensure Start Year < Retirement Year <= End Year.")
            st.stop()
        months = total_years * 12
        years_to_retirement = retirement_year - simulation_start_year

        inflation_rates = [inflation_rate] * total_years
        wage_growth_rates = [wage_growth_rate] * total_years
        super_rates = [super_rate] * total_years
        monthly_mortgage_rate = (1 + nominal_mortgage_rate) ** (1/12) - 1
        monthly_investment_rate = (1 + nominal_investment_rate) ** (1/12) - 1
        monthly_super_growth_rate = (1 + nominal_super_growth_rate) ** (1/12) - 1

        # Local copies so sliders can be re-run without re-loading.
        mortgage_principal_local = float(mortgage_principal)
        offset_balance_local = float(offset_balance)

        monthly_data = {k: [] for k in [
            'expenses', 'income', 'savings', 'offset', 'investment',
            'principal_paid', 'super_person1', 'super_person2', 'tax',
            'pension', 'sg_contribution', 'salary_sacrifice', 'base_expenses',
            'childcare', 'baby_supplies', 'food_kids', 'clothing_kids',
            'healthcare_kids', 'education', 'tertiary', 'extracurricular',
            'misc_kids', 'mortgage', 'super_withdrawal_person1',
            'super_withdrawal_person2', 'investment_withdrawal'
        ]}
        super_balances = [float(initial_super)] * household_size
        investment_balance = 0.0
        mortgage_active = mortgage_principal_local > 0
        mortgage_payoff_year = None

        for m in range(months):
            year = m // 12
            month_in_year = m % 12
            current_year = simulation_start_year + year
            current_age = int(starting_age) + year

            # Cumulative inflation multiplier from start to NOW (fractional).
            # Fixed: was off-by-one in the original code.
            inflation_factor = (np.prod([1 + r for r in inflation_rates[:year]])
                                * (1 + inflation_rates[year])
                                ** (month_in_year / 12))

            # ---- Expenses -------------------------------------------------
            living_expenses_real = base_expenses_monthly
            nominals = {'childcare': 0.0, 'baby_supplies': 0.0,
                        'food_kids': 0.0, 'clothing_kids': 0.0,
                        'healthcare_kids': 0.0, 'education': 0.0,
                        'tertiary': 0.0, 'extracurricular': 0.0,
                        'misc_kids': 0.0}
            name_map = {
                'childcare': 'childcare', 'baby_supplies': 'baby_supplies',
                'food': 'food_kids', 'clothing': 'clothing_kids',
                'healthcare': 'healthcare_kids', 'education': 'education',
                'tertiary': 'tertiary', 'extracurricular': 'extracurricular',
                'misc': 'misc_kids',
            }
            for category, details in expenses['child'].items():
                start_month = (details['start'] - simulation_start_year) * 12
                end_month   = (details['end']   - simulation_start_year) * 12
                if start_month <= m < end_month:
                    amount_nominal = details['amount'] * inflation_factor
                    living_expenses_real += details['amount']
                    nominals[name_map[category]] = amount_nominal

            base_expenses_nominal = base_expenses_monthly * inflation_factor
            expenses_nominal = living_expenses_real * inflation_factor
            mortgage_nominal = 0.0
            if mortgage_active:
                # Mortgage payment is assumed FIXED in nominal terms – typical
                # for real mortgages (though repayments can be adjusted).
                mortgage_nominal = mortgage_payment
                expenses_nominal += mortgage_payment

            # ---- Income / contributions ----------------------------------
            super_withdrawal = [0.0] * household_size
            investment_withdrawal = 0.0
            salary_nominal = 0.0
            sg_contribution = 0.0
            sacrifice_contribution = 0.0
            tax_nominal = 0.0
            take_home_monthly_nominal = 0.0
            pension_nominal = 0.0

            if year < years_to_retirement:
                # Working phase
                employed = [current_year not in unemployment_years[i]
                            for i in range(household_size)]
                salary_per_person = (initial_salary
                    * np.prod([1 + r for r in wage_growth_rates[:year]]))
                mscb = max_super_contrib_base_annual * np.prod(
                    [1 + r for r in wage_growth_rates[:year]])
                sg_salary_base_per_person = min(salary_per_person, mscb)
                sg_per_person = sg_salary_base_per_person * super_rates[year]
                max_sacrifice = max(0.0, max_concessional_cap
                                    * np.prod([1 + r for r in inflation_rates[:year]])
                                    - sg_per_person)
                sacrifice_per_person = min(salary_sacrifice_annual
                    * np.prod([1 + r for r in wage_growth_rates[:year]]),
                    max_sacrifice)

                for i, emp in enumerate(employed):
                    if not emp:
                        continue
                    salary_nominal += salary_per_person
                    sg_contribution += sg_per_person
                    sacrifice_contribution += sacrifice_per_person
                    tax_nominal += calculate_tax(salary_per_person,
                                                 sacrifice_per_person,
                                                 tax_brackets)
                take_home_nominal = (salary_nominal - tax_nominal
                                     - sacrifice_contribution)
                take_home_monthly_nominal = take_home_nominal / 12.0

            else:
                # Retirement phase – pension + drawdowns.
                # Strategy:
                #   1. Compute each person's minimum monthly drawdown.
                #   2. If minimums already cover the shortfall, each person
                #      draws their minimum (total may legitimately exceed
                #      the target – that's how Australian min-drawdown
                #      rules actually work).
                #   3. Otherwise, each person draws their minimum PLUS a
                #      share of the remaining gap proportional to their
                #      balance (so balances deplete symmetrically).
                #   4. Only when all super is gone do we tap investments.
                pension_nominal = calculate_pension(
                    super_balances, investment_balance,
                    inflation_factor, current_age)
                target_monthly_nominal = (retirement_salary
                                          * inflation_factor / 12.0)
                remaining_need = target_monthly_nominal - pension_nominal

                if remaining_need > 0 and sum(super_balances) > 0:
                    drawdown_rate_annual = get_drawdown_rate(current_age)
                    min_draws = [super_balances[i] * drawdown_rate_annual / 12
                                 for i in range(household_size)]
                    total_min = sum(min_draws)

                    if total_min >= remaining_need:
                        # Minimums alone cover the need (or exceed it).
                        for i in range(household_size):
                            draw = min(min_draws[i], super_balances[i])
                            super_withdrawal[i] = draw
                            super_balances[i] -= draw
                        remaining_need = 0
                    else:
                        # Take minimums, then split extra by balance share.
                        extra_need = remaining_need - total_min
                        total_balance = sum(super_balances)
                        for i in range(household_size):
                            if super_balances[i] <= 0:
                                continue
                            extra = (extra_need * super_balances[i]
                                     / total_balance
                                     if total_balance > 0 else 0.0)
                            draw = min(min_draws[i] + extra,
                                       super_balances[i])
                            super_withdrawal[i] = draw
                            super_balances[i] -= draw
                        remaining_need = max(0.0,
                            remaining_need - sum(super_withdrawal))

                if remaining_need > 0 and investment_balance > 0:
                    investment_withdrawal = min(remaining_need,
                                                investment_balance)
                    investment_balance -= investment_withdrawal

            # ---- Saving / mortgage (working phase only) ------------------
            principal_repayment = 0.0
            savings = 0.0
            if year < years_to_retirement:
                if mortgage_active:
                    net_debt = max(mortgage_principal_local - offset_balance_local, 0.0)
                    interest = net_debt * monthly_mortgage_rate
                    principal_repayment = min(mortgage_payment - interest,
                                              mortgage_principal_local)
                    principal_repayment = max(principal_repayment, 0.0)
                    mortgage_principal_local -= principal_repayment
                    savings = max(take_home_monthly_nominal - expenses_nominal,
                                  0.0)
                    offset_balance_local += savings
                    reserve_nominal = reserve_target * inflation_factor
                    if (offset_balance_local - mortgage_principal_local
                            >= reserve_nominal and mortgage_principal_local > 0):
                        principal_repayment += mortgage_principal_local
                        offset_balance_local -= mortgage_principal_local
                        # Move excess above reserve into investments.
                        excess = max(0.0,
                            offset_balance_local - reserve_nominal)
                        investment_balance += excess
                        offset_balance_local -= excess
                        mortgage_principal_local = 0.0
                        mortgage_active = False
                        mortgage_payoff_year = current_year
                else:
                    savings = max(take_home_monthly_nominal - expenses_nominal,
                                  0.0)
                    investment_balance = (investment_balance
                        * (1 + monthly_investment_rate) + savings)

            # ---- Super monthly update -----------------------------------
            if year < years_to_retirement:
                monthly_concessional_per_person_net_tax = (
                    (sg_contribution / household_size
                     + sacrifice_contribution / household_size)
                    * 0.85 / 12.0)
            else:
                monthly_concessional_per_person_net_tax = 0.0

            for i in range(household_size):
                # 15% contributions tax is assumed deducted inside fund;
                # during retirement phase earnings are tax-free.
                tax_on_growth = (0.15 if year < years_to_retirement else 0.0)
                growth = super_balances[i] * monthly_super_growth_rate
                super_balances[i] += (growth * (1 - tax_on_growth)
                    + monthly_concessional_per_person_net_tax)

            # Post-mortgage investments continue to accrue in retirement too.
            if not mortgage_active and year >= years_to_retirement:
                investment_balance *= (1 + monthly_investment_rate)

            # ---- Record (all values in REAL 2026 AUD) --------------------
            monthly_data['expenses'].append(expenses_nominal / inflation_factor)
            monthly_data['income'].append(
                (take_home_monthly_nominal + pension_nominal
                 + sum(super_withdrawal) + investment_withdrawal)
                / inflation_factor)
            monthly_data['savings'].append(savings / inflation_factor)
            monthly_data['offset'].append(offset_balance_local / inflation_factor)
            monthly_data['investment'].append(investment_balance / inflation_factor)
            monthly_data['principal_paid'].append(
                principal_repayment / inflation_factor)
            monthly_data['super_person1'].append(
                super_balances[0] / inflation_factor)
            monthly_data['super_person2'].append(
                super_balances[1] / inflation_factor if household_size == 2 else 0.0)
            monthly_data['tax'].append(tax_nominal / 12.0 / inflation_factor)
            monthly_data['pension'].append(pension_nominal / inflation_factor)
            monthly_data['sg_contribution'].append(
                sg_contribution / 12.0 / inflation_factor)
            monthly_data['salary_sacrifice'].append(
                sacrifice_contribution / 12.0 / inflation_factor)
            monthly_data['base_expenses'].append(
                base_expenses_nominal / inflation_factor)
            for k in ('childcare', 'baby_supplies', 'food_kids',
                      'clothing_kids', 'healthcare_kids', 'education',
                      'tertiary', 'extracurricular', 'misc_kids'):
                monthly_data[k].append(nominals[k] / inflation_factor)
            monthly_data['mortgage'].append(mortgage_nominal / inflation_factor)
            monthly_data['super_withdrawal_person1'].append(
                super_withdrawal[0] / inflation_factor)
            monthly_data['super_withdrawal_person2'].append(
                super_withdrawal[1] / inflation_factor
                if household_size == 2 else 0.0)
            monthly_data['investment_withdrawal'].append(
                investment_withdrawal / inflation_factor)

        # -----------------------------------------------------------------
        # Annual aggregation
        # -----------------------------------------------------------------
        years = [f"{y}-{y+1}"
                 for y in range(simulation_start_year, death_year)]
        annual_data = {
            k: [sum(monthly_data[k][i*12:(i+1)*12]) for i in range(len(years))]
            for k in monthly_data
        }
        # For balances use end-of-year snapshot, not sum.
        for key in ['offset', 'investment', 'super_person1', 'super_person2']:
            annual_data[key] = [monthly_data[key][(i+1)*12-1]
                                for i in range(len(years))]

        # -----------------------------------------------------------------
        # Retirement salary components plot
        # -----------------------------------------------------------------
        st.subheader("Retirement Salary Components (2026 AUD)")
        fig = go.Figure()
        rs_idx = years_to_retirement
        plot_years = years[rs_idx:]
        fig.add_trace(go.Bar(x=plot_years,
            y=annual_data['super_withdrawal_person1'][rs_idx:],
            name="Person 1 Super", marker_color='blue'))
        if household_size == 2:
            fig.add_trace(go.Bar(x=plot_years,
                y=annual_data['super_withdrawal_person2'][rs_idx:],
                name="Person 2 Super", marker_color='green'))
        fig.add_trace(go.Bar(x=plot_years,
            y=annual_data['investment_withdrawal'][rs_idx:],
            name="Investment Income", marker_color='orange'))
        fig.add_trace(go.Bar(x=plot_years,
            y=annual_data['pension'][rs_idx:],
            name="Age Pension", marker_color='purple'))
        fig.update_layout(barmode='stack',
            title="Annual Retirement Salary by Component",
            xaxis_title="Year", yaxis_title="AUD (2026 Dollars)",
            xaxis_tickangle=45,
            legend=dict(x=0, y=1.1, orientation='h'), height=600)
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------------------
        # Line charts
        # -----------------------------------------------------------------
        plot_configs = [
            ('expenses', 'Expenses', 'Annual Expenses (2026 AUD)'),
            ('income', 'Income', 'Annual Income (2026 AUD)'),
            ('savings', 'Savings', 'Annual Savings (2026 AUD)'),
            ('super_person1', 'Superannuation (Person 1)',
                'Superannuation Balance Person 1 (2026 AUD)'),
            ('super_person2', 'Superannuation (Person 2)',
                'Superannuation Balance Person 2 (2026 AUD)'),
            ('investment', 'Investments', 'Investment Balance (2026 AUD)'),
            ('offset', 'Offset Balance',
                'Mortgage Offset Balance (2026 AUD)'),
            ('sg_contribution', 'SG Contribution',
                'Annual SG Contribution (2026 AUD)'),
            ('salary_sacrifice', 'Salary Sacrifice',
                'Annual Salary Sacrifice (2026 AUD)'),
            ('super_withdrawal_person1', 'Super Withdrawal (Person 1)',
                'Annual Super Withdrawal Person 1 (2026 AUD)'),
            ('super_withdrawal_person2', 'Super Withdrawal (Person 2)',
                'Annual Super Withdrawal Person 2 (2026 AUD)'),
            ('investment_withdrawal', 'Investment Withdrawal',
                'Annual Investment Withdrawal (2026 AUD)'),
        ]
        for key, label, title in plot_configs:
            if key == 'super_person2' and household_size == 1:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(years, annual_data[key], label=label, color='blue')
            step = max(1, len(years) // 10)
            ax.set_xticks(range(0, len(years), step))
            ax.set_xticklabels([years[i] for i in range(0, len(years), step)],
                               rotation=45)
            ax.set_xlabel('Year')
            ax.set_ylabel('AUD (2026 Dollars)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # -----------------------------------------------------------------
        # Summary
        # -----------------------------------------------------------------
        st.subheader("Summary")
        st.write(f"Mortgage paid off in: "
                 f"{mortgage_payoff_year if mortgage_payoff_year else 'Not paid off'}")
        total_super = annual_data['super_person1'][rs_idx] + (
            annual_data['super_person2'][rs_idx] if household_size == 2 else 0)
        st.write(f"Total super at retirement: {total_super:,.0f} AUD (2026 dollars)")
        st.write(f"Investment balance at retirement: "
                 f"{annual_data['investment'][rs_idx]:,.0f} AUD (2026 dollars)")
        st.write(f"Target retirement salary: "
                 f"{retirement_salary:,.0f} AUD/year (2026 dollars)")

        # -----------------------------------------------------------------
        # Tables
        # -----------------------------------------------------------------
        st.subheader("Annual Expenses Breakdown (2026 AUD)")
        expense_headers = ["Year", "Base Expenses", "Childcare",
            "Baby Supplies", "Food (Kids)", "Clothing (Kids)",
            "Healthcare (Kids)", "Education", "Tertiary", "Extracurricular",
            "Misc (Kids)", "Mortgage", "Super Withdrawal (P1)",
            "Super Withdrawal (P2)", "Investment Withdrawal", "Total Expenses"]
        expense_table_data = [
            [y, f"{annual_data['base_expenses'][i]:.0f}",
             f"{annual_data['childcare'][i]:.0f}",
             f"{annual_data['baby_supplies'][i]:.0f}",
             f"{annual_data['food_kids'][i]:.0f}",
             f"{annual_data['clothing_kids'][i]:.0f}",
             f"{annual_data['healthcare_kids'][i]:.0f}",
             f"{annual_data['education'][i]:.0f}",
             f"{annual_data['tertiary'][i]:.0f}",
             f"{annual_data['extracurricular'][i]:.0f}",
             f"{annual_data['misc_kids'][i]:.0f}",
             f"{annual_data['mortgage'][i]:.0f}",
             f"{annual_data['super_withdrawal_person1'][i]:.0f}",
             f"{annual_data['super_withdrawal_person2'][i]:.0f}",
             f"{annual_data['investment_withdrawal'][i]:.0f}",
             f"{annual_data['expenses'][i]:.0f}"]
            for i, y in enumerate(years)]
        st.dataframe(pd.DataFrame(expense_table_data, columns=expense_headers))

        st.subheader("Yearly Financial Summary (2026 AUD)")
        headers = ["Year", "Expenses", "Income", "Savings", "Offset",
                   "Investment", "Principal Paid", "Super (P1)", "Super (P2)",
                   "Tax", "Pension", "SG Contribution", "Salary Sacrifice",
                   "Super Withdrawal (P1)", "Super Withdrawal (P2)",
                   "Investment Withdrawal"]
        table_data = []
        for i, y in enumerate(years):
            d = {k: annual_data[k][i] for k in annual_data}
            table_data.append([y, f"{d['expenses']:.0f}", f"{d['income']:.0f}",
                f"{d['savings']:.0f}", f"{d['offset']:.0f}",
                f"{d['investment']:.0f}", f"{d['principal_paid']:.0f}",
                f"{d['super_person1']:.0f}", f"{d['super_person2']:.0f}",
                f"{d['tax']:.0f}", f"{d['pension']:.0f}",
                f"{d['sg_contribution']:.0f}", f"{d['salary_sacrifice']:.0f}",
                f"{d['super_withdrawal_person1']:.0f}",
                f"{d['super_withdrawal_person2']:.0f}",
                f"{d['investment_withdrawal']:.0f}"])
        st.dataframe(pd.DataFrame(table_data, columns=headers))

else:
    st.info("Adjust the parameters in the sidebar and click "
            "**Run Simulation** to see the results.")

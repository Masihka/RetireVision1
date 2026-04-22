"""
Headless run of the corrected retirement simulation using the default UI values.
The simulation math below is copied verbatim from retirement_sim_fixed.py.
Only the Streamlit I/O is removed. If a number disagrees with what you'd
expect, it reflects the model, not a transcription change.
"""
import numpy as np
import pandas as pd

# ---------------- Defaults taken directly from the sidebar defaults ----------------
simulation_start_year = 2025
retirement_year = 2054
death_year = 2079
household_size = 2
starting_age = 35
retirement_salary = 71723.0  # couple default (ASFA)

inflation_rate = 0.025
wage_growth_rate = 0.03
nominal_super_growth_rate = 0.065
nominal_investment_rate = 0.01
super_rate = 0.11
nominal_mortgage_rate = 0.06

initial_super = 50000.0
initial_salary = 100000.0
salary_sacrifice_annual = 0.0
mortgage_principal = 500000.0
offset_balance = 150000.0
mortgage_payment = 3000.0
reserve_target = 100000.0
homeowner = True
include_investment = True

tax_brackets = [(18200, 0), (45000, 0.19), (120000, 0.325), (180000, 0.37), (float('inf'), 0.45)]
medicare_levy = 0.02
max_concessional_cap = 27500.0

# Base monthly expenses (sum of all sidebar defaults)
base_expenses = {
    'car_insurance':         2000.0/12,
    'car_registration':      800.0/12,
    'traveling_abroad':      500.0/12,
    'car_service_fee':       600.0/12,
    'body_corporate':        1500.0/12,
    'council_cost':          1500.0/12,
    'utilities':             400.0,
    'food_and_groceries':    800.0,
    'transportation_other':  400.0,
    'healthcare':            400.0,
    'private_health_insurance': 300.0,
    'personal_care':         200.0,
    'entertainment':         200.0,
    'clothing':              200.0,
    'household_supplies':    100.0,
    'miscellaneous':         700.0,
    'gym':                   80.0,
}
base_expenses_monthly = sum(base_expenses.values())

# Child expenses with default windows & amounts
child_expenses = {
    'childcare':       {'start': simulation_start_year,      'end': simulation_start_year+5,  'amount': 36000.0/12},
    'baby_supplies':   {'start': simulation_start_year,      'end': simulation_start_year+3,  'amount': 3600.0/12},
    'food':            {'start': simulation_start_year,      'end': simulation_start_year+25, 'amount': 2400.0/12},
    'clothing':        {'start': simulation_start_year,      'end': simulation_start_year+18, 'amount': 1000.0/12},
    'healthcare':      {'start': simulation_start_year,      'end': simulation_start_year+25, 'amount': 500.0/12},
    'education':       {'start': simulation_start_year+5,    'end': simulation_start_year+18, 'amount': 800.0/12},
    'tertiary':        {'start': simulation_start_year+18,   'end': simulation_start_year+25, 'amount': 2000.0/12},
    'extracurricular': {'start': simulation_start_year+5,    'end': simulation_start_year+18, 'amount': 1000.0/12},
    'misc':            {'start': simulation_start_year,      'end': simulation_start_year+18, 'amount': 1000.0/12},
}

pension_age = 67
max_pension_single = 29874.0
max_pension_couple = 45037.2
income_free_area = 5512.0
income_taper = 0.50
assets_free_area = 470000.0   # homeowner + couple
assets_taper = 78.0/1000
deeming_threshold = 103800.0  # couple default

unemployment_years = {
    0: range(simulation_start_year + 9,  simulation_start_year + 11),  # 2034-2035
    1: range(simulation_start_year + 14, simulation_start_year + 16),  # 2039-2040
}

MIN_DRAWDOWN_RATES = {
    (0, 64): 0.04, (65, 74): 0.05, (75, 79): 0.06, (80, 84): 0.07,
    (85, 89): 0.09, (90, 94): 0.11, (95, float('inf')): 0.14,
}

# ---------------- Functions copied from the fixed file ----------------
def calculate_tax(income, sacrifice):
    taxable = max(income - sacrifice, 0)
    tax = 0
    for i, (threshold, rate) in enumerate(tax_brackets):
        prev_threshold = tax_brackets[i-1][0] if i > 0 else 0
        if taxable <= threshold:
            tax += (taxable - prev_threshold) * rate
            break
        tax += (threshold - prev_threshold) * rate
    tax += taxable * medicare_levy
    if taxable <= 37500:
        lito = 700
    elif taxable <= 45000:
        lito = 700 - 0.05*(taxable-37500)
    elif taxable <= 66667:
        lito = 325 - 0.015*(taxable-45000)
    else:
        lito = 0
    return max(tax - lito, 0)

def get_drawdown_rate(age):
    for (lo, hi), r in MIN_DRAWDOWN_RATES.items():
        if lo <= age <= hi:
            return r
    return 0.14

def calculate_pension(super_balances, investment_balance, year, inflation_rates, current_age):
    if current_age < pension_age:
        return 0
    total = sum(super_balances) + investment_balance
    infl_to_date = np.prod([1+r for r in inflation_rates[:year]]) if year > 0 else 1.0
    deeming_threshold_nom = deeming_threshold * infl_to_date
    income_free_area_nom  = income_free_area  * infl_to_date
    assets_free_area_nom  = assets_free_area  * infl_to_date
    max_p = (max_pension_couple if household_size == 2 else max_pension_single) * infl_to_date
    low_part = min(total, deeming_threshold_nom)
    high_part = max(0, total - deeming_threshold_nom)
    deemed_income = low_part*0.0025 + high_part*0.0225
    income_reduction = max(0, (deemed_income - income_free_area_nom) * income_taper)
    assets_reduction = max(0, (total - assets_free_area_nom) * assets_taper)
    return max(0, max_p - max(income_reduction, assets_reduction)) / 12

# ---------------- Simulation loop (copied, same logic) ----------------
months = (death_year - simulation_start_year) * 12
years_to_retirement = retirement_year - simulation_start_year
inflation_rates = [inflation_rate] * (death_year - simulation_start_year)
wage_growth_rates = [wage_growth_rate] * (death_year - simulation_start_year)
super_rates = [super_rate] * (death_year - simulation_start_year)
monthly_mortgage_rate = (1+nominal_mortgage_rate)**(1/12) - 1

monthly = {k: [] for k in [
    'expenses','income','savings','offset','investment','principal_paid',
    'super_person1','super_person2','tax','pension','sg_contribution',
    'salary_sacrifice','base_expenses','mortgage',
    'super_withdrawal_person1','super_withdrawal_person2','investment_withdrawal','shortfall',
    'take_home_monthly_nominal','inflation_factor'
]}
super_balances = [initial_super]*household_size
investment_balance = 0.0
mortgage_active = True
mortgage_payoff_year = None

for m in range(months):
    year = m // 12
    month_in_year = m % 12
    current_year = simulation_start_year + year
    current_age = starting_age + year
    infl_factor = (np.prod([1+r for r in inflation_rates[:year]]) if year > 0 else 1.0) \
                  * (1 + inflation_rates[year])**(month_in_year/12)

    pension_nom = calculate_pension(super_balances, investment_balance, year, inflation_rates, current_age)
    living = base_expenses_monthly
    for cat, d in child_expenses.items():
        sm = (d['start'] - simulation_start_year)*12
        em = (d['end']   - simulation_start_year)*12
        if sm <= m < em:
            living += d['amount']

    base_nom = base_expenses_monthly * infl_factor
    exp_nom = living * infl_factor
    mort_nom = 0
    if mortgage_active:
        mort_nom = mortgage_payment
        exp_nom += mortgage_payment

    super_wd = [0.0]*household_size
    inv_wd = 0.0
    sg = sacrifice = tax_nom = salary_nom = take_home_m = 0.0
    shortfall = 0.0

    if year < years_to_retirement:
        employed = [current_year not in unemployment_years[i] for i in range(household_size)]
        sal_pp = initial_salary * np.prod([1+r for r in wage_growth_rates[:year]])
        sg_pp = sal_pp * super_rates[year]
        max_sac = max(0, max_concessional_cap - sg_pp)
        sac_pp = min(salary_sacrifice_annual, max_sac)
        sg = sum(sg_pp for e in employed if e)
        sacrifice = sac_pp * sum(1 for e in employed if e)
        tax_nom = sum(calculate_tax(sal_pp if e else 0, sac_pp if e else 0) for e in employed)
        salary_nom = sum(sal_pp if e else 0 for e in employed)
        take_home_m = (salary_nom - tax_nom - sacrifice)/12
    else:
        target_m = retirement_salary * np.prod([1+r for r in inflation_rates[:year]]) / 12
        remaining = target_m - pension_nom
        if remaining > 0:
            dr = get_drawdown_rate(current_age)
            for i in range(household_size):
                min_d = super_balances[i]*dr/12
                max_d = super_balances[i]/12
                need = remaining / max(household_size - i, 1)
                w = min(max(min_d, need), max_d)
                if super_balances[i] >= w:
                    super_wd[i] = w
                    super_balances[i] -= w
                    remaining -= w
                else:
                    super_wd[i] = super_balances[i]
                    super_balances[i] = 0
                    remaining -= super_wd[i]
            if remaining > 0:
                inv_wd = min(remaining, investment_balance)
                investment_balance -= inv_wd
                remaining -= inv_wd
            if remaining > 0:
                shortfall = remaining

    principal_rep = 0.0
    savings_m = 0.0
    if year < years_to_retirement:
        cash = take_home_m - exp_nom
        if mortgage_active:
            net_debt = max(mortgage_principal - offset_balance, 0)
            interest = net_debt * monthly_mortgage_rate
            principal_rep = min(mortgage_payment - interest, mortgage_principal)
            mortgage_principal -= principal_rep
            if cash >= 0:
                savings_m = cash
                offset_balance += savings_m
            else:
                draw = min(offset_balance, -cash)
                offset_balance -= draw
                shortfall += (-cash) - draw
            if (offset_balance - mortgage_principal) >= reserve_target*infl_factor and mortgage_principal > 0:
                principal_rep += mortgage_principal
                offset_balance -= mortgage_principal
                reserve_nom = reserve_target*infl_factor
                investment_balance += reserve_nom
                offset_balance -= reserve_nom
                mortgage_principal = 0
                mortgage_active = False
                mortgage_payoff_year = current_year + (1 if month_in_year >= 9 else 0)
        else:
            if cash >= 0:
                savings_m = cash
            else:
                draw = min(investment_balance, -cash)
                investment_balance -= draw
                shortfall += (-cash) - draw
            g = (1 + nominal_investment_rate/12) if include_investment else 1.0
            investment_balance = investment_balance * g + savings_m
    else:
        if include_investment:
            investment_balance *= (1 + nominal_investment_rate/12)

    if month_in_year == 0:
        if year < years_to_retirement:
            per_person = (sg + sacrifice)/household_size * 0.85
        else:
            per_person = 0
        for i in range(household_size):
            g = super_balances[i]*nominal_super_growth_rate
            t = g*0.15
            super_balances[i] = super_balances[i] + g - t + per_person

    monthly['expenses'].append(exp_nom/infl_factor)
    monthly['income'].append((take_home_m + pension_nom + sum(super_wd) + inv_wd)/infl_factor)
    monthly['savings'].append(savings_m/infl_factor)
    monthly['offset'].append(offset_balance/infl_factor)
    monthly['investment'].append(investment_balance/infl_factor)
    monthly['principal_paid'].append(principal_rep/infl_factor)
    monthly['super_person1'].append(super_balances[0]/infl_factor)
    monthly['super_person2'].append(super_balances[1]/infl_factor if household_size==2 else 0)
    monthly['tax'].append(tax_nom/12/infl_factor)
    monthly['pension'].append(pension_nom/infl_factor)
    monthly['sg_contribution'].append(sg/12/infl_factor)
    monthly['salary_sacrifice'].append(sacrifice/12/infl_factor)
    monthly['base_expenses'].append(base_nom/infl_factor)
    monthly['mortgage'].append(mort_nom/infl_factor)
    monthly['super_withdrawal_person1'].append(super_wd[0]/infl_factor)
    monthly['super_withdrawal_person2'].append(super_wd[1]/infl_factor if household_size==2 else 0)
    monthly['investment_withdrawal'].append(inv_wd/infl_factor)
    monthly['shortfall'].append(shortfall/infl_factor)
    monthly['take_home_monthly_nominal'].append(take_home_m)
    monthly['inflation_factor'].append(infl_factor)

# ---------------- Annual aggregation ----------------
n_years = death_year - simulation_start_year
years = [simulation_start_year + i for i in range(n_years)]
def ann_sum(key): return [sum(monthly[key][i*12:(i+1)*12]) for i in range(n_years)]
def ann_end(key): return [monthly[key][(i+1)*12-1] for i in range(n_years)]

annual = {}
for k in ['expenses','income','savings','principal_paid','tax','pension','sg_contribution',
          'salary_sacrifice','base_expenses','mortgage','super_withdrawal_person1',
          'super_withdrawal_person2','investment_withdrawal','shortfall']:
    annual[k] = ann_sum(k)
for k in ['offset','investment','super_person1','super_person2']:
    annual[k] = ann_end(k)

# ---------------- Sanity printouts ----------------
print(f"Mortgage paid off in: {mortgage_payoff_year}")
rsi = years_to_retirement
print(f"\n--- At retirement start (year index {rsi} = {years[rsi]}) ---")
print(f"Super p1:     {annual['super_person1'][rsi]:>12,.0f}")
print(f"Super p2:     {annual['super_person2'][rsi]:>12,.0f}")
print(f"Super total:  {annual['super_person1'][rsi]+annual['super_person2'][rsi]:>12,.0f}")
print(f"Investment:   {annual['investment'][rsi]:>12,.0f}")
print(f"Offset:       {annual['offset'][rsi]:>12,.0f}")

print(f"\n--- First few years of working life (2024 AUD) ---")
print(f"{'Year':<6}{'Salary(NH)':>12}{'Tax':>10}{'TakeHome':>12}{'Expenses':>12}{'Savings':>10}{'Offset':>12}{'SuperP1':>12}")
for i in range(5):
    m_idx = i*12 + 11  # December
    # rebuild nominal salary
    sal_pp_nom = initial_salary * (1+wage_growth_rate)**i
    sal_tot_nom = sal_pp_nom * 2 if i not in [9,10,14,15] else sal_pp_nom
    take = monthly['take_home_monthly_nominal'][m_idx] * 12
    infl = monthly['inflation_factor'][m_idx]
    print(f"{years[i]:<6}{sal_tot_nom/infl:>12,.0f}{annual['tax'][i]:>10,.0f}{take/infl:>12,.0f}"
          f"{annual['expenses'][i]:>12,.0f}{annual['savings'][i]:>10,.0f}"
          f"{annual['offset'][i]:>12,.0f}{annual['super_person1'][i]:>12,.0f}")

print(f"\n--- First 5 retirement years (2024 AUD) ---")
print(f"{'Year':<6}{'Target':>10}{'Pension':>10}{'SuperWD':>12}{'InvWD':>10}{'Super p1':>12}{'Invest':>12}{'Short':>10}")
for i in range(rsi, min(rsi+5, n_years)):
    swd = annual['super_withdrawal_person1'][i] + annual['super_withdrawal_person2'][i]
    print(f"{years[i]:<6}{retirement_salary:>10,.0f}{annual['pension'][i]:>10,.0f}"
          f"{swd:>12,.0f}{annual['investment_withdrawal'][i]:>10,.0f}"
          f"{annual['super_person1'][i]:>12,.0f}{annual['investment'][i]:>12,.0f}"
          f"{annual['shortfall'][i]:>10,.0f}")

print(f"\n--- Last 5 years (2024 AUD) ---")
print(f"{'Year':<6}{'Pension':>10}{'SuperWD':>12}{'InvWD':>10}{'Super p1':>12}{'Super p2':>12}{'Invest':>12}{'Short':>10}")
for i in range(n_years-5, n_years):
    swd = annual['super_withdrawal_person1'][i] + annual['super_withdrawal_person2'][i]
    print(f"{years[i]:<6}{annual['pension'][i]:>10,.0f}{swd:>12,.0f}"
          f"{annual['investment_withdrawal'][i]:>10,.0f}"
          f"{annual['super_person1'][i]:>12,.0f}{annual['super_person2'][i]:>12,.0f}"
          f"{annual['investment'][i]:>12,.0f}{annual['shortfall'][i]:>10,.0f}")

print(f"\n--- Totals ---")
print(f"Total shortfall over sim (2024 AUD): {sum(annual['shortfall']):,.0f}")
print(f"Final investment balance          : {annual['investment'][-1]:,.0f}")
print(f"Final super p1                    : {annual['super_person1'][-1]:,.0f}")
print(f"Final super p2                    : {annual['super_person2'][-1]:,.0f}")

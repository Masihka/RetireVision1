import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

# Title
st.title("Retirement Simulation App Created by Masih")
st.warning(
    "📢 **Disclaimer:** This app is not intended to provide financial advice and has not been validated against real-world case studies. It is for general informational use only."
)
#GithubIcon {
  visibility: hidden;
}

# Sidebar for Inputs
st.sidebar.header("Simulation Settings")
simulation_start_year = st.sidebar.number_input("Start Year", min_value=2000, max_value=2100, value=2025, step=1, help="Year the simulation begins")
retirement_year = st.sidebar.number_input("Retirement Year", min_value=2000, max_value=2100, value=2054, step=1, help="Year retirement begins")
death_year = st.sidebar.number_input("End Year", min_value=2000, max_value=2100, value=2079, step=1, help="Year the simulation ends")
retirement_salary = st.sidebar.number_input(
    "Desired Retirement Salary (Annual, 2024 AUD)",
    value=71723.0 if st.session_state.get('household_size', 2) == 2 else 50962.0,
    step=1000.0,
    help="Default is ASFA comfortable standard: $71,723 for couples, $50,962 for singles"
)

st.sidebar.header("Economic Assumptions")
inflation_rate = st.sidebar.slider("Annual Inflation Rate", 0.0, 0.1, 0.025, step=0.005, help="Average annual inflation rate (e.g., 0.025 = 2.5%)")
wage_growth_rate = st.sidebar.slider("Annual Wage Growth Rate", 0.0, 0.1, 0.03, step=0.005, help="Average annual wage growth (e.g., 0.03 = 3%)")
nominal_super_growth_rate = st.sidebar.slider("Super Growth Rate", 0.0, 0.1, 0.065, step=0.005, help="Nominal annual superannuation growth rate")
nominal_investment_rate = st.sidebar.slider("Investment Rate", 0.0, 0.1, 0.01, step=0.005, help="Nominal annual investment return rate")
super_rate = st.sidebar.slider("Super Contribution Rate", 0.0, 0.2, 0.11, step=0.005, help="Super Guarantee rate (11% in 2023, increasing to 12% by 2025)")
nominal_mortgage_rate = st.sidebar.slider("Mortgage Rate", 0.0, 0.1, 0.06, step=0.005, help="Nominal annual mortgage interest rate")

st.sidebar.header("Personal Finances")
initial_super = st.sidebar.number_input("Initial Super per Person", value=50000.0, step=1000.0, help="Starting super balance per household member")
initial_salary = st.sidebar.number_input("Initial Salary per Person", value=100000.0, step=1000.0, help="Starting annual salary per person")
household_size = st.sidebar.selectbox("Household Size", [1, 2], index=1, help="Number of income earners in household", key="household_size")
salary_sacrifice_annual = st.sidebar.number_input("Annual Salary Sacrifice", value=0.0, step=1000.0, help="Annual pre-tax super contribution")
mortgage_principal = st.sidebar.number_input("Mortgage Principal", value=500000.0, step=1000.0, help="Initial mortgage loan amount")
offset_balance = st.sidebar.number_input("Offset Balance", value=150000.0, step=1000.0, help="Initial balance in mortgage offset account")
mortgage_payment = st.sidebar.number_input("Monthly Mortgage Payment", value=3000.0, step=100.0, help="Fixed monthly payment towards mortgage")
reserve_target = st.sidebar.number_input("Reserve Target", value=100000.0, step=1000.0, help="Desired cash reserve before paying off mortgage")
homeowner = st.sidebar.checkbox("Homeowner", value=True, help="Check if household owns a home")

st.sidebar.header("Tax Settings")
tax_brackets_input = st.sidebar.text_area(
    "Tax Brackets (format: threshold:rate, e.g., 18200:0, 45000:0.19)",
    "18200:0, 45000:0.19, 120000:0.325, 180000:0.37, inf:0.45",
    help="Australian tax brackets for 2023-2024 (adjustable)"
)
medicare_levy = st.sidebar.slider("Medicare Levy", 0.0, 0.05, 0.02, step=0.005, help="Standard rate is 2%")
max_concessional_cap = st.sidebar.number_input("Max Concessional Cap", value=27500.0, step=1000.0, help="Annual cap for concessional super contributions (2023-2024: $27,500)")

# Parse tax brackets
try:
    tax_brackets = []
    for item in tax_brackets_input.split(','):
        threshold, rate = item.strip().split(':')
        threshold = float(threshold) if threshold != 'inf' else float('inf')
        rate = float(rate)
        tax_brackets.append((threshold, rate))
except:
    st.sidebar.error("Invalid tax brackets format. Use 'threshold:rate' pairs separated by commas.")
    st.stop()

st.sidebar.header("Base Expenses (Monthly, 2024 AUD)")
base_expenses = {}

# Expenses with Monthly/Yearly option
expenses_with_frequency = [
    ('car_insurance', 'Car Insurance', 2000.0 / 12, 'Monthly car insurance cost'),
    ('car_registration', 'Car Registration', 800.0 / 12, 'Monthly car registration cost'),
    ('traveling_abroad', 'Traveling Abroad', 500.0 / 12, 'Travel expenses (e.g., vacations)'),
    ('car_service_fee', 'Car Service Fee', 600.0 / 12, 'Monthly car maintenance cost'),
    ('body_corporate', 'Body Corporate', 1500.0 / 12, 'Monthly strata fees'),
    ('council_cost', 'Council Cost', 1500.0 / 12, 'Monthly council rates')
]

for key, label, default, help_text in expenses_with_frequency:
    frequency = st.sidebar.selectbox(f"{label} Frequency", ["Monthly", "Yearly"], index=0, key=f"{key}_frequency", help=f"Choose whether to input {label} as a monthly or yearly amount")
    value = st.sidebar.number_input(label, value=default, step=10.0, key=f"{key}_input", help=help_text)
    base_expenses[key] = value / 12 if frequency == "Yearly" else value

# Other expenses (remain monthly only)
other_expenses = {
    'utilities': st.sidebar.number_input("Utilities", value=400.0, step=10.0, help="Monthly utility bills"),
    'food_and_groceries': st.sidebar.number_input("Food and Groceries", value=800.0, step=10.0, help="Monthly grocery expenses"),
    'transportation_other': st.sidebar.number_input("Other Transportation", value=400.0, step=10.0, help="Other monthly transport costs"),
    'healthcare': st.sidebar.number_input("Healthcare", value=400.0, step=10.0, help="Monthly healthcare expenses"),
    'private_health_insurance': st.sidebar.number_input("Private Health Insurance", value=300.0, step=10.0, help="Monthly health insurance cost"),
    'personal_care': st.sidebar.number_input("Personal Care", value=200.0, step=10.0, help="Monthly personal care expenses"),
    'entertainment': st.sidebar.number_input("Entertainment", value=200.0, step=10.0, help="Monthly entertainment expenses"),
    'clothing': st.sidebar.number_input("Clothing", value=200.0, step=10.0, help="Monthly clothing expenses"),
    'household_supplies': st.sidebar.number_input("Household Supplies", value=100.0, step=10.0, help="Monthly household supply costs"),
    'miscellaneous': st.sidebar.number_input("Miscellaneous", value=700.0, step=10.0, help="Other monthly expenses"),
    'gym': st.sidebar.number_input("Gym", value=80.0, step=10.0, help="Monthly gym membership cost")
}
base_expenses.update(other_expenses)
base_expenses_monthly = sum(base_expenses.values())

st.sidebar.header("Child Expenses (Monthly, 2024 AUD)")
child_expenses = {
    'childcare': {
        'start': st.sidebar.number_input("Childcare Start Year", value=simulation_start_year, step=1, help="Year childcare begins"),
        'end': st.sidebar.number_input("Childcare End Year", value=simulation_start_year + 5, step=1, help="Year childcare ends"),
        'amount': st.sidebar.number_input("Childcare Amount", value=36000.0 / 12, step=100.0, help="Monthly childcare cost")
    },
    'baby_supplies': {
        'start': st.sidebar.number_input("Baby Supplies Start Year", value=simulation_start_year, step=1, help="Year baby supplies begin"),
        'end': st.sidebar.number_input("Baby Supplies End Year", value=simulation_start_year + 3, step=1, help="Year baby supplies end"),
        'amount': st.sidebar.number_input("Baby Supplies Amount", value=3600.0 / 12, step=100.0, help="Monthly baby supplies cost")
    },
    'food': {
        'start': st.sidebar.number_input("Food (Kids) Start Year", value=simulation_start_year, step=1, help="Year kids' food expenses begin"),
        'end': st.sidebar.number_input("Food (Kids) End Year", value=simulation_start_year + 25, step=1, help="Year kids' food expenses end"),
        'amount': st.sidebar.number_input("Food (Kids) Amount", value=2400.0 / 12, step=100.0, help="Monthly kids' food cost")
    },
    'clothing': {
        'start': st.sidebar.number_input("Clothing (Kids) Start Year", value=simulation_start_year, step=1, help="Year kids' clothing expenses begin"),
        'end': st.sidebar.number_input("Clothing (Kids) End Year", value=simulation_start_year + 18, step=1, help="Year kids' clothing expenses end"),
        'amount': st.sidebar.number_input("Clothing (Kids) Amount", value=1000.0 / 12, step=100.0, help="Monthly kids' clothing cost")
    },
    'healthcare': {
        'start': st.sidebar.number_input("Healthcare (Kids) Start Year", value=simulation_start_year, step=1, help="Year kids' healthcare expenses begin"),
        'end': st.sidebar.number_input("Healthcare (Kids) End Year", value=simulation_start_year + 25, step=1, help="Year kids' healthcare expenses end"),
        'amount': st.sidebar.number_input("Healthcare (Kids) Amount", value=500.0 / 12, step=100.0, help="Monthly kids' healthcare cost")
    },
    'education': {
        'start': st.sidebar.number_input("Education Start Year", value=simulation_start_year + 5, step=1, help="Year education expenses begin"),
        'end': st.sidebar.number_input("Education End Year", value=simulation_start_year + 18, step=1, help="Year education expenses end"),
        'amount': st.sidebar.number_input("Education Amount", value=800.0 / 12, step=100.0, help="Monthly education cost")
    },
    'tertiary': {
        'start': st.sidebar.number_input("Tertiary Start Year", value=simulation_start_year + 18, step=1, help="Year tertiary education expenses begin"),
        'end': st.sidebar.number_input("Tertiary End Year", value=simulation_start_year + 25, step=1, help="Year tertiary education expenses end"),
        'amount': st.sidebar.number_input("Tertiary Amount", value=2000.0 / 12, step=100.0, help="Monthly tertiary education cost")
    },
    'extracurricular': {
        'start': st.sidebar.number_input("Extracurricular Start Year", value=simulation_start_year + 5, step=1, help="Year extracurricular expenses begin"),
        'end': st.sidebar.number_input("Extracurricular End Year", value=simulation_start_year + 18, step=1, help="Year extracurricular expenses end"),
        'amount': st.sidebar.number_input("Extracurricular Amount", value=1000.0 / 12, step=100.0, help="Monthly extracurricular cost")
    },
    'misc': {
        'start': st.sidebar.number_input("Misc (Kids) Start Year", value=simulation_start_year, step=1, help="Year miscellaneous kids' expenses begin"),
        'end': st.sidebar.number_input("Misc (Kids) End Year", value=simulation_start_year + 18, step=1, help="Year miscellaneous kids' expenses end"),
        'amount': st.sidebar.number_input("Misc (Kids) Amount", value=1000.0 / 12, step=100.0, help="Monthly miscellaneous kids' cost")
    }
}
expenses = {'base': base_expenses, 'child': child_expenses}

st.sidebar.header("Pension Settings")
pension_age = st.sidebar.number_input("Pension Age", value=67, step=1, help="Age eligibility for Australian Age Pension")
max_pension_single = st.sidebar.number_input("Max Pension Single (Annual)", value=29874.0, step=100.0, help="2024 annual rate: $29,874")
max_pension_couple = st.sidebar.number_input("Max Pension Couple (Annual)", value=45037.2, step=100.0, help="2024 annual rate: $45,037.20")
income_free_area = st.sidebar.number_input("Income Test Free Area (Annual)", value=5512.0, step=100.0, help="Combined for couples, 2024")
income_taper = st.sidebar.slider("Income Taper Rate", 0.0, 1.0, 0.50, step=0.05, help="Reduction per $ over free area")
assets_free_area = st.sidebar.number_input(
    "Assets Test Free Area",
    value=470000.0 if homeowner and household_size == 2 else (314000.0 if homeowner else (722000.0 if not homeowner and household_size == 2 else 566000.0)),
    step=1000.0,
    help="Varies by homeowner status and household size"
)
assets_taper = st.sidebar.number_input("Assets Taper Rate (Annual per $1000)", value=78.0 / 1000, step=0.01, help="$78 per $1,000 over free area annually")

st.sidebar.header("Unemployment Periods")
unemployment_years = {}
for i in range(household_size):
    st.sidebar.subheader(f"Member {i+1}")
    start = st.sidebar.number_input(f"Unemployment Start Year (Member {i+1})", value=simulation_start_year + 9 if i == 0 else simulation_start_year + 14, step=1, help="Year unemployment begins")
    end = st.sidebar.number_input(f"Unemployment End Year (Member {i+1})", value=simulation_start_year + 10 if i == 0 else simulation_start_year + 15, step=1, help="Year unemployment ends")
    unemployment_years[i] = range(start, end + 1)

# Minimum super drawdown rates
MIN_DRAWDOWN_RATES = {
    (0, 64): 0.04,
    (65, 74): 0.05,
    (75, 79): 0.06,
    (80, 84): 0.07,
    (85, 89): 0.09,
    (90, 94): 0.11,
    (95, float('inf')): 0.14
}

# Simulation Logic
def calculate_tax(income, sacrifice, year, inflation_rates):
    taxable = max(income - sacrifice, 0)
    tax = 0
    for i, (threshold, rate) in enumerate(tax_brackets):
        if taxable <= threshold:
            prev_threshold = tax_brackets[i-1][0] if i > 0 else 0
            tax += (taxable - prev_threshold) * rate
            break
        if i > 0:
            tax += (threshold - tax_brackets[i-1][0]) * rate
    tax += taxable * medicare_levy
    lito = 700 if taxable <= 37500 else (700 - 0.05 * (taxable - 37500)) if taxable <= 45000 else (325 - 0.015 * (taxable - 45000)) if taxable <= 66667 else 0
    tax = max(tax - lito, 0) * np.prod([1 + r for r in inflation_rates[:year + 1]])
    return tax

def get_drawdown_rate(age):
    for (min_age, max_age), rate in MIN_DRAWDOWN_RATES.items():
        if min_age <= age <= max_age:
            return rate
    return 0.14

def calculate_pension(super_balances, investment_balance, year, inflation_rates, current_age):
    if current_age < pension_age:
        return 0
    total_assets = sum(super_balances) + investment_balance
    deemed_income = 0
    threshold = 89600.0 if household_size == 2 else 60200.0
    for balance in super_balances + [investment_balance]:
        low_rate = min(balance, threshold) * 0.0025
        high_rate = max(0, balance - threshold) * 0.0225
        deemed_income += low_rate + high_rate
    income_test_reduction = max(0, (deemed_income * 12 - income_free_area) * income_taper)
    assets_test_reduction = max(0, (total_assets - assets_free_area) * assets_taper)
    max_pension = (max_pension_couple if household_size == 2 else max_pension_single) * np.prod([1 + r for r in inflation_rates[:year]])
    return max(0, max_pension - max(income_test_reduction, assets_test_reduction)) / 12

# Button to run simulation
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        months = (death_year - simulation_start_year) * 12
        years_to_retirement = retirement_year - simulation_start_year
        inflation_rates = [inflation_rate] * (death_year - simulation_start_year)
        wage_growth_rates = [wage_growth_rate] * (death_year - simulation_start_year)
        super_rates = [super_rate] * (death_year - simulation_start_year)
        monthly_mortgage_rate = (1 + nominal_mortgage_rate) ** (1/12) - 1
        real_rate = lambda infl: (1 + nominal_investment_rate) / (1 + infl) - 1
        total_initial_salary = initial_salary * household_size

        monthly_data = {key: [] for key in [
            'expenses', 'income', 'savings', 'offset', 'investment', 'principal_paid',
            'super_person1', 'super_person2', 'tax', 'pension', 'sg_contribution',
            'salary_sacrifice', 'base_expenses', 'childcare', 'baby_supplies', 'food_kids',
            'clothing_kids', 'healthcare_kids', 'education', 'tertiary', 'extracurricular',
            'misc_kids', 'mortgage', 'super_withdrawal_person1', 'super_withdrawal_person2',
            'investment_withdrawal'
        ]}
        super_balances = [initial_super] * household_size
        investment_balance = 0
        mortgage_active = True
        mortgage_payoff_year = None

        for m in range(months):
            year = m // 12
            month_in_year = m % 12
            current_year = simulation_start_year + year
            current_age = 35 + year  # Assuming starting age of 35
            inflation_factor = np.prod([1 + r for r in inflation_rates[:year + 1]]) * (1 + inflation_rates[year]) ** (month_in_year / 12)

            pension_nominal = calculate_pension(super_balances, investment_balance, year, inflation_rates, current_age)
            living_expenses = base_expenses_monthly
            childcare_nominal = baby_supplies_nominal = food_kids_nominal = clothing_kids_nominal = 0
            healthcare_kids_nominal = education_nominal = tertiary_nominal = extracurricular_nominal = misc_kids_nominal = 0
            mortgage_nominal = 0

            for category, details in expenses['child'].items():
                start_month = (details['start'] - simulation_start_year) * 12
                end_month = (details['end'] - simulation_start_year) * 12
                if start_month <= m < end_month:
                    amount_nominal = details['amount'] * inflation_factor
                    living_expenses += details['amount']
                    if category == 'childcare': childcare_nominal = amount_nominal
                    elif category == 'baby_supplies': baby_supplies_nominal = amount_nominal
                    elif category == 'food': food_kids_nominal = amount_nominal
                    elif category == 'clothing': clothing_kids_nominal = amount_nominal
                    elif category == 'healthcare': healthcare_kids_nominal = amount_nominal
                    elif category == 'education': education_nominal = amount_nominal
                    elif category == 'tertiary': tertiary_nominal = amount_nominal
                    elif category == 'extracurricular': extracurricular_nominal = amount_nominal
                    elif category == 'misc': misc_kids_nominal = amount_nominal

            base_expenses_nominal = base_expenses_monthly * inflation_factor
            expenses_nominal = living_expenses * inflation_factor
            if mortgage_active:
                mortgage_nominal = mortgage_payment
                expenses_nominal += mortgage_payment

            super_withdrawal = [0] * household_size
            investment_withdrawal = 0
            salary_nominal = sg_contribution = sacrifice_contribution = tax_nominal = take_home_nominal = 0

            if year < years_to_retirement:
                employed = [current_year not in unemployment_years[i] for i in range(household_size)]
                salary_nominal_per_person = initial_salary * np.prod([1 + r for r in wage_growth_rates[:year]])
                sg_contribution_per_person = salary_nominal_per_person * super_rates[year]
                max_sacrifice = max(0, max_concessional_cap - sg_contribution_per_person)
                sacrifice_per_person = min(salary_sacrifice_annual, max_sacrifice)
                sg_contribution = sum(sg_contribution_per_person * (1 if emp else 0) for emp in employed)
                sacrifice_contribution = sacrifice_per_person * sum(employed)
                tax_nominal = sum(
                    calculate_tax(salary_nominal_per_person * (1 if emp else 0), sacrifice_per_person * (1 if emp else 0), year, inflation_rates)
                    for emp in employed
                )
                salary_nominal = sum(salary_nominal_per_person * (1 if emp else 0) for emp in employed)
                take_home_nominal = salary_nominal - tax_nominal - sacrifice_contribution
                take_home_monthly_nominal = take_home_nominal / 12
            else:
                target_salary_monthly = (retirement_salary * np.prod([1 + r for r in inflation_rates[:year]]) / 12)
                remaining_need = target_salary_monthly - pension_nominal
                if remaining_need > 0:
                    drawdown_rate = get_drawdown_rate(current_age)
                    for i in range(household_size):
                        min_drawdown = super_balances[i] * drawdown_rate / 12
                        max_drawdown = super_balances[i] / 12
                        need_per_person = remaining_need / (household_size - i)
                        withdrawal = min(max(min_drawdown, need_per_person), max_drawdown)
                        if super_balances[i] >= withdrawal:
                            super_withdrawal[i] = withdrawal
                            super_balances[i] -= withdrawal
                            remaining_need -= withdrawal
                        else:
                            super_withdrawal[i] = super_balances[i]
                            super_balances[i] = 0
                            remaining_need -= super_withdrawal[i]
                    if remaining_need > 0:
                        investment_withdrawal = min(remaining_need, investment_balance)
                        investment_balance -= investment_withdrawal

            principal_repayment = savings = 0
            if year < years_to_retirement:
                if mortgage_active:
                    net_debt = max(mortgage_principal - offset_balance, 0)
                    interest = net_debt * monthly_mortgage_rate
                    principal_repayment = min(mortgage_payment - interest, mortgage_principal)
                    mortgage_principal -= principal_repayment
                    savings = max(take_home_monthly_nominal - expenses_nominal, 0)
                    offset_balance += savings
                    if (offset_balance - mortgage_principal) >= reserve_target * inflation_factor and mortgage_principal > 0:
                        principal_repayment += mortgage_principal
                        offset_balance -= mortgage_principal
                        reserve_nominal = reserve_target * inflation_factor
                        investment_balance += reserve_nominal
                        offset_balance -= reserve_nominal
                        mortgage_principal = 0
                        mortgage_active = False
                        mortgage_payoff_year = current_year + (1 if month_in_year >= 9 else 0)
                else:
                    savings = max(take_home_monthly_nominal - expenses_nominal, 0)
                    investment_balance = investment_balance * (1 + nominal_investment_rate / 12) + savings

            if month_in_year == 0:
                for i in range(household_size):
                    total_concessional = ((sg_contribution / household_size + sacrifice_contribution / household_size) * 0.85) if year < years_to_retirement else 0
                    super_growth = super_balances[i] * nominal_super_growth_rate
                    super_tax = super_growth * 0.15
                    super_balances[i] = super_balances[i] + super_growth - super_tax + total_concessional / household_size

            if not mortgage_active and year < years_to_retirement:
                investment_balance *= (1 + nominal_investment_rate / 12)

            monthly_data['expenses'].append(expenses_nominal / inflation_factor)
            monthly_data['income'].append((take_home_monthly_nominal + pension_nominal + sum(super_withdrawal) + investment_withdrawal) / inflation_factor)
            monthly_data['savings'].append(savings / inflation_factor)
            monthly_data['offset'].append(offset_balance / inflation_factor)
            monthly_data['investment'].append(investment_balance / inflation_factor)
            monthly_data['principal_paid'].append(principal_repayment / inflation_factor)
            monthly_data['super_person1'].append(super_balances[0] / inflation_factor)
            monthly_data['super_person2'].append(super_balances[1] / inflation_factor if household_size == 2 else 0)
            monthly_data['tax'].append(tax_nominal / 12 / inflation_factor)
            monthly_data['pension'].append(pension_nominal / inflation_factor)
            monthly_data['sg_contribution'].append(sg_contribution / 12 / inflation_factor)
            monthly_data['salary_sacrifice'].append(sacrifice_contribution / 12 / inflation_factor)
            monthly_data['base_expenses'].append(base_expenses_nominal / inflation_factor)
            monthly_data['childcare'].append(childcare_nominal / inflation_factor)
            monthly_data['baby_supplies'].append(baby_supplies_nominal / inflation_factor)
            monthly_data['food_kids'].append(food_kids_nominal / inflation_factor)
            monthly_data['clothing_kids'].append(clothing_kids_nominal / inflation_factor)
            monthly_data['healthcare_kids'].append(healthcare_kids_nominal / inflation_factor)
            monthly_data['education'].append(education_nominal / inflation_factor)
            monthly_data['tertiary'].append(tertiary_nominal / inflation_factor)
            monthly_data['extracurricular'].append(extracurricular_nominal / inflation_factor)
            monthly_data['misc_kids'].append(misc_kids_nominal / inflation_factor)
            monthly_data['mortgage'].append(mortgage_nominal / inflation_factor)
            monthly_data['super_withdrawal_person1'].append(super_withdrawal[0] / inflation_factor)
            monthly_data['super_withdrawal_person2'].append(super_withdrawal[1] / inflation_factor if household_size == 2 else 0)
            monthly_data['investment_withdrawal'].append(investment_withdrawal / inflation_factor)

        # Annual Aggregation
        years = [f"{y}-{y+1}" for y in range(simulation_start_year, death_year)]
        annual_data = {key: [sum(monthly_data[key][i*12:(i+1)*12]) for i in range(len(years))] for key in monthly_data.keys()}
        for key in ['offset', 'investment', 'super_person1', 'super_person2']:
            annual_data[key] = [monthly_data[key][(i+1)*12-1] for i in range(len(years))]

        # Retirement Salary Components Plot
        st.subheader("Retirement Salary Components (2024 AUD)")
        fig = go.Figure()
        retirement_start_idx = years_to_retirement
        plot_years = years[retirement_start_idx:]
        fig.add_trace(go.Bar(
            x=plot_years,
            y=annual_data['super_withdrawal_person1'][retirement_start_idx:],
            name="Person 1 Super",
            marker_color='blue'
        ))
        if household_size == 2:
            fig.add_trace(go.Bar(
                x=plot_years,
                y=annual_data['super_withdrawal_person2'][retirement_start_idx:],
                name="Person 2 Super",
                marker_color='green'
            ))
        fig.add_trace(go.Bar(
            x=plot_years,
            y=annual_data['investment_withdrawal'][retirement_start_idx:],
            name="Investment Income",
            marker_color='orange'
        ))
        fig.add_trace(go.Bar(
            x=plot_years,
            y=annual_data['pension'][retirement_start_idx:],
            name="Age Pension",
            marker_color='purple'
        ))
        fig.update_layout(
            barmode='stack',
            title="Annual Retirement Salary by Component",
            xaxis_title="Year",
            yaxis_title="AUD (2024 Dollars)",
            xaxis_tickangle=45,
            legend=dict(x=0, y=1.1, orientation='h'),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # Original Plots
        plot_configs = [
            ('expenses', 'Expenses', 'Annual Expenses (2024 AUD)'),
            ('income', 'Income', 'Annual Income (2024 AUD)'),
            ('savings', 'Savings', 'Annual Savings (2024 AUD)'),
            ('super_person1', 'Superannuation (Person 1)', 'Superannuation Balance Person 1 (2024 AUD)'),
            ('super_person2', 'Superannuation (Person 2)', 'Superannuation Balance Person 2 (2024 AUD)'),
            ('investment', 'Investments', 'Investment Balance (2024 AUD)'),
            ('offset', 'Offset Balance', 'Mortgage Offset Balance (2024 AUD)'),
            ('sg_contribution', 'SG Contribution', 'Annual SG Contribution (2024 AUD)'),
            ('salary_sacrifice', 'Salary Sacrifice', 'Annual Salary Sacrifice (2024 AUD)'),
            ('super_withdrawal_person1', 'Super Withdrawal (Person 1)', 'Annual Super Withdrawal Person 1 (2024 AUD)'),
            ('super_withdrawal_person2', 'Super Withdrawal (Person 2)', 'Annual Super Withdrawal Person 2 (2024 AUD)'),
            ('investment_withdrawal', 'Investment Withdrawal', 'Annual Investment Withdrawal (2024 AUD)')
        ]
        for key, label, title in plot_configs:
            if key == 'super_person2' and household_size == 1:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(years, annual_data[key], label=label, color='blue')
            ax.set_xticks(range(0, len(years), max(1, len(years)//10)))
            ax.set_xticklabels([years[i] for i in range(0, len(years), max(1, len(years)//10))], rotation=45)
            ax.set_xlabel('Year (March to February)')
            ax.set_ylabel('AUD (2024 Dollars)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # Summary
        st.subheader("Summary")
        st.write(f"Mortgage paid off in: {mortgage_payoff_year if mortgage_payoff_year else 'Not paid off'}")
        total_super = annual_data['super_person1'][retirement_start_idx] + (annual_data['super_person2'][retirement_start_idx] if household_size == 2 else 0)
        st.write(f"Total super at retirement: {total_super:.2f} AUD (2024 dollars)")
        st.write(f"Investment balance at retirement: {annual_data['investment'][retirement_start_idx]:.2f} AUD (2024 dollars)")
        st.write(f"Target retirement salary: {retirement_salary:.2f} AUD/year (2024 dollars)")

        # Tables
        st.subheader("Annual Expenses Breakdown (2024 AUD)")
        expense_table_data = [
            [year, f"{annual_data['base_expenses'][i]:.0f}", f"{annual_data['childcare'][i]:.0f}", 
             f"{annual_data['baby_supplies'][i]:.0f}", f"{annual_data['food_kids'][i]:.0f}", 
             f"{annual_data['clothing_kids'][i]:.0f}", f"{annual_data['healthcare_kids'][i]:.0f}", 
             f"{annual_data['education'][i]:.0f}", f"{annual_data['tertiary'][i]:.0f}", 
             f"{annual_data['extracurricular'][i]:.0f}", f"{annual_data['misc_kids'][i]:.0f}", 
             f"{annual_data['mortgage'][i]:.0f}", f"{annual_data['super_withdrawal_person1'][i]:.0f}", 
             f"{annual_data['super_withdrawal_person2'][i]:.0f}", f"{annual_data['investment_withdrawal'][i]:.0f}", 
             f"{annual_data['expenses'][i]:.0f}"]
            for i, year in enumerate(years)
        ]
        expense_headers = ["Year", "Base Expenses", "Childcare", "Baby Supplies", "Food (Kids)", "Clothing (Kids)", 
                           "Healthcare (Kids)", "Education", "Tertiary", "Extracurricular", "Misc (Kids)", "Mortgage", 
                           "Super Withdrawal (Person 1)", "Super Withdrawal (Person 2)", "Investment Withdrawal", "Total Expenses"]
        st.dataframe(pd.DataFrame(expense_table_data, columns=expense_headers))

        st.subheader("Yearly Financial Summary (2024 AUD)")
        table_data = [
            [y, f"{d['expenses']:.0f}", f"{d['income']:.0f}", f"{d['savings']:.0f}", f"{d['offset']:.0f}",
             f"{d['investment']:.0f}", f"{d['principal_paid']:.0f}", f"{d['super_person1']:.0f}",
             f"{d['super_person2']:.0f}", f"{d['tax']:.0f}", f"{d['pension']:.0f}",
             f"{d['sg_contribution']:.0f}", f"{d['salary_sacrifice']:.0f}",
             f"{d['super_withdrawal_person1']:.0f}", f"{d['super_withdrawal_person2']:.0f}",
             f"{d['investment_withdrawal']:.0f}"]
            for y, d in zip(years, [{k: v[i] for k, v in annual_data.items()} for i in range(len(years))])
        ]
        headers = ["Year", "Expenses", "Income", "Savings", "Offset", "Investment", "Principal Paid",
                   "Super (Person 1)", "Super (Person 2)", "Tax", "Pension", "SG Contribution",
                   "Salary Sacrifice", "Super Withdrawal (Person 1)", "Super Withdrawal (Person 2)",
                   "Investment Withdrawal"]
        st.dataframe(pd.DataFrame(table_data, columns=headers))

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see the results.")

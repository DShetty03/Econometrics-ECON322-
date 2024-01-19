import numpy as np
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import pandas as pd
import wooldridge as woo


data = pd.read_csv('fertility.csv')

#Question 1 
print("Question 1\n")

# Regression
results = model = smf.ols(formula='agefbrth ~ ceb + monthfm + idlnchld + educ', data=data).fit()
summary = model.summary()
print(summary)

coefficients = model.params

# Extract the estimated slope for 'educ'
estimated_slope_educ = coefficients['educ']

# Report the estimated slope for 'educ'
print(f"\nEstimated slope for 'educ': {estimated_slope_educ: .3f}\n")

#Question 2
print("Question 2\n")

def test_instrument_relevance(data, endogenous_var, instrument, exogenous_vars):
    # Testing Reg Model
    formula = f"{endogenous_var} ~ {' + '.join(exogenous_vars + [instrument])}"

    # Run testing regression
    reg_test = smf.ols(formula=formula, data=data)
    results_test = reg_test.fit(cov_type='HC3')

    # Robust t stat and respective p-value for the instrument ('electric')
    t_stat_instrument = results_test.tvalues[instrument]
    p_value_instrument = results_test.pvalues[instrument]

    # The formula used for the testing regression model
    print(f"Formula for the testing regression model: {formula}\n")

    # The robust t statistics and the corresponding p-value for 'electric'
    print(f"Robust t-statistic for 'electric': {t_stat_instrument:.3f}")
    print(f"P-value for 'electric': {p_value_instrument:.3f}\n")

# Run the function with your data and variables
test_instrument_relevance(data, 'educ', 'electric', ['ceb', 'monthfm', 'idlnchld'])




#Question 3 (Pt 1)
print("Question 3 (Pt.1)\n")

# Test for endogeneity
data['educ_resid'] = results.resid
reg_endo = smf.ols(formula='agefbrth ~ educ + educ_resid + ceb + monthfm + idlnchld', data=data)
results_endo = reg_endo.fit(cov_type='HC3')
print(results_endo.summary())

# t-test for educ_resid
t_test_endo = results_endo.t_test("educ_resid = 0")
print(t_test_endo)


#Question 3 (Pt. 2)
print("Question 3 (Pt. 2)\n")
# IV (2SLS) regression
reg_iv_2sls = IV2SLS.from_formula(formula = 'agefbrth ~ 1 + [educ~electric] + ceb + monthfm + idlnchld', data=data)
results_iv_2sls = reg_iv_2sls.fit(cov_type='robust', debiased=True)
print(results_iv_2sls)
    
 # Estimated slope for 'educ' in the IV (2SLS) regression
slope_educ_iv_2sls = results_iv_2sls.params['educ']
print(f"Estimated slope for 'educ' in 2SLS regression: {slope_educ_iv_2sls:.3f}\n")


#Question 4
print("Question 4 \n")
reg_iv = IV2SLS.from_formula(formula='agefbrth ~ 1 + [educ ~ electric + urban] + ceb + monthfm + idlnchld', data=data)
results_iv = reg_iv.fit(cov_type='robust', debiased=True)
print(results_iv)

#slope of educ with 2 instruments
slope_educ = results_iv.params['educ']
print(f"\nEstimated slope for 'educ' with multiple instruments: {slope_educ:.3f}\n")

#Overidentification
print(results_iv.wooldridge_overid)

import numpy as np
import wooldridge as woo
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS


# %% David Card's dataset

card = woo.dataWoo('card')

# first stage regression
reg_1st = smf.ols(formula='educ ~ nearc4', data=card)
results_1st = reg_1st.fit(cov_type='HC3')
print(results_1st.summary()) 

# second stage regression
card['educ_hat'] = results_1st.fittedvalues
reg_2nd = smf.ols(formula='lwage ~ educ_hat', data=card)
results_2nd = reg_2nd.fit(cov_type='HC3')
print(results_2nd.summary()) 

# IV directly
reg_iv = IV2SLS.from_formula(formula='lwage ~ 1 + [educ ~ nearc4]', data=card)
results_iv = reg_iv.fit(cov_type='robust', debiased=True)
print(results_iv)

# you can compare the estimate slope of educ in the 2nd-stage regression and that from the IV regression
# check the corresponding standard errors

# test for endogeneity
card['resid'] = results_1st.resid
reg_endo = smf.ols(formula='lwage ~ educ + resid', data=card)
results_endo = reg_endo.fit(cov_type='HC3')
print(results_endo.summary())

results_endo.t_test("resid = 0")


# %% More exogenous variables

reg_iv = IV2SLS.from_formula(formula='lwage ~ 1+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669+\
                                [educ ~ nearc4]', data=card)
results_iv = reg_iv.fit(cov_type='robust', debiased=True)
print(results_iv)

# comparison
reg_OLS = smf.ols(formula='lwage ~ exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669+educ', data=card)
results_OLS = reg_OLS.fit(cov_type='HC3')
print(results_OLS.summary())

# test for relevance
reg= smf.ols(formula='educ ~ nearc2+nearc4+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669', data=card)
results = reg.fit(cov_type='HC3')
print(results.summary()) 

print(results.f_test('nearc2=0, nearc4=0'))

# one more instrument
reg_iv = IV2SLS.from_formula(formula='lwage ~ 1+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669+\
                                [educ ~ nearc4 + nearc2]', data=card)
results_iv = reg_iv.fit(cov_type='robust', debiased=True)
print(results_iv)

results_iv.wooldridge_overid

# %% The Mroz dataset

mroz = woo.dataWoo('mroz')
mroz = mroz.dropna(subset=['lwage'])

reg_1st = smf.ols(formula='educ ~ exper + I(exper**2) + motheduc + fatheduc', data=mroz)
results_1st = reg_1st.fit(cov_type='HC3')
print(results_1st.summary())

# test for endogeneity
mroz['resid'] = results_1st.resid
reg_endo = smf.ols(formula='lwage ~ exper + I(exper**2) + educ + resid', data=mroz)
results_endo = reg_endo.fit(cov_type='HC3')
print(results_endo.summary())

results_endo.t_test("resid = 0")

# overidentification test with two instruments
reg_iv = IV2SLS.from_formula(formula='lwage ~ 1 + exper + I(exper**2) + [educ ~ motheduc + fatheduc]', data=mroz)
results_iv = reg_iv.fit(cov_type='robust', debiased=True)
print(results_iv)

results_iv.wooldridge_overid

# overidentification test with three instruments
reg_iv = IV2SLS.from_formula(formula='lwage ~ 1 + exper + I(exper**2) + [educ ~ motheduc + fatheduc + huseduc]', data=mroz)
results_iv = reg_iv.fit(cov_type='robust', debiased=True)
print(results_iv)

results_iv.wooldridge_overid


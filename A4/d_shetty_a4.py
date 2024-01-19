import pandas as pd
import statsmodels.formula.api as smf 
import scipy.stats as stats


df = pd.read_csv("CASchools.csv")
dg = pd.read_csv("employment_08_09.csv")

#NS dummy variable
df['NS'] = (df['str_s'] > 25).astype(int)

BayAreaCounties = ["San Francisco", "San Mateo", "Santa Clara",
                   "Contra Costa", "Marin", "Alameda", "Solano",
                   "Sonoma", "Napa", "Santa Cruz", "San Benito"]

df['BayArea'] = df['countyname'].apply(lambda text: text in BayAreaCounties)
df['NS_Bay'] = df['NS'] * df['BayArea']
df['NS_NonBay'] = df['NS'] * (1-df['BayArea'])
df['S_Bay'] = (1-df['NS']) * df['BayArea']
df['S_NonBay'] = (1-df['NS']) * (1-df['BayArea'])

#Question 1(a)
print("Question 1 \n")

#avg testscore for all schools
avg_test = df['testscore'].mean()

#avg testcore for schools with large class size
avg_large = df[df['NS']==1]['testscore'].mean()

#avg testscofe for schools with small class size
avg_small = df[df['NS']==0]['testscore'].mean()

diff_avg = avg_large - avg_small

print(f"Average testscore for all schools: {avg_test:.3f}")
print(f"Average testscore for schools with a relatively large class size: {avg_large:.3f}")
print(f"Average testscore for schools with a small class size: {avg_small:.3f}")
print(f"Difference in averages (large-class minus small-class): {diff_avg:.3f} \n")

#Question 1(b)
print("Question 1(b) \n")

model = smf.ols(formula = 'testscore ~ NS', data = df).fit()

print(model.summary())

slope = model.params['NS']
intercept = model.params['Intercept']

#Statistical Significance
p_value = model.pvalues['NS']

# 95% CI.
ci = model.conf_int(alpha=0.05)
lower = (round(ci.loc['NS', 0], 3))
upper = (round(ci.loc['NS', 1], 3))

print(f"\nEstimated Slope: {slope: .3f}")
print(f"Estimated Intercept: {intercept: .3f}")
print(f"The p-value, {p_value: .3f}, is greater than 0.05, thus the estimated slope is not statistically significant")
print(f"The 95% Confidence Interval is: [{lower}, {upper}]\n")

#Question 1(c)
print("Question 1(c)\n")

def get_t_and_p_values(model, variable_name):
    t_value = abs(model.tvalues[variable_name])
    p_value = model.pvalues[variable_name]
    return t_value, p_value

model_S_Bay = smf.ols(formula='testscore ~ S_Bay', data=df).fit()
t_S_Bay, p_S_Bay = get_t_and_p_values(model_S_Bay, 'S_Bay')

model_NS_NonBay = smf.ols(formula='testscore ~ NS_NonBay', data=df).fit()
t_NS_NonBay, p_NS_NonBay = get_t_and_p_values(model_NS_NonBay, 'NS_NonBay')

model_NS_Bay = smf.ols(formula='testscore ~ NS_Bay', data=df).fit()
t_NS_Bay, p_NS_Bay = get_t_and_p_values(model_NS_Bay, 'NS_Bay')

print(f"S_Bay: t-value = {t_S_Bay:.3f}, p-value = {p_S_Bay:.3f}")
print(f"NS_NonBay: t-value = {t_NS_NonBay:.3f}, p-value = {p_NS_NonBay:.3f}")
print(f"NS_Bay: t-value = {t_NS_Bay:.3f}, p-value = {p_NS_Bay:.3f} \n")

# Question 2
print("Question 2 \n")

#Question 2(a)
print("Question 2(a) \n")

#create age^2
dg['age2']= dg['age']**2

#regression model
model = smf.ols(formula = 'employed ~ age+age2', data = dg).fit()

print(model.summary())

#F-test
f_test = model.f_test("age =0 + age2 =0")
print("\nFormal F-test:")
print(f"F-statistic: {f_test.fvalue:.3f}")
print(f"P-value: {f_test.pvalue:.3f} \n")

# predicted prob
ages = [20, 40, 60]
predicted_prob = model.predict(pd.DataFrame({'age': ages, 'age2': [age**2 for age in ages]}))

#print
for age, prob in zip(ages, predicted_prob):
    print(f"Predicted Probability for {age}-year-old worker: {prob:.2f}")
    

# Question 2(b)
print("\nQuestion 2(b)\n")

#create age^2
dg['age2'] = dg['age']**2

#Logit regression model 
logit_model = smf.logit(formula='employed ~ age + age2', data=dg).fit(disp=False)
print(logit_model.summary())

# Likelihood ratio test (F-Test)
restricted_model = smf.logit(formula='employed ~ age', data=dg).fit(disp=False)

# likelihood ratio test stat
lr_stat = 2 * (logit_model.llf - restricted_model.llf)

# df
df = logit_model.df_model - restricted_model.df_model

# p-value
p_value = 1 - stats.chi2.cdf(lr_stat, df)

print("\nLikelihood Ratio Test:")
print(f"Likelihood Ratio Test Statistic:{lr_stat: .3f}")
print(f"P-value: {p_value: .3f}\n")

# predict probablities
ages = [20, 40, 60]
predicted_prob_logit = logit_model.predict(pd.DataFrame({'age': ages, 'age2': [age**2 for age in ages]}))

#print
for age, prob in zip(ages, predicted_prob_logit):
    print(f"Predicted Probability for {age}-year-old worker: {prob:.2f}")











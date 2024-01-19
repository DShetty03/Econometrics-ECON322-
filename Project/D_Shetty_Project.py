import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv("CollegeDistance.csv")

# Question 1
print("\nQuestion 1 \n")

model = smf.ols(formula ='ed ~ dist', data = df).fit()
print(model.summary())
#Specifically, for every ten miles closer the college is to the high school, 
# the average years of completed schooling tends to decrease by approximately 0.734 years.

# Question 1(a)
print("Question 1(a) \n")
intercept = model.params['Intercept']
slope = model.params['dist']

print(f"Intercept: {intercept: .4f} \n ")
print(f"Slope: {slope: .4f} \n")

# Question 1(b)
print("Question 1(b) \n")

bob_completed = model.predict({"dist": 2})
bob_changed = model.predict({"dist": 1})

print(f"If Bob's HS was 20 miles away from the nearest college, then its predicted that he completed {bob_completed.iloc[0]:.4f} years of education \n")
print(f"If Bob lived 10 miles from the nearest college, the prediction changes to {bob_changed.iloc[0]:.4f} years \n ")

# Question 1(c) on pdf
# Question 1(d) on pdf

# Question 1(e)
print("Question 1(e) \n")

confidence_interval = model.conf_int(alpha = 0.05) # significance level denoted by alpha. In this instance, because alpha = 0.05, signficance level = 5%
# signficance level = alpha. Helps determine level of cofidence. The choice of alpha is typically based on the level of confidence you want to achieve for your statistical analysis.
# The common choices for alpha are 0.05, 0.01, and 0.10, corresponding to significance levels of 5%, 1%, and 10%, respectively.

lower = confidence_interval.loc["dist"][0]
upper = confidence_interval.loc["dist"][1]

print(f"A 95% confidence interval for the slope coefficient is: ({lower: .4f}, {upper: .4f}) \n")

# Question 1(f)
print("Question 1(f) \n")

edu = model.predict({"dist": -2})
difference = edu[0] - intercept
print(f"If distance to the nearest college is decreased by 20 miles, educational attainment is {edu[0]: .4f}. This is {difference:.4f} greater than the intercept which is approximately equal to 0.15. \n")

# Question 2

print("Question 2 \n")
model = smf.ols(formula ='ed ~ dist + bytest + female + black + hispanic + incomehi + ownhome + dadcoll + cue80 + stwmfg80', data = df).fit()
print(model.summary())

dist_effect = model.params['dist']

print(f"\nThe effect of Dist on ED is {dist_effect: .4f}. \nThis means that for every 10 mile increase in dist, educational attainment decreases by approx{abs(dist_effect): .4f}. \n")

# Question 2(a)
print("Question 2(a) \n")
print(f"The estimated effect in the regression in Q1 is {slope: .4f} while the estimated effect in Q2 is {dist_effect: .4f} \n")

# Question 2(b) on pdf. 

#Question 2(c)
print("Question 2(c) \n")
cue80 = model.params["cue80"]
swmgf80 = model.params["stwmfg80"]
print(f"Cue80:{cue80: .4f} \nStwmfg80:{swmgf80: .4f}\n")

# Question 2(d)
print("Question 2(d) \n")
bob = {
    'dist': 2,
    'female': 0,
    'black': 1,
    'hispanic': 0,
    'bytest': 58,
    'incomehi': 1,
    'ownhome': 1,
    'momcoll': 1,
    'dadcoll': 0,
    'cue80': 0.075,
    'stwmfg80': 9.75,
}

bob_predict = model.predict(bob)

print(f"Bob's years of schooling are {bob_predict[0]: .4f} \n")

# Question 2(e)
print("Question 2(e)\n")

jim = {
    'dist': 4,
    'female': 0,
    'black': 1,
    'hispanic': 0,
    'bytest': 58,
    'incomehi': 1,
    'ownhome': 1,
    'momcoll': 1,
    'dadcoll': 0,
    'cue80': 0.075,
    'stwmfg80': 9.75,
}
jim_predict = model.predict(jim)

print(f"Jim's years of schooling are {jim_predict[0]: .4f} \n")

# Question 2(f)
print("Question 2(f) \n")

black = {
    'dist': 0,
    'female': 0,
    'black': 1,
    'hispanic': 0,
    'bytest': 0,
    'incomehi': 0,
    'ownhome': 0,
    'momcoll': 0,
    'dadcoll': 0,
    'cue80': 0,
    'stwmfg80': 0,
    
    
}
black_predict = model.predict(black)
print(f"Blacks complete{black_predict[0]: .4f} years of education \n")

hispanic = {
    'dist': 0,
    'female': 0,
    'black': 0,
    'hispanic': 1,
    'bytest': 0,
    'incomehi': 0,
    'ownhome': 0,
    'momcoll': 0,
    'dadcoll': 0,
    'cue80': 0,
    'stwmfg80': 0,
}
hispanic_predict = model.predict(hispanic)
print(f"Hispanics complete{hispanic_predict[0]: .4f} years of education \n")

whites = model.params['Intercept']
print(f"Whites complete {whites: .4f} years of education")




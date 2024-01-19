import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("Birthweight_Smoking.csv")

# Question 1: 
print("Question 1 \n")

model = smf.ols(formula ='birthweight ~ smoker', data = df).fit()
print(model.summary())

# What is the estimated effect of smoking on birth weight? 
print("What is the estimated effect of smoking on birthweight? ")

# Get coefficient of smoking 
effect = model.params['smoker']
print(f"The effect of smoking on birthweight is that infants born to mothers who smoke have a birthweight that is lower by{abs(effect): .2f} grams. \n") #abs to make sentence make sense. Coefficient was negative

# Question 2
print("Question 2 \n")

# Regress Birthweight on Smoker, Alcohol, and Nprevist.
model = smf.ols(formula='birthweight ~ smoker + alcohol + nprevist', data=df).fit()
print(model.summary())

#Question 2a & 2b on pdf

# Question 2(c)
print ("Question 2(c)")

#create Jane
jane = {
    'smoker': 1,
    'alcohol': 0,
    'nprevist': 8
}

jane_df = pd.DataFrame([jane])

predict = model.predict(jane)
int_predict = int(round(predict.iloc[0]))

print(f"The predicted birthweight of Jane's child is {int_predict} grams \n ")


# Question 2(d)
print("Question 2(d)")
# Find R^2 and R^2 (adjusted R-squared). Why are they so similar?

r_squared = model.rsquared
r_adjusted = model.rsquared_adj

print(f"R-squared: {r_squared:.3f}")
print(f"Adjusted R-squared: {r_adjusted:.3f} \n")


# Question 2(e): Interpretation on pdf
print("Question 2(e)")
# find slope of nprevist
coefficient_nprevist = model.params['nprevist']
print(f"Coefficient (slope) of nprevist:{coefficient_nprevist: .2f} \n")

# Question 3
print("Question 3 \n")

#Regress Birthweight on Smoker, Alcohol, Tripre0, Tripre2, and Tripre3.
model = smf.ols(formula='birthweight ~ smoker + alcohol + nprevist + tripre0 + tripre2 + tripre3', data=df).fit()
print(model.summary())

# Questions 3(a) & 3(b) are on pdf




 
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# You need to replace the input file name by a full path (e.g., r"C:\user\Documents\Earnings_and_Height.csv")
# if the csv file is not saved under your working directory
df = pd.read_csv("Earnings_and_Height.csv")

# Question 1: What are the mean values of height and earnings, respectively? Round up the numbers up to 2 decimal points.
print("Question 1")

mean_height = df['height'].mean()
mean_earnings = df['earnings'].mean()

print(f"Mean values of height:{mean_height: .2f}") # f creates an f string that allows us to embed expressions inside string literals
print(f"Mean values of earnings:{mean_earnings: .2f}") # .2f formats the value to 2 deccimal places

# Question 2: Draw a scatterplot of annual earnings (Earnings) on height (Height).
print("Question 2")
print("Scatterplot is on the pdf")

def draw_scatterplot(dataframe, x_column, y_column):
    # plt.figure creates a new figure (and specifies figure size) or modifies properties of existing figure.
    #figure is a top level container for all the plot elements in a plot (ie; aces, labels, titles, subplots)
    plt.figure(figsize=(8,6)) #8 = width, 6 = height
    # plt.scatter used to create scatter plots
    plt.scatter(dataframe[x_column], dataframe[y_column], alpha = 0.5) #alpha controls transparency of the markers of scatterplot. 0.5 = semi transparent markers
    # add labels and titles
    plt.title(" Annual Earnings over Height")
    plt.xlabel("Height")
    plt.ylabel("Earnings")
    # plt.grid adds gridlines. Make it easier to read and interpret data
    # Enable gridlines by calling plt.grid(True). Remove them by calling plt.grid(False)
    plt.grid(True)
    plt.show()

draw_scatterplot(df, 'height', 'earnings')

# Question 3: Run a regression of Earnings on Height
print("Question 3")

model = smf.ols(formula='earnings ~ height', data=df).fit()
print(model.summary())

# Question 3(a)
print("Question 3(a)")
sample_size = model.nobs
print(f"Sample Size:{sample_size: .0f} ")

# Question 3(b)
print("Question 3(b)")
r_squared = model.rsquared
print(f"R-squared:{r_squared: .2f}")

# Question 3(c)
print("Question 3(c)")
estimated_slope = model.params['height']
print(f"Estimated Slope:{estimated_slope: .2f}")

# Question 3(d)
print("Question 3(d)")
predict_earnings = model.predict({'height': 70})
print(f"Predicted Earnings for 70 inches tall worker:{predict_earnings[0]: .2f}")

# Question 4: Run a regression of Earnings on Height, using data for female workers only.
print("Question 4")

female_workers = df[df['sex'] == 1]  

model = smf.ols(formula='earnings ~ height', data=female_workers).fit()
print(model.summary())

# Question 4(a)
print("Question 4(a)")
sample_size = model.nobs
print(f"Sample Size: {sample_size: .0f}")

# Question 4(b)
print("Question 4(b)")
r_squared = model.rsquared
print(f"R-Squared: {r_squared: .2f}")

# Question 4(c)
print("Question 4(c)")
estimated_slope = model.params['height']
print(f"Estimated Slope:{estimated_slope: .2f}")

#Question 4(d)
print("Question 4(d)")

avg_height = female_workers['height'].mean()
avg_earnings = female_workers['earnings'].mean()
taller_woman = avg_height + 1

predict_earnings = model.predict({'height': taller_woman})

# if higher or lower than the average earnings
if predict_earnings.iloc[0] > avg_earnings:
    result = "higher"
else:
    result = "lower"

difference = abs(predict_earnings.iloc[0] - avg_earnings)

print(f"The predicted earnings for a woman 1 inch taller than the average are {result} than the average earnings for women by ${difference:.2f}.")








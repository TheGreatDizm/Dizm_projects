# Coefficient
# The coefficient is a factor that describes the relationship with an unknown variable.
#
# Example: if x is a variable, then 2x is x two times. x is the unknown variable, and the number 2 is the coefficient.
#
# In this case, we can ask for the coefficient value of weight against CO2, and for volume against CO2. The answer(s) we
# get tells us what would happen if we increase, or decrease, one of the independent values.

import pandas as pd
from sklearn import linear_model

df = pd.read_csv("C:\\Users\\esg\\PycharmProjects\\Multiple_Regression\\data (1).csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)

# Result Explained
# The result array represents the coefficient values of weight and volume.
#
# Weight: 0.00755095
# Volume: 0.00780526
#
# These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.
#
# And if the engine size (Volume) increases by 1 cm3, the CO2 emission increases by 0.00780526 g.
#
# I think that is a fair guess, but let test it!
#
# We have already predicted that if a car with a 1300cm3 engine weighs 2300kg, the CO2 emission will be approximately 107g.

predictedCO2 = regr.predict([[3300, 1300]])

print(predictedCO2)

# We have predicted that a car with 1.3 liter engine, and a weight of 3300 kg, will release approximately 115 grams of CO2 for every kilometer it drives.
#
# Which shows that the coefficient of 0.00755095 is correct:
#
# 107.2087328 + (1000 * 0.00755095) = 114.75968
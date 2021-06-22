import numpy as np
import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('FuelConsumptionCo2.csv')
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Pandas head() method is used to return top n (5 by default) rows of a data frame or series
# Here 9 refers to the top 9 rows which will be printed

print(cdf.head(9))

# plt. show() starts an event loop, looks for all currently active figure objects, and opens one or more interactive
# windows that display your figure or figures.

# cdf.hist()
# plt.show()


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()

# asanyarray() function is used when we want to convert input to an array
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

print('Coefficient:', regr.coef_)
print('Intercept:', regr.intercept_)

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="red")
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-b')
plt.xlabel("Engine Size")
plt.ylabel("Co2 Emissions")
plt.show()




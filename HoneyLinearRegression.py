import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())

#using groupby to get the mean of totalprod per year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()


#creating our X and y values for the linear regression and then reshaping it to the right gormat
X = prod_per_year.year
X = X.values.reshape(-1, 1)
y = prod_per_year.totalprod

#plotting y vs x as a scatterplot
plt.scatter(X, y)


#creating our linear regression model, fitting the line, and predictions
regr = linear_model.LinearRegression()
regr.fit(X, y)
y_predict = regr.predict(X)

plt.plot(X, y_predict)
#looks like honey production has been in decline since 1998. Next, going to look at when honey production might end

X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)
plt.plot(X_future, future_predict)
print(future_predict[-1])
#by 2050, honey production would be at 186,000 pounds per year.


plt.show()
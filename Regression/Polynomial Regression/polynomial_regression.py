# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Linear regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
poly_pred = lin_reg2.predict(X_poly)

#Visualising the Linear Regression results

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position lvl')
plt.ylabel('Salary')

#Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'green')

# Predicting a new result with Linear Regression
lin_pred = lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
poly_pred = lin_reg2.predict(poly_reg.fit_transform(6.5))


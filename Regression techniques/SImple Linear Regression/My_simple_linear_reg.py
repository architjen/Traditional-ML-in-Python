# Simple linear Reg
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
#No feat scal in simpl reg, lib auto takes care 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting simple linear reg in training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prediciting test se results
y_pred = regressor.predict(X_test)

#Visualizing training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue') #forming a predicted line using the training examples
plt.title('sal vs exp(train set)')
plt.xlabel('years of exp')
plt.ylabel('sal')
plt.show()

#visualizing test set result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue') #same as of line 35, coz we are forming the hypothesis using training data
plt.title('sal vs exp(tesr set)')
plt.xlabel('years of exp')
plt.ylabel('sal')
plt.show()
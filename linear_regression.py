# Simple Linear Regression

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the saved_data_set
saved_data_set = pd.read_csv('Income_Data.csv')
X = saved_data_set.iloc[:, :-1].values
y = saved_data_set.iloc[:, 1].values

# Split the saved_data_set into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Perform Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fit Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test set results
y_pred = regressor.predict(X_test)

# Visualise the Training set results
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Income vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Income')
plt.show()

# Visualise the Test set results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.title('Income vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Income')
plt.show()
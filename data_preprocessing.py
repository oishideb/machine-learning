# Data Preprocessing

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the saved_data_set
saved_data_set = pd.read_csv('Data.csv')
X_axis = saved_data_set.iloc[:, :-1].values
y_axis = saved_data_set.iloc[:, 3].values

# Splitting the saved_data_set into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_axis, y_axis, test_size = 0.2, random_state = 0)

# Performing Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
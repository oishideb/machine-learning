# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the saved_data_set
saved_data_set = pd.read_csv('Data.csv')
x_axis = saved_data_set.iloc[:, :-1].values
y_axis = saved_data_set.iloc[:, 3].values

# Dealing with missing data
from sklearn.preprocessing import Imputer
imput = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imput = imput.fit(x_axis[:, 1:3])
x_axis[:, 1:3] = imput.transform(x_axis[:, 1:3])
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the saved_data_set
saved_data_set = pd.read_csv('Data.csv')
X = saved_data_set.iloc[:, :-1].values
y = saved_data_set.iloc[:, 3].values

# Dealing with missing data
from sklearn.preprocessing import Imputer
imput = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imput = imput.fit(X[:, 1:3])
X[:, 1:3] = imput.transform(X[:, 1:3])

# Encoding categorical data and the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
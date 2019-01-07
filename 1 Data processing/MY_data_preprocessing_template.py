# Data processing

# IMporting libraries
import numpy as np     #Maths 
import matplotlib.pyplot as plt    #Plotting 
import pandas as pd    #IMport and manage dataset

# importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  #always matrix
Y = dataset.iloc[:, 3].values   #always vector
# taking care of missing values
"""
from sklearn.preprocessing import Imputer #imputer used for missing values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0) #axis 0 is for col and 1 for row
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3]) """

#encoding categrical data
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)"""

#Split data in train_set and Test_set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)



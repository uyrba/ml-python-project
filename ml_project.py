import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error

#Load data from the csv file.
#https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17
data = pd.read_csv('star_classification.csv')

print('Original number of data points: ' + str(np.shape(data)[0]))

#Drop unneeded columns.
data.drop(columns=['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID'], inplace=True)

#Select only star observations.
data = data[data['class'] == 'STAR']

print('Number of data points after filtering: %d\n' % np.shape(data)[0])

#Separate features (X) and labels (y) from the data.
X = data[['u', 'g', 'r', 'i', 'z']].to_numpy()
y = data['redshift'].to_numpy()

#Split data point to training, testing, and validation sets.
X_training, X_validation, y_training, y_validation = train_test_split(X, y, train_size=0.9, random_state=42)

print('data points for training: %d\ndata points for validation: %d\n' % (X_training.shape[0], X_validation.shape[0]))

#The degree of the polynomial model used.
degree = 12

#Generate polynomial features.
poly = PolynomialFeatures(degree=degree)
X_training_poly = poly.fit_transform(X_training)
X_validation_poly = poly.fit_transform(X_validation)

#Apply regression model.
regression = HuberRegressor()
regression.fit(X_training_poly, y_training)

#Calculate training error.
y_predicted = regression.predict(X_training_poly)
print('Training error: ' + str(mean_squared_error(y_training, y_predicted)))

#Calculate validation error
y_predicted = regression.predict(X_validation_poly)
print('Validation error: ' + str(mean_squared_error(y_validation, y_predicted)))
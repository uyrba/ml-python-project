import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error

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
X_training, X_remaining, y_training, y_remaining = train_test_split(X, y, train_size=0.9, random_state=42)
X_validation, X_testing, y_validation, y_testing = train_test_split(X_remaining, y_remaining, train_size=0.5, random_state=42)

print('data points for training: %d\ndata points for validation: %d\ndata points for testing: %d\n' % (X_training.shape[0], X_validation.shape[0], X_testing.shape[0]))

#The degrees of the polynomial models used.
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

#A list for storing the validation errors.
validation_errors = []

#Try out all listed degrees
i = 0
while(i < len(degrees)):
    #Generate polynomial features.
    poly = PolynomialFeatures(degree=degrees[i])
    X_training_poly = poly.fit_transform(X_training)
    X_validation_poly = poly.fit_transform(X_validation)

    #Apply regression model.
    regression = HuberRegressor(max_iter=200)
    regression.fit(X_training_poly, y_training)

    #Calculate training error.
    y_predicted = regression.predict(X_training_poly)
    print('Training error for degree %d polynomial: %s' % (degrees[i], str(mean_absolute_error(y_training, y_predicted))))

    #Calculate validation error
    y_predicted = regression.predict(X_validation_poly)
    validation_errors.append(mean_absolute_error(y_validation, y_predicted))
    print('Validation error for degree %d polynomial: %s' % (degrees[i], str(validation_errors[i])))

    i += 1

#Find the polynomial with the lowest validation error
min_index = 0
i = 1
while(i < len(validation_errors)):
    if(validation_errors[i] < validation_errors[min_index]):
        min_index = i
    i += 1

print('\nDegree %d polynomial has the lowest validation error (%s).' % (degrees[min_index], str(validation_errors[min_index])))

#Calculate testing error for the chosen polynomial model.
#Generate polynomial features.
poly = PolynomialFeatures(degree=degrees[min_index])
X_training_poly = poly.fit_transform(X_training)
X_testing_poly = poly.fit_transform(X_testing)

#Apply regression model.
regression = HuberRegressor(max_iter=200)
regression.fit(X_training_poly, y_training)

#Calculate training error.
y_predicted = regression.predict(X_testing_poly)
print('Testing error for degree %d polynomial: %s' % (degrees[min_index], str(mean_absolute_error(y_testing, y_predicted))))

#Calculate simple reference error using average value.
average = np.average(y)
print('Reference error: %s' % (str(mean_absolute_error(y, np.full(y.shape, average)))))
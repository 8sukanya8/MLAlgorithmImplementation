import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy


def lin_reg_train(data, predicted_col_num):
    y_train = data.iloc[ : ,predicted_col_num:(predicted_col_num + 1)]
    predictors = data.iloc[ : ,0:predicted_col_num]
    predictors_transpose = predictors.transpose()
    predictors_transpose_dot_predicted = predictors_transpose.dot(y_train)
    predictors_dot_predictors_transpose = predictors_transpose.dot(predictors)
    predictors_predictors_transpose_inverse = pd.DataFrame(np.linalg.pinv(predictors_dot_predictors_transpose.values),
                                                           predictors_dot_predictors_transpose.columns,
                                                           predictors_dot_predictors_transpose.index)
    beta = predictors_predictors_transpose_inverse.dot(predictors_transpose_dot_predicted)
    return beta


def lin_reg_test(data, predicted_col_num, beta):
    predictors = data.iloc[:, 0:predicted_col_num]
    y_pred = predictors.dot(beta)
    return y_pred


input_file_path = 'D:\PhD\MLImplementation\HousePricesNumerical.csv'
data = pd.read_csv(input_file_path)
train = data.sample(frac=0.7)
test = data.sample(frac=0.3)
beta = lin_reg_train(train, 2)
y_test = test.iloc[:, 2:3]
y_pred = lin_reg_test(test, 2, beta)
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print(rms)

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train.iloc[:, 0:2], train.iloc[:, 2:3])

# Make predictions using the testing set
y_pred_sklearn = regr.predict(test.iloc[:, 0:2])
rms_sklearn = np.sqrt(mean_squared_error(y_test, y_pred))
print(rms_sklearn)
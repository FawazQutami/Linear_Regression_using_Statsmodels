# File: Linear_Regression_using_Statsmodels.py

import time
import numpy as np

# scikit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split

# statsmodels
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")


def statsmodels_lr(X_trn, X_tst, y_trn):
    """Linear Regression using statsmodels """
    """
            Linear model: y = α + βX

            The parameters α and β of the model are selected through the Ordinary 
            least squares (OLS) method. It works by minimizing the sum of squares 
            of residuals (actual value - predicted value). """
    start = time.time()

    # Create the X matrix by appending a column of ones to x_train (Transpose)
    X = sm.add_constant(X_trn)

    # Build the OLS model (Ordinary Least Squares) from the training data
    reg = sm.OLS(y_trn, X)

    # Fit and save regression info (parameters, etc)
    lr_info = reg.fit()

    # Pull the α and β parameters
    """
        Coefficients are estimated using the least squares criterion. 
        In other words, we find the line (mathematically) which minimizes the 
        sum of squared residuals (or "sum of squared errors"):
    """
    alpha = lr_info.params[0]
    beta = lr_info.params[1:]
    # Print lr coefficients
    print("--- Intercept - α = ", alpha)
    print("--- Slope's - β's = ", beta)

    # Print the OLS model information
    print('\n', lr_info.summary())

    # Using the Model for Prediction
    predicted_y = lr_info.predict(lr_info.params, X_tst)

    end = time.time()
    print('Execution Time: {%f}' % ((end - start) / 1000)
          + ' seconds.')


if __name__ == '__main__':
    try:
        # Create regression data
        X, y = datasets.make_regression(n_samples=1000,
                                        n_features=3,
                                        noise=20,
                                        random_state=5)
        # Split the data to training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=5)

        # Test using statsmodels library
        print('\n ---------- LR using statsmodels library ----------')
        statsmodels_lr(X_train, X_test, y_train)

    except:
        pass



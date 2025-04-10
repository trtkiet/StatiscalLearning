import numpy as np
from numpy.random import default_rng
rng = default_rng()
from numpy.linalg import pinv
from copy import copy



# Linear Regression using LS method

class LinearRegressionModel:
    def __init__(self):
        super().__init__()
        self.beta = None
    def fit(self, X, y):
        self.beta = np.dot(np.dot(pinv(np.dot(X.T, X)), X.T), y)
    def predict(self, X):
        return np.dot(X, self.beta)

# RANSAC 

class RANSAC:
    def __init__(self, model = None, m = None, k = None, t = None, d = None, seed = None):
        self.m = m              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.largest = 0
        self.best_fit = None
        self.best_outliers = []
        self.maybe_inliers_set = []
        self.seed = seed

    def fit(self, X, y):
        data_size = X.shape[0]
        for _ in range(self.k):
            # Get n random data points from dataset
            ids = rng.permutation(data_size)
            maybe_inliers = ids[: self.m]
            if self.seed is not None:
                maybe_inliers = self.seed[_]
            self.maybe_inliers_set.append(maybe_inliers)
            maybe_model = copy(self.model)

            maybe_model.fit(X[maybe_inliers], y[maybe_inliers])

            outlier = []
            inlier = []
            # Classify data points as inlier or outlier
            for i in range(data_size):
                if self.SE(y[i], maybe_model.predict(X[i])) > self.t:
                    outlier.append(i)
                else:
                    inlier.append(i)

            # Consider if the number of data points that are classified as inliers is sufficient to fit a better model
            if len(inlier) > self.d:
                # Train a model with data points in inlier array
                better_model = copy(self.model)
                better_model.fit(X[inlier], y[inlier])

                if len(inlier) > self.largest:
                    self.largest = len(inlier)
                    self.best_maybe_inliers = maybe_inliers
                    self.best_outliers = outlier
                    self.best_fit = better_model


        return self
    
    
    def predict(self, X):
        return self.best_fit.predict(X)
    
    # Loss 
    def SE(self, y_true, y_pred):
        return (y_true - y_pred)**2



    

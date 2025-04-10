
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
    def __init__(self, m = None, k = None, t = None, d = None, seed = None, model = None, metric = None):
        self.m = m              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.metric = metric    # MSE: Mean Square Error   LCS: Largest consensus set 
        self.best_fit = None
        self.best_error = np.inf
        self.best_outliers = []
        self.best_inliers = []
        self.maybe_inliers_set = []
        self.inliers_set = []
        self.accept_inliers_set = []
        self.maybe_model = None
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
        
            inlier = []
            outlier = []

            # Classify data points as inlier or outlier
            for i in range(data_size):
                if self.SE(y[i], maybe_model.predict(X[i])) > self.t :
                    outlier.append(i)
                else:
                    inlier.append(i)
            inlier = np.array(inlier)
            outlier = np.array(outlier)
            self.inliers_set.append(inlier)

            # Consider if the number of data points that are classified as inliers is sufficient to fit a better model
            if len(inlier) > self.d:
                # Train a model with data points in inlier array
                better_model = copy(self.model)
                better_model.fit(X[inlier], y[inlier])
                yin = better_model.predict(X[inlier])
                this_error = self.MSE(y[inlier], yin) 
                if self.metric == "MSE":
                    if this_error < self.best_error:
                        self.maybe_model = maybe_model
                        self.best_maybe_inliers = maybe_inliers
                        self.best_outliers = outlier
                        self.best_error = this_error
                        self.best_fit = better_model
                        self.best_inliers = inlier
                if self.metric == "LCS":
                    if len(inlier) < len(self.best_inliers):
                        continue

                    if len(inlier) > len(self.best_inliers):
                        self.accept_inliers_set.clear()
                        
                    self.accept_inliers_set.append(inlier)
                    if len(inlier) > len(self.best_inliers) or this_error < self.best_error:
                        self.maybe_model = maybe_model
                        self.best_maybe_inliers = maybe_inliers
                        self.best_outliers = outlier
                        self.best_error = this_error
                        self.best_fit = better_model
                        self.best_inliers = inlier


        return self
    
    
    def predict(self, X):
        return self.best_fit.predict(X)
    
    # Loss and Metric
    def SE(self, y_true, y_pred):
        return (y_true - y_pred)**2

    def MSE(self, y_true, y_pred):
        return np.sum(self.SE(y_true, y_pred)) / y_true.shape[0]


    

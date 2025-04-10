from copy import copy
import multiprocessing
import concurrent.futures
import numpy as np
import random
from numpy.linalg import inv
from numpy.random import default_rng
rng = default_rng()
import matplotlib.pyplot as plt
from mpmath import mp

class LinearRegressionModel:
    def __init__(self):
        super().__init__()
        self.beta = None
    def fit(self, X, y):
        self.beta = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)
    def predict(self, X):
        return self.beta * X
    
class RANSAC:
    
    def __init__(self, n = 20, k = None, t = None, d = 0, model = None, loss = None, metric = None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.loss = loss        # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric    # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf
        self.best_outliers = []
        self.best_inliers = []
        self.maybe_inliers_set = []
        self.inliers_set = []
        self.accept_inliers_set = []

    def fit(self, X, y):
        
        for _ in range(self.k):
            # Get n random data points from dataset
            ids = rng.permutation(X.shape[0])
            maybe_inliers = ids[: self.n]
            self.maybe_inliers_set.append(maybe_inliers)
            maybe_model = copy(self.model)

            maybe_model.fit(X[maybe_inliers], y[maybe_inliers])

            inlier = []
            outlier = []
            # Classify data points as inlier or outlier
            for i in range(X.shape[0]):
                if self.loss(y[i], maybe_model.predict(X[i])) > self.t :
                    outlier.append(i)
                else:
                    inlier.append(i)
            inlier = np.array(inlier)
            outlier = np.array(outlier)
            self.inliers_set.append(inlier)

            # Consider if the number of data points that are classified as inliers is sufficient to fit a better model
            if inlier.size > self.d:
                self.accept_inliers_set.append(inlier)
                # Train a model with data points in inlier array
                better_model = copy(self.model)
                better_model.fit(X[inlier], y[inlier])

                # calcualte the error in this iteration
                yin = better_model.predict(X[inlier])                
                this_error = self.metric(y[inlier], yin) 

                if this_error < self.best_error:
                    self.best_maybe_inliers = maybe_inliers
                    self.best_outliers = outlier
                    self.best_error = this_error
                    self.best_fit = better_model
                    self.best_inliers = inlier
        return self
    
    def predict(self, X):
        return self.best_fit.predict(X)

# Loss and Metric
    
def absolute_error_loss(y_true, y_pred):
    return abs(y_true - y_pred)

def mean_absolute_error(y_true, y_pred):
    return np.sum(absolute_error_loss(y_true, y_pred)) / y_true.shape[0]

# Generate data

""" p: num_predictor
    q: outcome_dim
"""
def genX(sample_size, p, q):
    X = []
    for i in range(sample_size):
        x = 4 * (np.random.rand() - 0.5)   # Generate X in [-2, 2]
        X.append(x)
    
    return X

def geny(X, inlier_size, outlier_size, p, q):
    y = []
    for i in range(inlier_size):
        mu = X[i] * 5
        y.append(mu + np.random.normal(0, 1, 1)[0])

    for i in range(outlier_size):
        mu = X[i+inlier_size] * 10 * (np.random.rand() - 0.5)
        y.append(mu + np.random.normal(0, 1, 1)[0])

    return y

# Calculate V-, V+
def V(n, A, b, c, z):
    V_minus = np.NINF
    V_plus = np.Inf

    Ac = np.dot(A, c)
    Az = np.dot(A, z)

    for i in range(b.shape[0]):
        numerator = b[i][0] - Az[i][0]
        denominator = Ac[i][0]

        if(denominator == 0):
            if(numerator < 0): 
                print("Error")

        else:
            tmp = numerator/denominator
            
            if(denominator < 0) :
                V_minus = max(V_minus, tmp)
            else :
                V_plus = min(V_plus, tmp)

    return V_minus, V_plus

# Selective p-value

""" p: num_predictor
    q: outcome_dim
    n: sample_size
"""

def check(model, p, q, n, index, t, X, y):
    best_inliers = model.best_inliers
    maybe_inliers_set = model.maybe_inliers_set
    inliers_set = model.inliers_set
    accept_inliers_set = model.accept_inliers_set

    Sigma = np.identity(n)
    
    # construct eta
    vec_index = np.zeros((n,1))
    vec_index[index][0] = 1
    xi = X[index].reshape((p,1))

    MatI = np.zeros((n,n))
    for i in best_inliers:
        MatI[i][i] = 1

    XI = np.dot(MatI, X)
    e1 = vec_index.T 
    inve = inv(np.dot(XI.T, XI))
    temp = np.dot( inve , XI.T)
    e2 = np.dot(np.dot(xi.T, temp), MatI) 
    eta = (e1 - e2).T

    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)[0][0]
    etaT_Y = np.dot((eta.T), y)[0][0]

    # construct A, b
    A = None
    b = None

    for loop in range(k): 
        maybe_inliers = maybe_inliers_set[loop]
        inliers = inliers_set[loop]

        MatL = np.zeros((n, n))
        for i in maybe_inliers :
            MatL[i][i] = 1

        XL = np.dot(MatL, X)

        for i in range(n):
            vec_i = np.zeros((n,1))
            vec_i[i][0] = 1
            x_i = X[i].reshape((p,1))
            temp = np.dot( inv( np.dot(XL.T, XL) ), XL.T)
            a = vec_i.T - np.dot(np.dot(x_i.T, temp), MatL)
            a = np.array(a)
            s = np.sign(np.dot(a, y))

            var1 = s[0][0]*a
            var2 = t
            if i not in inliers:
                var1 = -1 * var1
                var2 = -1 * var2
            
            if A is None:
                A = var1
                b = var2
            else:
                A = np.vstack((A, var1))
                b = np.vstack((b, var2))

            A = np.vstack((A, -s[0][0] * a))
            b = np.vstack((b,[0]))
          
    for inliers in accept_inliers_set:
        MatIloop = np.zeros((n,n))
        for i in inliers:
            MatIloop[i][i] = 1

        X_Iloop = np.dot(MatIloop, X)

        for i in inliers:
            vec_i = np.zeros((n,1))
            vec_i[i][0] = 1
            x_i = X[i].reshape((p,1))
            a = vec_i.T - np.dot(x_i,np.dot(np.dot(inv(np.dot(X_Iloop.T, X_Iloop)), X_Iloop.T), MatIloop))
            s = np.dot(a,y)[0][0]
            A = np.vstack((A, -s*a))
            b = np.vstack((b, [0]))
    
    # construct for best inliers
    BI_size = len(best_inliers)
    a = MatI - np.dot(XI, np.dot(np.dot(inv(np.dot(XI.T, XI)), XI.T), MatI))
    sa = np.sign(np.dot(a, y)) / BI_size

    for inliers in accept_inliers_set:
        MatIloop = np.zeros((n,n))
        for i in inliers:
            MatIloop[i][i] = 1
        X_Iloop = np.dot(MatIloop, X)
        I_size = len(inliers)
        a2 = MatIloop - np.dot(X_Iloop, np.dot(np.dot(inv(np.dot(X_Iloop.T, X_Iloop)), X_Iloop.T), MatIloop))
        sa2 = np.sign(np.dot(a2, y)) / I_size
        e = np.dot(sa.T,a) - np.dot(sa2.T,a2)
        A = np.vstack((A, e))
        b = np.vstack((b, [0]))

    A = np.array(A)
    b = np.array(b)
    c = np.dot(Sigma, eta) / etaT_Sigma_eta
    z = np.dot((np.identity(n) - np.dot(c,eta.T)), y)

    Vminus,Vplus = V(n, A, b, c, z)

    numerator = mp.ncdf(etaT_Y / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    denominator = mp.ncdf(Vplus / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    #print("deno = ", denominator)
    
    cdf = float(numerator / denominator)

    p_value = 2*min(cdf,1-cdf)
    #print("pvalue = ", p_value)
    return p_value

def run(trial):
    np.random.seed(trial)
    print(f'trial: {trial}')
    regressor = RANSAC(model = LinearRegressionModel(), t = threshold, 
        k = k, loss = absolute_error_loss, metric = mean_absolute_error)
    #if (i + 1) % 50 == 0:
    X = genX(sample_size, num_predictor, outcome_dim)
    y = geny(X, inliers_size, outliers_size, num_predictor, outcome_dim)

    X = np.array(X).reshape((sample_size, num_predictor))
    y = np.array(y).reshape((sample_size, outcome_dim))

    regressor.fit(X, y)
            
    outliers = regressor.best_outliers    

    #print(maybe_inliers_set)
    #print(inliers_set)

    rand_value = np.random.randint(len(outliers))
    j_selected = outliers[rand_value]
    return check(regressor, num_predictor, outcome_dim, sample_size, j_selected, threshold, X, y)

if __name__ == "__main__":
    inliers_size = 200
    outliers_size = 0
    sample_size = inliers_size + outliers_size
    num_predictor = 1
    outcome_dim = 1
    k = 1
    threshold = np.array([1.8])
    alpha = 0.05
    trials = 50
    print(f"Core available: {multiprocessing.cpu_count()}")
    p_values = []
    for trial in range(trials):
        p_values.append(run(trial))

    reject_count = 0
    
    for p_value in p_values:
        if p_value < alpha:
            reject_count += 1
    # Print FPR and plot the p_value distribution
    print("Percentage of the rejection: ",reject_count / len(p_values))
    plt.hist(p_values)
    plt.savefig("./image/kloop")
    

from copy import copy
import multiprocessing
import concurrent.futures
import multiprocessing.pool
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
        return np.dot(X, self.beta)
    
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
            #print("Beta_hat = ", maybe_model.beta)
            inlier = []
            outlier = []
            # Classify data points as inlier or outlier
            for i in range(X.shape[0]):
                #print(f"Loss {i}: {self.loss(y[i], maybe_model.predict(X[i]))}")
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
                #print(better_model.beta)

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
    
def square_error_loss(y_true, y_pred):
    return (y_true - y_pred)**2

def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]

# Generate data

""" p: num_predictor
    q: outcome_dim
"""
def genX(sample_size, p):
    X = []
    for i in range(sample_size):
        Xi = []
        for j in range(p):
            Xi.append(4 * (np.random.rand() - 0.5) )  # Generate X in [-2, 2]
        X.append(Xi)
    
    return X

def geny(X, sample_size , p):
    y = []
    Beta = [np.random.randint(5) + 1 for _ in range(p)]
    #print("Beta = ", Beta)
    for i in range(sample_size):
        mu = 0
        for j in range(p):
            mu += Beta[j] * X[i][j]
        y.append(mu + np.random.normal(0, 1, 1)[0])

    return y

# Selective p-value

def check(model, p, n, index, t, X, y):
    best_inliers = model.best_inliers
    maybe_inliers_set = model.maybe_inliers_set
    inliers_set = model.inliers_set
    accept_inliers_set = model.accept_inliers_set
    #print("best error = ", model.best_error)
    

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

    c = np.dot(Sigma, eta) / etaT_Sigma_eta
    z = np.dot((np.identity(n) - np.dot(c,eta.T)), y)

    Segments = []
    Segments_count = 0


    for loop in range(k): 
        maybe_inliers = maybe_inliers_set[loop]
        inliers = inliers_set[loop]
        MatL = np.zeros((n,n))

        for i in maybe_inliers:
            MatL[i][i] = 1

        XL = np.dot(MatL, X)

        for i in range(n):
            ei = np.zeros((n,1))
            ei[i][0] = 1
            Xi = X[i].reshape((1,p))
            elementA = np.dot(np.dot(Xi, inv(np.dot(XL.T, XL))), np.dot(XL.T, MatL))
            A = np.dot(ei, ei.T) - 2*(np.dot(ei,elementA)) + np.dot(elementA.T, elementA)

            Q1 = np.dot(np.dot(c.T, A), c)[0][0]
            Q2 = np.dot(np.dot(z.T, (A + A.T)), c)[0][0]
            Q3 = np.dot(np.dot(z.T, A),z)[0][0] - t[0]
            if i not in inliers:
                Q1 = -1*Q1
                Q2 = -1*Q2
                Q3 = -1*Q3
            
            # Trường hợp Q1 = 0
            if Q1 == 0:
                left = None
                right = None
                # Trường hợp Q1, Q2 = 0 
                if Q2 == 0:
                    if Q3 > 0:
                        print("Error1")
                # Trường hợp Q2 != 0
                else: 
                    #Trường hợp Q2 > 0
                    if Q2 > 0:
                        # Đoạn [-oo, -Q3/Q2]
                        left = np.NINF
                        right = -Q3/Q2
                    # Trường hợp Q2 < 0
                    else:
                        # Đoạn [-Q3/Q2, oo]
                        left = -Q3/Q2
                        right = np.Inf
                    Segments.append((left, 1))
                    Segments.append((right, -1))
                    Segments_count += 1
            # Trường hợp Q1 != 0
            else:
                Delta = Q2**2 - 4*Q1*Q3
                # Trường hợp Delta <= 0
                if Delta <= 0:
                    if Q1 > 0:
                        print("Error2")
                # Trường hợp Delta > 0
                else:
                    sol1 = (-Q2 + np.sqrt(Delta))/(2*Q1)
                    sol2 = (-Q2 - np.sqrt(Delta))/(2*Q1)
                    # Đảm báo sol1 < sol2
                    if sol1 > sol2:
                        sol1, sol2 = sol2, sol1
                    # Trường hợp Q1 > 0
                    if Q1 > 0:
                        # Đoạn [sol1, sol2]
                        Segments.append((sol1, 1))
                        Segments.append((sol2, -1))
                    else:
                        # Đoạn [-oo, sol1]
                        Segments.append((np.NINF, 1))
                        Segments.append((sol1, -1))
                        # Đoạn [sol2, oo]
                        Segments.append((sol2, 1))
                        Segments.append((np.Inf, -1))
                    Segments_count += 1

    MatBI = np.zeros((n, n))
    for i in best_inliers:
        MatBI[i][i] = 1
    X_BI = np.dot(MatBI, X)
    element1 = np.dot(np.dot(X_BI,inv(np.dot(X_BI.T, X_BI))), np.dot(X_BI.T, MatBI))
    A1 = np.dot(MatBI.T, MatBI) - np.dot(MatBI.T, element1) - np.dot(element1.T, MatBI) + np.dot(element1.T, element1)
    A1 = A1/len(best_inliers)

    temp = np.dot(MatBI, y)
    temp2 = np.dot(element1, y)
    temp3 = temp - temp2
    #best_error = np.dot(temp3.T, temp3)/len(best_inliers)
    #print("best error' = ", best_error)

    for inliers in accept_inliers_set:
        MatI = np.zeros((n,n))
        for i in inliers:
            MatI[i][i] = 1
        X_I = np.dot(MatI, X)
        element2 = np.dot(np.dot(X_I,inv(np.dot(X_I.T, X_I))), np.dot(X_I.T, MatI))
        A2 = np.dot(MatI.T, MatI) - np.dot(MatI.T, element2) - np.dot(element2.T, MatI)  + np.dot(element2.T, element2)
        A2 = A2/len(inliers)

        A = A1 - A2
        Q1 = np.dot(np.dot(c.T, A), c)[0][0]
        Q2 = np.dot(np.dot(z.T, (A + A.T)), c)[0][0]
        Q3 = np.dot(np.dot(z.T, A),z)[0][0]
            
        # Trường hợp Q1 = 0
        if Q1 == 0:
            left = None
            right = None
            # Trường hợp Q1, Q2 = 0 
            if Q2 == 0:
                if Q3 > 0:
                    print("Error3")
            # Trường hợp Q2 != 0
            else: 
                #Trường hợp Q2 > 0
                if Q2 > 0:
                    # Đoạn [-oo, -Q3/Q2]
                    left = np.NINF
                    right = -Q3/Q2
                # Trường hợp Q2 < 0
                else:
                    # Đoạn [-Q3/Q2, oo]
                    left = -Q3/Q2
                    right = np.Inf
                Segments.append((left, 1))
                Segments.append((right, -1))
                Segments_count += 1
        # Trường hợp Q1 != 0
        else:
            Delta = Q2**2 - 4*Q1*Q3
            # Trường hợp Delta <= 0
            if Delta <= 0:
                if Q1 > 0:
                    print("Error4")
                    #print(f'Q1 = {Q1}, Q2 = {Q2}, Q3 = {Q3}')
                    #print(np.dot(np.dot(y.T, A1), y))
                    #print(np.dot(np.dot(y.T, A2), y))
                    betabest = np.dot(inv(np.dot(X_BI.T, X_BI)), np.dot(X_BI.T,np.dot(MatBI,y)))
                    betai = np.dot(inv(np.dot(X_I.T, X_I)),np.dot(X_I.T,np.dot(MatI,y)))
                    #print(f'betabest = {betabest}')
                    #print(f'betai = {betai} ')

            # Trường hợp Delta > 0
            else:
                sol1 = (-Q2 + np.sqrt(Delta))/(2*Q1)
                sol2 = (-Q2 - np.sqrt(Delta))/(2*Q1)
                # Đảm báo sol1 < sol2
                if sol1 > sol2:
                    sol1, sol2 = sol2, sol1
                # Trường hợp Q1 > 0
                if Q1 > 0:
                    # Đoạn [sol1, sol2]
                    Segments.append((sol1, 1))
                    Segments.append((sol2, -1))
                else:
                    # Đoạn [-oo, sol1]
                    Segments.append((np.NINF, 1))
                    Segments.append((sol1, -1))
                    # Đoạn [sol2, oo]
                    Segments.append((sol2, 1))
                    Segments.append((np.Inf, -1))
                Segments_count += 1

    Segments.sort(reverse=False)
    #print("Segment_count = ", Segments_count)
    #print(Segments)
    
    Regions = []
    cur = 0
    Left = None
    Right = None
    for (point, state) in Segments:
        cur += state
        #print("cur = ", cur)
        #print("point = ", point)
        if cur == Segments_count:
            Left = point
        if cur < Segments_count and Left is not None:
            Right = point
            Regions.append((Left, Right))
            Left = None
            Right = None

    #print(Regions)
    numerator = 0
    denominator = 0
    mu = 0
    tn_sigma = np.sqrt(etaT_Sigma_eta)
    for (left, right) in Regions:
        
        denominator = denominator + mp.ncdf((right - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)

        if etaT_Y >= right:
            numerator = numerator + mp.ncdf((right - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
        elif (etaT_Y >= left) and (etaT_Y < right):
            numerator = numerator + mp.ncdf((etaT_Y - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
        
    if denominator == 0:
        return None
    else:
        cdf = float(numerator/denominator)
        pvalue = 2*min(cdf, 1 - cdf)
        return pvalue
        

sample_size = 100
num_predictor = 10
k = 10
threshold = np.array([1.8])
alpha = 0.05
trials = 6000
def run(trial):
    np.random.seed(trial)
    print(f'trial: {trial}')
    regressor = RANSAC(model = LinearRegressionModel(), t = threshold, 
        k = k, loss = square_error_loss, metric = mean_square_error)
    #if (i + 1) % 50 == 0:
    X = genX(sample_size, num_predictor)
    y = geny(X, sample_size, num_predictor)

    X = np.array(X).reshape((sample_size, num_predictor))
    y = np.array(y).reshape((sample_size, 1))

    regressor.fit(X, y)
    
    outliers = regressor.best_outliers   
    #print(maybe_inliers_set)
    #print(inliers_set)

    rand_value = np.random.randint(len(outliers))
    j_selected = outliers[rand_value]
    return check(regressor, num_predictor, sample_size, j_selected, threshold, X, y)

if __name__ == "__main__":
    
    print(f"Core available: {multiprocessing.cpu_count()}")
    
    p_values = []
    
    p = multiprocessing.Pool(4)
    result = p.map_async(run, range(trials), chunksize=int(trials/4))
    p.close()
    p.join()
    
    for p_value in result.get():
        if p_value is not None:
            p_values.append(p_value)
    """
    
    for trial in range(trials):
        p_value = run(trial)
        if p_value != None :
            p_values.append(p_value)
    """
    reject_count = 0
    
    for p_value in p_values:
        if p_value < alpha:
            reject_count += 1
    # Print FPR and plot the p_value distribution
    #print(p_values)
    print("Percentage of the rejection: ",reject_count / len(p_values))
    plt.hist(p_values)
    plt.show()
    

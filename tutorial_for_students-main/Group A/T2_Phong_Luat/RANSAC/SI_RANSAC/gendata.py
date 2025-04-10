import numpy as np
from scipy.stats import skewnorm
from scipy.stats import t
from scipy.stats import laplace

# Generate noise

def gen_noise(n, type = None, sigma = None):
    if sigma is None:
        sigma = np.identity(n)
    mean = np.zeros(n)
    """
    type 1: normal distribution
    type 2: skew normal distribution
    type 3: t distribution
    type 4: Laplace distribution
    """
    if type == 1:
        return np.random.multivariate_normal(mean = mean, cov = sigma)
    if type == 2:
        eps =  skewnorm.rvs(a = 10, loc = 0, scale = 1, size = n)
        mean = skewnorm.mean(a=10, loc=0, scale=1)
        std  = skewnorm.std(a=10, loc=0, scale=1)
        eps = (eps - mean)/std
        return eps
    if type == 3:
        eps = t.rvs(df = 20, loc = 0, scale = 1, size = n)
        mean = t.mean(df=20, loc=0, scale=1)
        std  = t.std(df=20, loc=0, scale=1)
        eps = (eps - mean)/std
        return eps
    if type == 4:
        eps = laplace.rvs(loc=0, scale=1, size=n)
        mean = laplace.mean(loc=0, scale=1)
        std = laplace.std(loc = 0, scale=1)
        eps = (eps - mean)/std
        return eps
    

# Generate data
    
def gen(n, p, num_outliers, delta, sigma = None, type = None):
    X = []
    for i in range(n):
        Xi = []
        for j in range(p):
            Xi.append(4 * (np.random.rand() - 0.5) )  # Generate X in [-2, 2]
        X.append(Xi)

    y = []
    Beta = np.array([2 + i/5 for i in range(p)])
    inliers_size = n - num_outliers
    
    noise = gen_noise(n, type, sigma)

    for i in range(inliers_size):
        mu = np.dot(X[i], Beta)
        y.append(mu)

    for i in range(num_outliers):
        mu = np.dot(X[i + inliers_size], Beta)
        y.append(mu + delta)

    y = y + noise
    y = np.array(y).reshape((n, 1))
    
    X = np.array(X)
    return X, y

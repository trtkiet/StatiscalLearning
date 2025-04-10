import numpy as np
from numpy.linalg import inv
from Regressor import RANSAC
from Regressor import LinearRegressionModel
from gendata import gen
from solve_equation import solveEquation
from interval_operation import getIntersection
from interval_operation import getUnion
from interval_operation import getComplement
from calculate_selective_pvalue import truncated_p_value

def prebuild(model, X, a, b):
    maybe_inliers_set = model.maybe_inliers_set
    n, p = X.shape[0], X.shape[1]
    k = model.k
    t = model.t
    L = []
    M = []
    for i in range(k):
        L.append([])
        M.append([])
        maybe_inliers = maybe_inliers_set[i]
        MatL = np.zeros((n,n))

        for j in maybe_inliers:
            MatL[j][j] = 1
        XL = np.dot(MatL, X)
        for j in range(n):
            L[i].append([])
            M[i].append([])
            # If j'th instance is inlier
            ei = np.zeros((n,1))
            ei[j][0] = 1
            Xi = X[j].reshape((1,p))
            elementA = np.dot(np.dot(Xi, inv(np.dot(XL.T, XL))), np.dot(XL.T, MatL))
            A = np.dot(ei, ei.T) - 2*(np.dot(ei,elementA)) + np.dot(elementA.T, elementA)

            Q1 = np.dot(np.dot(b.T, A), b)[0][0]
            Q2 = np.dot(np.dot(a.T, (A + A.T)), b)[0][0]
            Q3 = np.dot(np.dot(a.T, A),a)[0][0] - t[0]
            L[i][j].append(solveEquation(Q1, Q2, Q3))
            L[i][j].append(getComplement(L[i][j][0]))

    return L

# Prepare for compute
def prepare(model, X, y, true_Sigma, j_selected, isEstimated):
    outliers = model.best_outliers
    n = X.shape[0]
    p = X.shape[1]
    Sigma = true_Sigma
    variance = 0
    for i in range(n):
        if i not in outliers:
            variance = max(variance, (y[i] - model.predict(X[i]))**2)
    if isEstimated == True:
        Sigma *= variance
    # construct eta
    vec_index = np.zeros((n, 1))
    vec_index[j_selected][0] = 1
    xi = X[j_selected].reshape((p, 1))

    MatI = np.zeros((n, n))
    for i in range(n):
        if i not in outliers:
            MatI[i][i] = 1

    XI = np.dot(MatI, X)
    e1 = vec_index.T
    inve = inv(np.dot(XI.T, XI))
    temp = np.dot( inve , XI.T)
    e2 = np.dot(np.dot(xi.T, temp), MatI)
    eta = (e1 - e2).T

    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)[0][0]
    etaT_y = np.dot((eta.T), y)[0][0]

    b = np.dot(Sigma, eta) / etaT_Sigma_eta
    a = np.dot((np.identity(n) - np.dot(b, eta.T)), y)
    return etaT_y, etaT_Sigma_eta, a, b

# Compare two lists, 1 if equivalent, otherwise 0
def compare_list(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    if len(l1) != len(l2):
        return False
    return (l1 == l2).all()
    
def conditionalRegion(model, X, a, b):
    n = X.shape[0]
    k = model.k
    outliers = model.best_outliers

    L = prebuild(model, X, a, b)

    dp = [[[[] for i in range(n+1)] for i in range(n+1)] for i in range(k+1)]

    for i in range(1, k + 1):
        dp[i][0][0].append((np.NINF, np.Inf))

    for i in range(1, k + 1):
        for j in range(1, n + 1):
            for o in range(0, j + 1):
                res1 = []
                res2 = []
                if j-1 < o:
                    res1 = L[i-1][j-1][0]
                else:
                    res1 = getIntersection(dp[i][j-1][o], L[i-1][j-1][0])
                if o != 0:
                    res2 = getIntersection(dp[i][j-1][o-1], L[i-1][j-1][1])
            
                dp[i][j][o] = getUnion(res1, res2)

    RegionSet = []
    for i in range(1, k + 1):
        ans = [(np.NINF, np.Inf)]
        for j in range(1, n + 1):
            state = 0
            if j-1 in outliers:
                state = 1
            ans = getIntersection(ans, L[i-1][j-1][state])
      
        for ii in range(1, i):
            ans = getIntersection(ans, getComplement(dp[ii][n][len(outliers)]))
        for ii in range(i+1, k+1):
            ans = getIntersection(ans, getComplement(dp[ii][n][len(outliers)-1]))

        RegionSet = getUnion(RegionSet, ans)
    
    return RegionSet

# Proposed method
def run(n, p, m, t, k, d, num_outliers, delta, distribution_type, isCorrelated, isEstimated, phi):
    
    true_Sigma = None
    
    if isCorrelated == True:
        true_Sigma = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                true_Sigma[i][j] = phi**(abs(i - j))
    else:
        true_Sigma = np.identity(n)
    
    X, y = gen(n = n, p = p, num_outliers = num_outliers, delta = delta, type = distribution_type)

    regressor = RANSAC(model = LinearRegressionModel(), m = m, t = t, k = k, d = d)
    regressor.fit(X, y)
    outliers = regressor.best_outliers

    if len(outliers) == 0:
        return None
    rand_value = np.random.randint(len(outliers))
    j_selected = outliers[rand_value]
    if num_outliers != 0:
        valid_to_check = []
        for i in range(n - num_outliers, n):
            if i in outliers:
                valid_to_check.append(i)
        rand_value = np.random.randint(len(valid_to_check))
        j_selected = valid_to_check[rand_value]
    
    etaT_y, etaT_Sigma_eta, a, b = prepare(regressor, X, y, true_Sigma, j_selected, isEstimated)
    RegionSet = conditionalRegion(regressor, X, a, b)

    p_value = truncated_p_value(RegionSet, etaT_y, etaT_Sigma_eta)
    return p_value

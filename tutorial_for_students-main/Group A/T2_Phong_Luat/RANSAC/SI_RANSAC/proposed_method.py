import numpy as np
import heapq
from gendata import gen
from Regressor import RANSAC
from Regressor import LinearRegressionModel
from calculate_OC_region import calc_OC_regions
from calculate_selective_pvalue import truncated_p_value
from solve_equation import solveEquation
from numpy.linalg import pinv

def prebuild(model, inequalities_sol_for_detecting_stage, X, a, b):
    maybe_inliers_set = model.maybe_inliers_set
    n, p = X.shape[0], X.shape[1]
    k = model.k
    t = model.t
    
    for i in range(k):
        inequalities_sol_for_detecting_stage.append([])
        maybe_inliers = maybe_inliers_set[i]
        MatL = np.zeros((n,n))

        for j in maybe_inliers:
            MatL[j][j] = 1
        XL = np.dot(MatL, X)
        for j in range(n):
            inequalities_sol_for_detecting_stage[i].append([])
            # If j'th instance is inlier
            ei = np.zeros((n,1))
            ei[j][0] = 1
            Xi = X[j].reshape((1,p))
            elementA = np.dot(np.dot(Xi, pinv(np.dot(XL.T, XL))), np.dot(XL.T, MatL))
            A = np.dot(ei, ei.T) - 2*(np.dot(ei,elementA)) + np.dot(elementA.T, elementA)

            Q1 = np.dot(np.dot(b.T, A), b)[0][0]
            Q2 = np.dot(np.dot(a.T, (A + A.T)), b)[0][0]
            Q3 = np.dot(np.dot(a.T, A),a)[0][0] - t[0]
            Intervals1 = []
            solveEquation(Intervals1, Q1, Q2, Q3)
            inequalities_sol_for_detecting_stage[i][j].append(Intervals1)
            # If j'th instance is outlier
            Intervals2 = []
            solveEquation(Intervals2, -Q1, -Q2, -Q3)
            inequalities_sol_for_detecting_stage[i][j].append(Intervals2)

# Prepare for compute
def prepare(model, X, y, true_Sigma, j_selected, isEstimated):
    outliers = model.best_outliers
    best_inliers = model.best_inliers
    n = X.shape[0]
    p = X.shape[1]
    SSR = 0
    Sigma = true_Sigma
    for i in range (n):
        if i not in outliers:
            SSR += (y[i] - model.predict(X[i]))**2
    variance = SSR/(n - len(outliers) - 1)
    if isEstimated == True:
        Sigma *= variance
    # construct eta
    vec_index = np.zeros((n, 1))
    vec_index[j_selected][0] = 1
    xi = X[j_selected].reshape((p, 1))

    MatI = np.zeros((n, n))
    for i in best_inliers:
        MatI[i][i] = 1

    XI = np.dot(MatI, X)
    e1 = vec_index.T
    inve = pinv(np.dot(XI.T, XI))
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
    if len(l1) != len(l2):
        return False
    return (l1 == l2).all()
    
def conditionalRegion(model, X, a, b, limit, metric):
    n = X.shape[0]
    p = X.shape[1]
    m = model.m
    t = model.t
    k = model.k
    d = model.d
    maybe_inliers = model.maybe_inliers_set
    outliers = model.best_outliers

    inequalities_sol_for_detecting_stage = []
    prebuild(model, inequalities_sol_for_detecting_stage, X, a, b)
    z_min = -limit
    z_max = limit
    z_cur = z_min
    a_small_gap = 0.001

    Region = calc_OC_regions(model = model, n = n, p = p, X = X, a = a, b = b, 
                             inequalities_sol_for_detecting_stage = inequalities_sol_for_detecting_stage)
    RegionSet = []
    CheckedRegion = []
    heapq.heapify(CheckedRegion)
    for i in Region:
        RegionSet.append(i)
        heapq.heappush(CheckedRegion, i)

    while len(CheckedRegion) != 0:
        i = heapq.nsmallest(1, CheckedRegion)
        left, right = i[0][0], i[0][1]
        if right < z_cur:
            heapq.heappop(CheckedRegion)
            continue
        if z_cur >= left and z_cur <= right:
            z_cur = right + a_small_gap
            heapq.heappop(CheckedRegion)
            break
        break
    
        
    while z_cur <= z_max:
        y_z = a + b*z_cur
        sub_model = RANSAC(model = LinearRegressionModel(), m = m, t = t,
        k = k, d = d, seed = maybe_inliers, metric = metric)
        sub_model.fit(X, y_z)
        Oy_z = sub_model.best_outliers
        Region = calc_OC_regions(model = sub_model, n = n, p = p, X = X, a = a, b = b,
                                inequalities_sol_for_detecting_stage = inequalities_sol_for_detecting_stage)
        # find the outlier set in the new potential region
        for i in Region:
            heapq.heappush(CheckedRegion, i)
        
        while len(CheckedRegion) != 0:
            two_region = heapq.nsmallest(2, CheckedRegion)
            right1 = two_region[0][1]
            if  right1 <= z_min:
                heapq.heappop(CheckedRegion)
                continue
            if right1 + a_small_gap >= z_max:
                z_cur = right1 + a_small_gap
                break
            left2 = z_max
            if len(CheckedRegion) >= 2:
                left2 = two_region[1][0]
            if right1 + a_small_gap >= left2:
                heapq.heappop(CheckedRegion)
                continue
            z_cur = right1 + a_small_gap
            break
        if compare_list(outliers, Oy_z) == True:
            for i in Region:
                RegionSet.append(i)
    return RegionSet

# Proposed method
def run(n, p, m, t, k, d, num_outliers, delta, distribution_type, isCorrelated, isEstimated, phi, metric):
    
    true_Sigma = None
    
    if isCorrelated == True:
        true_Sigma = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                true_Sigma[i][j] = phi**(abs(i - j))
    else:
        true_Sigma = np.identity(n)
    
    X, y = gen(n = n, p = p, num_outliers = num_outliers, delta = delta, type = distribution_type)

    regressor = RANSAC(model = LinearRegressionModel(), m = m, t = t,
        k = k, d = d, metric = metric)
    regressor.fit(X, y)
    outliers = regressor.best_outliers

    if len(outliers) == 0:
        return None
    rand_value = np.random.randint(len(outliers))
    j_selected = outliers[rand_value]
    if num_outliers != 0:
        if j_selected < n - num_outliers:
            return None
    
    etaT_y, etaT_Sigma_eta, a, b = prepare(regressor, X, y, true_Sigma, j_selected, isEstimated)
    limit = 20 * np.sqrt(etaT_Sigma_eta)
    RegionSet = conditionalRegion(regressor, X, a, b, limit, metric)
    p_value = truncated_p_value(RegionSet, etaT_y, etaT_Sigma_eta)
    return p_value

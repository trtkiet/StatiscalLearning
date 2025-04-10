import numpy as np
from numpy.linalg import pinv
from solve_equation import solveEquation

# Calculate OC region
def calc_OC_regions(model, n, p, X, a, b, inequalities_sol_for_detecting_stage):
    best_inliers = model.best_inliers
    inliers_set = model.inliers_set
    accept_inliers_set = model.accept_inliers_set
    Segments = []
    Segments_count = 0
    k = model.k
    for loop in range(k):
        inliers = inliers_set[loop]

        for i in range(n):
            Region = []
            if i in inliers:
                Region = inequalities_sol_for_detecting_stage[loop][i][0]
            else:
                Region = inequalities_sol_for_detecting_stage[loop][i][1]
            if len(Region) == 0:
                continue
            Segments_count += 1
            for region in Region:
                Segments.append(region)

    if len(best_inliers) != 0: 
        MatBI = np.zeros((n, n))
        for i in best_inliers:
            MatBI[i][i] = 1
        X_BI = np.dot(MatBI, X)
        element1 = np.dot(np.dot(X_BI,pinv(np.dot(X_BI.T, X_BI))), np.dot(X_BI.T, MatBI))
        A1 = np.dot(MatBI.T, MatBI) - np.dot(MatBI.T, element1) - np.dot(element1.T, MatBI) + np.dot(element1.T, element1)
        A1 = A1/len(best_inliers)

        for inliers in accept_inliers_set:
            MatI = np.zeros((n,n))
            for i in inliers:
                MatI[i][i] = 1
            X_I = np.dot(MatI, X)
            element2 = np.dot(np.dot(X_I, pinv(np.dot(X_I.T, X_I))), np.dot(X_I.T, MatI))
            A2 = np.dot(MatI.T, MatI) - np.dot(MatI.T, element2) - np.dot(element2.T, MatI)  + np.dot(element2.T, element2)
            A2 = A2/len(inliers)

            A = A1 - A2
            Q1 = np.dot(np.dot(b.T, A), b)[0][0]
            Q2 = np.dot(np.dot(a.T, (A + A.T)), b)[0][0]
            Q3 = np.dot(np.dot(a.T, A), a)[0][0]
            RegionMSE = []
            solveEquation(RegionMSE, Q1, Q2, Q3)
            if len(RegionMSE) != 0:
                Segments_count += 1
                for regionMSE in RegionMSE:
                    Segments.append(regionMSE)
            
    Segments.sort(reverse=False)
    Regions = []
    cur = 0
    Left = None
    Right = None
    for (point, state) in Segments:
        cur += state
        if cur == Segments_count:
            Left = point
        if cur < Segments_count and Left is not None:
            Right = point
            Regions.append((Left, Right))
            Left = None
            Right = None

    return Regions

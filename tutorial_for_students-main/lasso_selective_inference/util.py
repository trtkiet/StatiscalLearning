import numpy as np
from mpmath import mp

mp.dps = 500


def construct_s(bh):
    s = []

    for bhj in bh:
        if bhj != 0:
            s.append(np.sign(bhj))

    s = np.array(s)
    s = s.reshape((len(s), 1))
    return s


def check_KKT(XA, XAc, y, bhA, lamda, n):

    print("\nCheck Active")
    e1 = y - np.dot(XA, bhA)
    e2 = np.dot(XA.T, e1)
    print(e2 / lamda)

    if XAc is not None:
        print("\nCheck In Active")
        e1 = y - np.dot(XA, bhA)
        e2 = np.dot(XAc.T, e1)
        print(e2 / lamda)


def construct_A_XA_Ac_XAc_bhA(X, bh, p):
    A = []
    Ac = []
    bhA = []

    for j in range(p):
        bhj = bh[j]
        if bhj != 0:
            A.append(j)
            bhA.append(bhj)
        else:
            Ac.append(j)

    XA = X[:, A]
    XAc = X[:, Ac]
    bhA = np.array(bhA).reshape((len(A), 1))

    return A, XA, Ac, XAc, bhA


def construct_test_statistic(j, XA, y, A):
    ej = []
    for each_j in A:
        if j == each_j:
            ej.append(1)
        else:
            ej.append(0)

    ej = np.array(ej).reshape((len(A), 1))

    inv = np.linalg.pinv(np.dot(XA.T, XA))
    XAinv = np.dot(XA, inv)
    etaj = np.dot(XAinv, ej)

    etajTy = np.dot(etaj.T, y)[0][0]

    return etaj, etajTy


def pivot_with_specified_interval(z_interval, etaj, etajTy, cov, tn_mu):

    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etajTy >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etajTy >= al) and (etajTy < ar):
            numerator = numerator + mp.ncdf((etajTy - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None

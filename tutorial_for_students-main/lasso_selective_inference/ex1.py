import numpy as np
from sklearn import linear_model
from mpmath import mp
mp.dps = 500

import matplotlib.pyplot as plt
import statsmodels.api as sm

import gen_data
import util


def run():
    n = 100
    p = 5
    lamda = 5
    beta_vec = [0, 0, 0, 0, 0]
    cov = np.identity(n)

    X, y, true_y = gen_data.generate(n, p, beta_vec)

    Lasso = linear_model.Lasso(alpha=lamda/n, fit_intercept=False, tol=1e-10)
    Lasso.fit(X, y)

    bh = Lasso.coef_

    y = y.reshape((n, 1))
    true_y = true_y.reshape((n, 1))

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, p)

    s = util.construct_s(bh)

    # You have to always check the KKT condition to confirm the correctness
    # print("=== Check KKT ===")
    # util.check_KKT(XA, XAc, y, bhA, lamda, n)
    # print("=== Check KKT ===")

    if len(A) == 0:
        return None

    rand_value = np.random.randint(len(A))
    j_selected = A[rand_value]

    etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

    XM = XA
    XMc = XAc

    inv = np.linalg.pinv(np.dot(XM.T, XM))
    e1 = np.dot(XM, inv)
    PM = np.dot(e1, XM.T)

    XMTplus = np.dot(XM, inv)

    lens = s.shape[0]
    diag = np.zeros((lens, lens))
    for i in range(lens):
        diag[i][i] = s[i][0]

    A01 = np.dot(XMc.T, (np.identity(n) - PM))
    A02 = - np.dot(XMc.T, (np.identity(n) - PM))
    A0 = (np.vstack((A01, A02))) / lamda

    lenAc = len(Ac)
    b01 = (np.ones(lenAc)).reshape((lenAc, 1)) - np.dot(XMc.T, np.dot(XMTplus, s))
    b02 = (np.ones(lenAc)).reshape((lenAc, 1)) + np.dot(XMc.T, np.dot(XMTplus, s))
    b0 = np.vstack((b01, b02))

    A1 = - np.dot(diag, np.dot(inv, XM.T))
    b1 = - lamda * np.dot(diag, np.dot(inv, s))

    A_matrix = np.vstack((A0, A1))
    b = np.vstack((b0, b1))

    c1 = np.dot(cov, etaj)
    c2 = np.linalg.pinv(np.dot(etaj.T, np.dot(cov, etaj)))
    c = np.dot(c1, c2)

    z = np.dot((np.identity(n) - np.dot(c, etaj.T)), y)

    Vminus = np.NINF
    Vplus = np.Inf

    Adotz = np.dot(A_matrix, z)
    Adotc = np.dot(A_matrix, c)

    for j in range(b.shape[0]):

        if - 1e-10 <= Adotc[j][0] <= 1e-10:
            Adotc[j][0] = 0

        if Adotc[j][0] < 0:
            temp = (b[j][0] - Adotz[j][0]) / Adotc[j][0]
            Vminus = max(Vminus, temp)
        elif Adotc[j][0] > 0:
            temp = (b[j][0] - Adotz[j][0]) / Adotc[j][0]
            Vplus = min(Vplus, temp)

    if Vminus >= Vplus:
        print('Error')

    tn_mu = np.dot(etaj.T, true_y)[0][0]
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    numerator = mp.ncdf((etajTy - tn_mu) / tn_sigma) - mp.ncdf((Vminus - tn_mu) / tn_sigma)
    denominator = mp.ncdf((Vplus - tn_mu) / tn_sigma) - mp.ncdf((Vminus - tn_mu) / tn_sigma)

    cdf = float(numerator/denominator)

    # compute two-sided selective p_value
    selective_p_value = 2 * min(cdf, 1 - cdf)

    return selective_p_value


if __name__=="__main__":
    run()

    max_iteration = 1200
    list_p_value = []

    for each_iter in range(max_iteration):
        print(each_iter)
        p_value = run()
        if p_value is not None:
            list_p_value.append(p_value)

    plt.rcParams.update({'font.size': 18})
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, sm.distributions.ECDF(np.array(list_p_value))(grid), 'r-', linewidth=6, label='p-value')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.hist(list_p_value)
    plt.show()

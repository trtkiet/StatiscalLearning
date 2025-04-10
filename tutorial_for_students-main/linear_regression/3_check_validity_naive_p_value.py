import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

import matplotlib.pyplot as plt
import statsmodels.api as sm

import gen_data

def run():
    true_beta = [0]
    X, y_obs = gen_data.generate(20, 1, true_beta)

    # Estimate beta
    XTX = np.dot(X.T, X)
    XTXinv = np.linalg.inv(XTX)
    XTXinvXT = np.dot(XTXinv, X.T)
    beta = np.dot(XTXinvXT, y_obs)

    # beta is considered as the test-statistic and can be decomposed in the form of eta^T y
    eta = XTXinvXT.T

    # Observed test-statistic
    etaTy_obs = np.dot(eta.T, y_obs)[0][0]  # This should be equal to beta

    # Compute two-sided naive-p value
    cdf = norm.cdf(etaTy_obs, loc=0, scale=np.sqrt(np.dot(eta.T, eta)[0][0]))
    naive_p_value = 2 * min(1 - cdf, cdf)

    return naive_p_value


if __name__ == '__main__':

    detect = 0
    reject = 0

    max_iteration = 1200
    list_naive_p_value = []

    for each_iter in range(max_iteration):
        print(each_iter)
        naive_p_value = run()

        list_naive_p_value.append(naive_p_value)

        detect = detect + 1
        if naive_p_value <= 0.05:
            reject = reject + 1

    print('False Positive Rate (FPR):', reject/detect)
    # In the case of simple linear regression, the naive-p value is VALID in the sense that,
    # when we set alpha = 0.05, the FPR is controlled under alpha.
    # However, in the future, I will show you an example when naive-p value is INVALID

    plt.rcParams.update({'font.size': 18})
    grid = np.linspace(0, 1, 101)
    # plt.switch_backend('agg')
    plt.plot(grid, sm.distributions.ECDF(np.array(list_naive_p_value))(grid), 'r-', linewidth=6, label='Pivot')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('z_pivot.png', dpi=100)
    plt.show()







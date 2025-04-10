import numpy as np
import matplotlib.pyplot as plt

from mpmath import mp
mp.dps = 500


def run():
    # generate synthetic data
    n = 3
    mu_vec = np.zeros((n, 1))
    Sigma = np.identity(n)

    # observed data
    x_obs = np.random.multivariate_normal(mu_vec.flatten(), Sigma)
    x_obs = x_obs.reshape((n, 1))

    # select maximum element
    i_max = np.argmax(x_obs)

    # we want to the following hypotheses
    # H_0: mu_{i_max} = 0   vs.   H_1: mu_{i_max} != 0

    # construct eta
    eta = np.zeros((n, 1))
    eta[i_max][0] = 1.0

    # observed value of test-statistic
    etaTx_obs = np.dot(eta.T, x_obs)[0][0]

    # Construct martix A and vector b
    A = None
    b = None

    for i in range(n):
        # x_i <= x_i_max <=> x_i - x_i_max <= 0

        e_i = np.zeros((n, 1))
        e_i[i][0] = 1.0

        e_i_max = np.zeros((n, 1))
        e_i_max[i_max][0] = 1.0

        if A is None:
            A = (e_i - e_i_max).T
            b = [0]
        else:
            A = np.vstack((A, (e_i - e_i_max).T))
            b = np.vstack((b, 0))

    A = np.array(A)
    b = np.array(b)

    # Compute vector c in Equation 5.3 of Lee et al. 2016
    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)[0][0]
    c = np.dot(Sigma, eta) / etaT_Sigma_eta

    # Compute vector z in Equation 5.2 of Lee et al. 2016
    z = np.dot(np.identity(n) - np.dot(c, eta.T), x_obs)

    # Following Lemma 5.1 of Lee et al. 2016 to compute V^{-} and V^{+}
    Az = np.dot(A, z)
    Ac = np.dot(A, c)

    Vminus = np.NINF
    Vplus = np.Inf

    for j in range(len(b)):
        left = np.around(Ac[j][0], 5)
        right = np.around(b[j][0] - Az[j][0], 5)

        if left == 0:
            if right < 0:
                print('Error')
        else:
            temp = right / left

            if left > 0:
                Vplus = min(temp, Vplus)
            else:
                Vminus = max(temp, Vminus)

    # compute cdf of truncated gaussian distribution
    numerator = mp.ncdf(etaTx_obs / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    denominator = mp.ncdf(Vplus / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
    cdf = float(numerator / denominator)

    # compute two-sided selective p_value
    selective_p_value = 2 * min(cdf, 1 - cdf)

    return selective_p_value


if __name__ == '__main__':
    # run()

    max_iteration = 1200
    list_p_value = []

    alpha = 0.05
    count = 0

    for iter in range(max_iteration):
        if iter % 100 == 0:
            print(iter)

        selective_p_value = run()
        list_p_value.append(selective_p_value)

        if selective_p_value <= alpha:
            count = count + 1

    print()
    print('False positive rate:', count / max_iteration)
    plt.hist(list_p_value)
    plt.show()

    # VALID p-value should follow uniform distribution
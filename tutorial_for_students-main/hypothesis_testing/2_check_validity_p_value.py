import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def run():
    # generate synthetic data
    mu = 1
    std = 1  # standard deviation

    x = np.random.normal(loc=mu, scale=1)

    # we want to the following hypotheses
    # H_0: mu = 0   vs.   H_1: mu != 0

    # construct test-statistic
    T = x

    # compute two-sided p-value
    cdf = norm.cdf(T, loc=0, scale=1)
    p_value = 2 * min(cdf, 1 - cdf)

    return p_value


if __name__ == '__main__':

    max_iteration = 1000
    list_p_value = []

    alpha = 0.05
    count = 0

    for _ in range(max_iteration):
        p_value = run()
        list_p_value.append(p_value)

        if p_value <= alpha:
            count = count + 1

    print(count/max_iteration)
    plt.hist(list_p_value)
    plt.show()

    # VALID p-value should follow uniform distribution

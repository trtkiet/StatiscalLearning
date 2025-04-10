import numpy as np
from scipy.stats import norm

def run():
    # generate synthetic data
    mu = 0
    std = 1  # standard deviation
    x = np.random.normal(loc=mu, scale=std)

    # we want to the following hypotheses
    # H_0: mu = 0   vs.   H_1: mu != 0

    # construct test-statistic
    T = x

    # compute two-sided p-value
    cdf = norm.cdf(T, loc=0, scale=1)
    p_value = 2 * min(cdf, 1 - cdf)

    print(p_value)


if __name__ == '__main__':
    run()
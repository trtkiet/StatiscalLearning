import numpy as np
from tqdm import tqdm
import mpmath as mp
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
mp.dps = 500

output_figs_dir = Path("output_figs")
output_figs_dir.mkdir(exist_ok=True, parents=True)

def calulate_p_value():
    n = 3
    mu = np.zeros((n, 1))
    Sigma = np.identity(n)
    x_obs = np.random.multivariate_normal(mu.flatten(), Sigma)
    x_obs = x_obs.reshape((n, 1))
    
    # Selete max of abs(x_obs)
    i_max = np.argmax(np.abs(x_obs))
    
    eta = np.zeros((n, 1))
    eta[i_max][0] = 1
    
    # Observed value of the test statistic
    etaTx_obs  = np.dot(eta.T, x_obs)[0][0] # Scalar
    sign_max = np.sign(x_obs[i_max][0])
    # Compute A and b
    A = None
    b = None
    
    for i in range(n):
        # |x_i_max| >= |x_i|
        eta_i = np.zeros((n, 1))
        eta_i[i][0] = 1
        
        e_i_max = np.zeros((n, 1))
        e_i_max[i_max][0] = 1
        
        sign = np.sign(x_obs[i])
        if A is None:
            A = (sign * eta_i - sign_max*e_i_max).T
            b = [0]
        else:
            A = np.vstack((A, (sign * eta_i - sign_max*e_i_max).T))
            b = np.vstack((b, 0))
    
    A = np.array(A)
    b = np.array(b)
    # Compute vector c 
    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)[0][0]
    c = np.dot(Sigma, eta) / etaT_Sigma_eta
    
    # Compute vector z
    z = np.dot((np.identity(n) - np.dot(c, eta.T)), x_obs)
    
    # Ccompute Az and Ac
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


def run_exp(iter, n=100000):
    count = 0
    p_values = []
    for _ in tqdm(range(n)):
        p_value = calulate_p_value()
        if p_value < 0:
            warnings.warn(f"p_value: {p_value}")
        p_values.append(p_value)
        if p_value < 0.05:
            count += 1
    fig = plt.hist(p_values)
    plt.savefig(output_figs_dir / f"p_values_{iter}.png")
    plt.close()
    return count/n

if __name__ == "__main__":
    count = 0
    num_exp = 5
    count_success = 0
    for i in range(num_exp):
        res = run_exp(iter = i + 1)
        if 0.04 <= res <= 0.06:
            count_success += 1
        print(f"Exp: {i + 1}, p: {res}")
    print(count_success/num_exp)
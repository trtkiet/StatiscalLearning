import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from proposed_method import run

if __name__ == "__main__":
    n = 50
    p = 5   
    m = p + 1
    d = 3/5 * n
    k = 10
    t = np.array([2])
    alpha = 0.05
    alpha2 = 0.1
    trials = 10000
    num_outliers = 0            # for generate data
    delta = 0                   # for generate data
    isEstimated = False
    isCorrelated = False
    phi = 0.5
    distribution_type = 1   # "1. Normal  2. Skew normal  3. Student  4. Laplace"
    
    p_values = []   
    with tqdm(range(trials),unit="trial", mininterval=0, disable=False) as bar:
        bar.set_description("Process")
        for trial in bar: 
            p_value = run(n = n, p = p, m = m, d = d, t = t, k = k, num_outliers = num_outliers, delta = delta,
                    distribution_type = distribution_type, isCorrelated = isCorrelated, isEstimated = isEstimated, phi = phi)
            if p_value != None :
                p_values.append(p_value)
            bar.set_postfix(p_value = p_value)
    
    reject_count = 0
    for i in range(len(p_values)):
        p_value = p_values[i]
        if p_value < alpha:
            reject_count += 1

    print(p_values)
    print("len = ", len(p_values))
    print("Percentage of the rejection with alpha 0.05: ",reject_count / len(p_values))
    #plt.hist(p_values)     
    #plt.show()

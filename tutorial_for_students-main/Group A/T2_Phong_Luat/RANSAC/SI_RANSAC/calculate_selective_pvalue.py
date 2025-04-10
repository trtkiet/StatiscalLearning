from mpmath import mp
import numpy as np

mp.dps = 1000

# Compute selective p-value
def truncated_p_value(Regions, etaT_y, etaT_Sigma_eta):
    numerator = 0
    denominator = 0 
    mu = 0
    tn_sigma = np.sqrt(etaT_Sigma_eta)
    for i in Regions:
        left = i[0]
        right = i[1]
        denominator = denominator + mp.ncdf((right - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
        if etaT_y >= right:
            numerator = numerator + mp.ncdf((right - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
        elif (etaT_y >= left) and (etaT_y < right):
            numerator = numerator + mp.ncdf((etaT_y - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
    
    if denominator == 0:
        print("Error5")
        return None
    else:
        cdf = float(numerator/denominator) 
        pvalue = 2*min(cdf, 1 - cdf)
        return pvalue

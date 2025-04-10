import numpy as np

def solveEquation(Q1, Q2, Q3):
    # Case: Q1 = 0
    if Q1 == 0:
        left = None
        right = None
        # Case: Q1, Q2 = 0 
        if Q2 == 0:
            if Q3 > 0:
                return []
            else:
                return [(np.NINF, np.Inf)]
        # Case: Q2 != 0
        else: 
            # Case: Q2 > 0
            if Q2 > 0:
                #  [-oo, -Q3/Q2]
                left = np.NINF
                right = -Q3/Q2
            # Case: Q2 < 0
            else:
                # [-Q3/Q2, oo]
                left = -Q3/Q2
                right = np.Inf
            return [(left, right)]
            
    # Case: Q1 != 0
    else:
        Delta = Q2**2 - 4*Q1*Q3
        # Case: Delta <= 0
        if Delta <= 0:
            if Q1 > 0:
                if Delta == 0:
                    return [(-Q2/(2*Q1), -Q2/(2*Q1))]
                return []
            else:
                return [(np.NINF, np.Inf)]
        # Case: Delta > 0
        else:
            sol1 = (-Q2 + np.sqrt(Delta))/(2*Q1)
            sol2 = (-Q2 - np.sqrt(Delta))/(2*Q1)
            # Ensure sol1 < sol2
            if sol1 > sol2:
                sol1, sol2 = sol2, sol1
            # Case: Q1 > 0
            if Q1 > 0:
                return [(sol1, sol2)]
            else:
                return [(np.NINF, sol1), (sol2, np.Inf)]



                

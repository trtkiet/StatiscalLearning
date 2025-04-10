import numpy as np

def solveEquation(Intervals, Q1, Q2, Q3):
    # Case: Q1 = 0
    if Q1 == 0:
        left = None
        right = None
        # Case: Q1, Q2 = 0 
        if Q2 == 0:
            if Q3 > 0:
                return 
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
            Intervals.append((left, 1))
            Intervals.append((right, -1))
            
    # Case: Q1 != 0
    else:
        Delta = Q2**2 - 4*Q1*Q3
        # Case: Delta <= 0
        if Delta <= 0:
            if Q1 > 0:
                return
        # Case: Delta > 0
        else:
            sol1 = (-Q2 + np.sqrt(Delta))/(2*Q1)
            sol2 = (-Q2 - np.sqrt(Delta))/(2*Q1)
            # Ensure sol1 < sol2
            if sol1 > sol2:
                sol1, sol2 = sol2, sol1
            # Case: Q1 > 0
            if Q1 > 0:
                # [sol1, sol2]
                Intervals.append((sol1, 1))
                Intervals.append((sol2, -1))
            else:
                # [-oo, sol1]
                Intervals.append((np.NINF, 1))
                Intervals.append((sol1, -1))
                # [sol2, oo]
                Intervals.append((sol2, 1))
                Intervals.append((np.Inf, -1))



                

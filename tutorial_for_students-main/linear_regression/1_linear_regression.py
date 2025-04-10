import numpy as np
import gen_data
import matplotlib.pyplot as plt

def run():
    true_beta = [0]
    X, y = gen_data.generate(20, 1, true_beta)

    # Estimate beta
    XTX = np.dot(X.T, X)
    XTXinv = np.linalg.inv(XTX)
    XTXinvXT = np.dot(XTXinv, X.T)
    beta = np.dot(XTXinvXT, y)

    print('Beta:', beta)

if __name__ == '__main__':
    run()
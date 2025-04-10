import numpy as np


def generate(n, p, true_beta):
    X = np.random.normal(loc=0, scale=1, size=(n, p))
    true_beta = np.reshape(true_beta, (p, 1))

    true_y = np.dot(X, true_beta)
    noise = np.random.normal(loc=0, scale=1, size=(n, 1))
    y = true_y + noise
    return X, y

# if __name__ == '__main__':
#     true_beta = [0, 0, 0, 0, 0]
#     X, y = generate(10, 5, true_beta)
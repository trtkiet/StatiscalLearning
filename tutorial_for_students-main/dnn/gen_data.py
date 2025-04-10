import numpy as np


def generate_train(sample_size, n, mu_background, mu_object, flag):
    set_data = []
    set_label = []

    mu = []
    true_label = []

    for i in range(n):
        if (i < n / 4) or (i >= 3 * n / 4):
            mu.append(mu_background)
            true_label.append(0.0)
        else:
            mu.append(mu_object)
            if flag == 1:
                true_label.append(1.0)
            else:
                true_label.append(0.0)

    for _ in range(sample_size):

        noise = np.random.multivariate_normal(np.zeros(n), np.identity(n))
        X = mu + noise
        set_data.append(X)
        set_label.append(true_label)

    return np.array(set_data), np.array(set_label)


def generate_test(n, mu_background, mu_object):
    mu = []

    for i in range(n):
        if (i < n / 4) or (i >= 3 * n / 4):
            mu.append(mu_background)
        else:
            mu.append(mu_object)

    noise = np.random.multivariate_normal(np.zeros(n), np.identity(n))
    X = mu + noise

    return np.array(X)


# if __name__ == '__main__':
#     # generate(1, 4, 0, 0)
#
#     X, y = generate_train(1, 8, 0, 1)
#
#     print(X)
#     print(y)
#     print(X.shape)
#     print(y.shape)
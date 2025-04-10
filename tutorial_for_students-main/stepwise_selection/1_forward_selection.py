import numpy as np
import gen_data


def calculate_rss(X_temp, y):
    # OLS
    inv = np.linalg.inv(
        np.dot(X_temp.T, X_temp)
    )
    beta = np.dot(np.dot(inv, X_temp.T), y)

    # y_hat
    y_hat = np.dot(X_temp, beta)

    # rss
    rss = np.sum((y - y_hat) ** 2)

    return rss


def run():
    n = 20
    p = 5
    k = 3
    true_beta = [1, 2, 0.25, 0.0, 0.75]
    X, y = gen_data.generate(n, p, true_beta)

    set_selected_features = []
    remaining_features = np.arange(p).tolist()

    # Forward selection
    for _ in range(k):
        rss_min = np.inf
        selected_feature = None

        for j in remaining_features:

            set_features_temp = set_selected_features.copy()
            set_features_temp.append(j)
            set_features_temp.sort()

            rss_temp = calculate_rss(X[:, set_features_temp].copy(), y)

            if rss_temp < rss_min:
                rss_min = rss_temp
                selected_feature = j

        set_selected_features.append(selected_feature)
        remaining_features.remove(selected_feature)

    set_selected_features.sort()

    print('Selected feature set:', set_selected_features)


if __name__ == '__main__':
    run()
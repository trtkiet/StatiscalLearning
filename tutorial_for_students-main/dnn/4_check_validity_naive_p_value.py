import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

import matplotlib.pyplot as plt
import statsmodels.api as sm

import gen_data

def run(n, model):
    mu_background = 0
    mu_object = 0

    X_obs = gen_data.generate_test(n, mu_background, mu_object)

    prediction = model(torch.tensor(X_obs, dtype=torch.float32))
    prediction = prediction.detach().numpy()  # convert to numpy array

    classification_result = (prediction > 0.6).astype(int)

    # Construct eta
    eta_1 = np.zeros((n, 1))
    for i in range(n):
        if classification_result[i] == 1:
            eta_1[i][0] = 1

    eta_2 = np.zeros((n, 1))
    for i in range(n):
        if classification_result[i] == 0:
            eta_2[i][0] = 1

    if (np.sum(classification_result) == 0.0):
        # Skip
        return None

    eta = 1 / np.sum(eta_1) * eta_1 - 1 / np.sum(eta_2) * eta_2

    # print(prediction)
    # print(classification_result)

    X_obs = X_obs.reshape((n, 1))

    eta = eta * np.sign(np.dot(eta.T, X_obs)[0][0])

    # Observed test-statistic
    etaTX_obs = np.dot(eta.T, X_obs)[0][0]

    # Compute two-sided naive-p value
    cdf = norm.cdf(etaTX_obs, loc=0, scale=np.sqrt(np.dot(eta.T, eta)[0][0]))
    naive_p_value = 2 * min(1 - cdf, cdf)

    return naive_p_value


if __name__ == '__main__':
    n = 16

    class DNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(n, 128)
            self.relu = nn.ReLU()
            self.hidden_2 = nn.Linear(128, 64)
            self.relu_2 = nn.ReLU()
            self.output = nn.Linear(64, n)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.hidden(x))
            x = self.relu_2(self.hidden_2(x))
            x = self.sigmoid(self.output(x))
            return x

    model = DNN()
    model = torch.load('./model/model.pth')

    # run(n, model)

    detect = 0
    reject = 0

    max_iteration = 1200
    list_naive_p_value = []

    for each_iter in range(max_iteration):
        print(each_iter)
        naive_p_value = run(n, model)
        if naive_p_value is not None:
            list_naive_p_value.append(naive_p_value)

            detect = detect + 1
            if naive_p_value <= 0.05:
                reject = reject + 1

    print('False Positive Rate (FPR):', reject/detect)
    # Although we set alpha = 0.05, the FPR is usually greater than 0.05. This indicates that naive-p value
    # is UNRELIABLE in the sense that the FPR can NOT be controlled under alpha

    plt.rcParams.update({'font.size': 18})
    grid = np.linspace(0, 1, 101)
    # plt.switch_backend('agg')
    plt.plot(grid, sm.distributions.ECDF(np.array(list_naive_p_value))(grid), 'r-', linewidth=6, label='Pivot')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('z_pivot.png', dpi=100)
    plt.show()







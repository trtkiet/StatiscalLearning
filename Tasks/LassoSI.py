import numpy as np
from sklearn.linear_model import Lasso
from scipy.stats import norm

# Helper function to compute the selective bias correction
def selective_inference(beta, X, y, lambda_, active_set):
    n, p = X.shape
    active_X = X[:, active_set]

    # Refit the model on the active set
    beta_active = np.linalg.inv(active_X.T @ active_X) @ active_X.T @ y

    # Compute the standard error of the estimates
    residual = y - active_X @ beta_active
    sigma = np.sqrt(np.sum(residual ** 2) / (n - len(active_set)))
    
    # Confidence intervals for the active variables
    confidence_intervals = []
    p_values = []
    for j in range(len(active_set)):
        x_j = active_X[:, j]
        h_ii = x_j.T @ np.linalg.inv(active_X.T @ active_X) @ x_j
        se = sigma * np.sqrt(h_ii)

        # Confidence interval
        ci_lower = beta_active[j] - norm.ppf(0.975) * se
        ci_upper = beta_active[j] + norm.ppf(0.975) * se
        confidence_intervals.append((ci_lower, ci_upper))

        # p-value
        z = beta_active[j] / se
        p_value = 2 * (1 - norm.cdf(np.abs(z)))
        p_values.append(p_value)

    return beta_active, confidence_intervals, p_values

# Example data
np.random.seed(42)
n, p = 100, 10
X = np.random.randn(n, p)
beta_true = np.array([1.5, -2.0, 0, 0, 0, 0, 0, 0, 0, 0])
y = X @ beta_true + np.random.randn(n) * 0.5

# Fit the Lasso model
lambda_ = 0.1
lasso = Lasso(alpha=lambda_, fit_intercept=False)
lasso.fit(X, y)

# Identify the active set
active_set = np.where(lasso.coef_ != 0)[0]
print(f"Active set: {active_set}")

# Perform post-selection inference
beta_active, confidence_intervals, p_values = selective_inference(lasso.coef_, X, y, lambda_, active_set)

print("Post-selection inference results:")
for i, j in enumerate(active_set):
    print(f"Variable {j}: Beta = {beta_active[i]:.4f}, CI = {confidence_intervals[i]}, p-value = {p_values[i]:.4f}")
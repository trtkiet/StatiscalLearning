import numpy as np
import scipy.stats as stats
from sklearn.linear_model import Lasso
from scipy.optimize import minimize

class ExactPostSelectionInference:
    def __init__(self, X, y, lambda_param):
        """
        Initialize the post-selection inference for Lasso.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Design matrix of predictors
        y : numpy.ndarray
            Response vector
        lambda_param : float
            Regularization parameter for Lasso
        """
        self.X = X
        self.y = y
        self.lambda_param = lambda_param
        self.n, self.p = X.shape
        
        # Perform Lasso regression
        self.lasso = Lasso(alpha=lambda_param, fit_intercept=False)
        self.lasso.fit(X, y)
        
        # Get selected model coefficients
        self.beta_selected = self.lasso.coef_
        self.selected_indices = np.where(np.abs(self.beta_selected) > 0)[0]
    
    def _compute_active_set(self):
        """
        Compute the active set of selected predictors.
        
        Returns:
        --------
        numpy.ndarray
            Indices of selected predictors
        """
        return self.selected_indices
    
    def _compute_sign_vector(self):
        """
        Compute the sign vector of the selected coefficients.
        
        Returns:
        --------
        numpy.ndarray
            Sign vector of selected coefficients
        """
        return np.sign(self.beta_selected[self.selected_indices])
    
    def _compute_projection_matrix(self):
        """
        Compute the projection matrix for the selected predictors.
        
        Returns:
        --------
        numpy.ndarray
            Projection matrix
        """
        X_selected = self.X[:, self.selected_indices]
        return X_selected @ np.linalg.inv(X_selected.T @ X_selected) @ X_selected.T
    
    def _compute_residual_projection(self):
        """
        Compute the projection of residuals.
        
        Returns:
        --------
        numpy.ndarray
            Residual projection
        """
        I = np.eye(self.n)
        proj_matrix = self._compute_projection_matrix()
        return I - proj_matrix
    
    def compute_inference(self):
        """
        Perform exact post-selection inference.
        
        Returns:
        --------
        dict
            Inference results including p-values and confidence intervals
        """
        # Get active set and sign vector
        active_set = self._compute_active_set()
        sign_vector = self._compute_sign_vector()
        X_selected = self.X[:, active_set]
        
        # Compute projection matrices
        proj_matrix = self._compute_projection_matrix()
        residual_proj = self._compute_residual_projection()
        
        # Compute standard error
        residuals = self.y - proj_matrix @ self.y
        sigma_hat = np.sqrt(np.sum(residuals**2) / (self.n - len(active_set)))
        
        # Compute inference for each selected coefficient
        results = {}
        for i, (idx, coef_sign) in enumerate(zip(active_set, sign_vector)):
            # Compute test statistic
            beta_j = self.beta_selected[idx]
            X_j = self.X[:, idx]
            
            # Compute conditional test statistic
            def objective_function(t):
                # Conditional test statistic computation
                # This is a simplified version and may need refinement
                return np.abs(t - beta_j) / (sigma_hat * np.sqrt(np.linalg.inv(X_j.T @ X_j)[0,0]))
            
            # Minimize the objective function
            res = minimize(objective_function, beta_j)
            conditional_statistic = res.x[0]
            
            # Compute p-value (simplified approach)
            p_value = 2 * (1 - stats.norm.cdf(np.abs(conditional_statistic)))
            
            # Compute confidence interval (approximate)
            margin_of_error = stats.norm.ppf(0.975) * sigma_hat
            ci_lower = beta_j - margin_of_error
            ci_upper = beta_j + margin_of_error
            
            results[idx] = {
                'coefficient': beta_j,
                'p_value': p_value,
                'confidence_interval': (ci_lower, ci_upper),
                'selected_sign': coef_sign
            }
        
        return results

# Example usage
def example_demonstration():
    # Generate synthetic data
    np.random.seed(42)
    n, p = 100, 10
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -1.0, 0.5]  # True non-zero coefficients
    X = np.random.randn(n, p)
    y = X @ true_beta + np.random.randn(n) * 0.5
    
    # Perform post-selection inference
    lambda_param = 0.1  # Lasso regularization parameter
    psi = ExactPostSelectionInference(X, y, lambda_param)
    
    # Compute inference results
    inference_results = psi.compute_inference()
    
    # Print results
    print("Post-Selection Inference Results:")
    for idx, result in inference_results.items():
        print(f"Predictor {idx}:")
        print(f"  Coefficient: {result['coefficient']:.4f}")
        print(f"  P-value: {result['p_value']:.4f}")
        print(f"  95% Confidence Interval: {result['confidence_interval']}")
        print(f"  Selected Sign: {result['selected_sign']}")

if __name__ == "__main__":
    example_demonstration()

# Notes on Limitations and Considerations:
# 1. This is a simplified implementation of exact post-selection inference
# 2. The conditional testing approach is approximated
# 3. More sophisticated methods exist in advanced statistical literature
# 4. Requires careful interpretation and validationx
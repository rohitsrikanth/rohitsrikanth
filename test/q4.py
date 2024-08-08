import numpy as np
from scipy.stats import norm

np.random.seed(42)
n_samples = 500
data = np.concatenate([np.random.normal(0, 2, n_samples),
                       np.random.normal(5, 1, n_samples)])

weights = np.array([0.5, 0.5])
means = np.array([0, 1])
variances = np.array([1, 1])

def gaussian(x, mean, var):
    return norm.pdf(x, mean, np.sqrt(var))

max_iter = 100
tolerance = 1e-6

for _ in range(max_iter):
    resp1 = weights[0] * gaussian(data, means[0], variances[0])
    resp2 = weights[1] * gaussian(data, means[1], variances[1])
    total_resp = resp1 + resp2
    gamma1 = resp1 / total_resp
    gamma2 = resp2 / total_resp
    N1 = np.sum(gamma1)
    N2 = np.sum(gamma2)

    weights[0] = N1 / len(data)
    weights[1] = N2 / len(data)

    means[0] = np.sum(gamma1 * data) / N1
    means[1] = np.sum(gamma2 * data) / N2

    variances[0] = np.sum(gamma1 * (data - means[0])**2) / N1
    variances[1] = np.sum(gamma2 * (data - means[1])**2) / N2

    # Check for convergence (this is a simple check on the weights)
    if np.abs(weights[0] - N1 / len(data)) < tolerance and \
       np.abs(weights[1] - N2 / len(data)) < tolerance:
        break

print("Weights:", weights)
print("Means:", means)
print("Variances:", variances)

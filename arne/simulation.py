import numpy as np


def simulate_categorical(n_samples: int, relevance: float):
    X = np.zeros((n_samples, 5))
    X[:, 0] = np.random.normal(0, 1, n_samples)
    n_categories = [2, 4, 10, 20]
    for i in range(1, 5):
        X[:, i] = np.random.choice(
            a=n_categories[i-1], size=n_samples,
              p=np.ones(n_categories[i - 1]) / n_categories[i - 1])
    y = np.zeros(n_samples)
    y[X[:, 1] == 0] = np.random.binomial(1, 0.5 - relevance, np.sum(X[:, 1] == 0))
    y[X[:, 1] == 1] = np.random.binomial(1, 0.5 + relevance, np.sum(X[:, 1] == 1))
    return X, y

def simulate_continuous(n_samples: int, relevance: float):
    X = np.zeros((n_samples, 5))
    variances = [1, 2, 4, 10, 20]
    for i in range(5):
        X[:, i] = np.random.normal(0, variances[i], n_samples)
    y = np.zeros(n_samples)
    y[X[:, 1] > 0] = np.random.binomial(1, 0.5 - relevance, np.sum(X[:, 1] > 0))
    y[X[:, 1] <= 0] = np.random.binomial(1, 0.5 + relevance, np.sum(X[:, 1] <= 0))
    return X, y

def simulate_continuous_gmm(n_samples: int, relevance: float):
    X = np.zeros((n_samples, 5))
    distances = [0.5, 1, 2, 5, 10]
    for i in range(5):
        gaussian_choice = np.random.choice([0, 1], p=[0.5, 0.5], size=n_samples)
        d = distances[i]
        X[gaussian_choice == 0, i] = np.random.normal(-d/2, 1/d, np.sum(gaussian_choice == 0))
        X[gaussian_choice == 1, i] = np.random.normal(d/2, 1/d, np.sum(gaussian_choice == 1))
    y = np.zeros(n_samples)
    y[X[:, 1] > 0] = np.random.binomial(1, 0.5 - relevance, np.sum(X[:, 1] > 0))
    y[X[:, 1] <= 0] = np.random.binomial(1, 0.5 + relevance, np.sum(X[:, 1] <= 0))
    return X, y
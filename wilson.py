import scipy.stats
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import numpy as np

def wilson(p_hat, n, alpha=0.05):
    z = scipy.stats.norm.ppf(1 - alpha/2)
    margin = z/(1 + z**2/n)*(p_hat*(1 - p_hat)/n + z**2/4/n**2)**0.5
    est = (p_hat + z**2/2/n) / (1 + z**2/n)
    return est - margin, est + margin

print(wilson(0.9, 10))

p_hat = 0.9
n = 10
z = scipy.stats.norm.ppf(1 - 0.05/2)

res_upper = root_scalar(lambda p: p*(1-p)/n - ((p - p_hat)/z)**2, x0=1.0)
res_lower = root_scalar(lambda p: p*(1-p)/n - ((p - p_hat)/z)**2, x0=0.0)
print(res_lower.root, res_upper.root)
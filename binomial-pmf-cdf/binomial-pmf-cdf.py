import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    n: number of trials
    p: probability of success
    k: number of successes
    """

    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    cdf = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))
    
    return pmf, cdf

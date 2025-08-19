import numpy as np
import torch

import pandas as pd

from scipy.stats import truncnorm
from scipy.stats import skewnorm
from scipy.stats import lognorm
from scipy.stats import powerlaw


def truncated_normal_dist(a_trunc, b_trunc, seq_len):
    loc = np.random.uniform(0.2, 0.8)
    scale = np.random.uniform(0.0, 0.3)
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=seq_len)

def truncated_left_skew_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(-20.0, 0.0)
    loc = np.random.uniform(0.5, 1.0)
    scale = np.random.uniform(0.0, 0.2)

    quantile1 = skewnorm.cdf(a_trunc, skewness, loc=loc, scale=scale)
    quantile2 = skewnorm.cdf(b_trunc, skewness, loc=loc, scale=scale)

    return skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        skewness,
        loc=loc,
        scale=scale,
    )


def truncated_neg_log_normal_dist(a_trunc, b_trunc, seq_len):
    sigma = np.random.uniform(0.25, 1.0)
    loc = np.random.uniform(0.0, 0.4)
    scale = np.random.uniform(0.0, 0.2)

    quantile1 = lognorm.cdf(a_trunc, sigma, loc=loc, scale=scale)
    quantile2 = lognorm.cdf(b_trunc, sigma, loc=loc, scale=scale)

    return 1 - lognorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        sigma,
        loc=loc,
        scale=scale,
    )


def truncated_neg_powerlaw_dist(a_trunc, b_trunc, seq_len):
    a = np.random.uniform(0.0, 0.5)
    loc = np.random.uniform(0.0, 0.5)
    scale = np.random.uniform(0.0, 0.2)

    quantile1 = powerlaw.cdf(a_trunc, a, loc=loc, scale=scale)
    quantile2 = powerlaw.cdf(b_trunc, a, loc=loc, scale=scale)

    return 1 - powerlaw.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a,
        loc=loc,
        scale=scale,
    )


def get_batch_func(
    batch_size,
    seq_len,
    num_features = 1,
    device="cpu",
    hyperparameters=None,
    noisy_target=True,
    num_outputs = 3,
    **_,
):
    list_of_dist_functions = [
        truncated_normal_dist,
        truncated_left_skew_normal_dist,
        truncated_neg_log_normal_dist,
        truncated_neg_powerlaw_dist,
    ]
    a_trunc, b_trunc = 0, 1

    xs = np.zeros((seq_len, batch_size, num_features))
    ys = np.zeros((seq_len, batch_size, num_outputs))

    for i in range(batch_size):
        xs[:, i, 0] = np.arange(1, seq_len + 1)
        func = np.random.choice(list_of_dist_functions)
        data = func(a_trunc, b_trunc, seq_len)
        ys[:, i, 0] = np.maximum.accumulate(data, axis=0)
        ys[:, i, 1] =  np.cumsum(data) / np.arange(1, len(data) + 1)
        ys[:, i, 2] = np.minimum.accumulate(data, axis=0)

    xs = torch.from_numpy(xs.astype(np.float32))
    ys = torch.from_numpy(ys.astype(np.float32))

    return xs.to(device), ys.to(device), ys.to(device)


# data = get_batch_func(100,200)
# print(data[1].shape)
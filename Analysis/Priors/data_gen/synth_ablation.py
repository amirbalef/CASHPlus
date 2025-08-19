import numpy as np
import torch

import pandas as pd

from scipy.stats import truncnorm
from scipy.stats import skewnorm
from scipy.stats import lognorm
from scipy.stats import powerlaw


def neg_skew_plus_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(-100.0, 100.0)
    loc = np.random.uniform(0.2, 0.8)
    scale = min((1 - loc) / 3, (loc) / 3, np.random.uniform(0.0, 0.3))

    quantile1 = skewnorm.cdf(0.001, a=skewness, loc=loc, scale=scale)
    quantile2 = skewnorm.cdf(0.999, a=skewness, loc=loc, scale=scale)

    samples = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len)**(1/ np.linspace(1, seq_len + 1, seq_len)),
        a=skewness,
        loc=loc,
        scale=scale,
    )
    samples[samples > 0.999] = 0.999
    samples[samples < 0.001] = 0.001
    return samples



def neg_skew_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(-100.0, -20.0)
    loc = np.random.uniform(0.0, 1.0)
    scale = min((loc) / 3, np.random.uniform(0.0, 0.2))

    quantile1 = skewnorm.cdf(0.001, a=skewness, loc=loc, scale=scale)
    quantile2 = skewnorm.cdf(0.999, a=skewness, loc=loc, scale=scale)

    samples = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=loc,
        scale=scale,
    )
    samples[samples > 0.999] = 0.999
    samples[samples < 0.001] = 0.001
    return samples


def pos_skew_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(20.0, 100.0)
    loc = np.random.uniform(0.0, 0.3)
    scale = min((1 - loc) / 3, np.random.uniform(0.0, 0.3))

    quantile1 = skewnorm.cdf(0.01, a=skewness, loc=loc, scale=scale)
    quantile2 = skewnorm.cdf(0.99, a=skewness, loc=loc, scale=scale)
    samples = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=loc,
        scale=scale,
    )
    samples[samples > 0.99] = 0.99
    samples[samples < 0.01] = 0.01
    return samples


def mix_skew_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(-100.0, 100.0)
    loc = np.random.uniform(0.2, 0.8)
    scale = min((1 - loc) / 3, (loc) / 3, np.random.uniform(0.0, 0.3))

    quantile1 = skewnorm.cdf(0.01, a=skewness, loc=loc, scale=scale)
    quantile2 = skewnorm.cdf(0.99, a=skewness, loc=loc, scale=scale)
    samples = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=loc,
        scale=scale,
    )
    samples[samples > 0.99] = 0.99
    samples[samples < 0.01] = 0.01
    return samples


def get_batch_func(
    batch_size,
    seq_len,
    num_features=1,
    device="cpu",
    hyperparameters=None,
    noisy_target=True,
    num_outputs=1,
    function_index=0,
    to_torch=False,
    **_,
):
    list_of_dist_functions = [
        neg_skew_plus_normal_dist,
        neg_skew_normal_dist,
        pos_skew_normal_dist,
        mix_skew_normal_dist,
    ]
    a_trunc, b_trunc = 0, 1

    xs = np.zeros((seq_len, batch_size, num_features))
    ys = np.zeros((seq_len, batch_size, num_outputs))
    ys_noisy = np.zeros((seq_len, batch_size, num_outputs))

    for i in range(batch_size):
        xs[:, i, 0] = np.arange(1, seq_len + 1)
        func = list_of_dist_functions[function_index]
        data = func(a_trunc, b_trunc, seq_len)
        ys[:, i, 0] = np.maximum.accumulate(data, axis=0)
        ys_noisy[:, i, 0] = np.maximum.accumulate(
            data + np.random.normal(0.0, 0.001, seq_len), axis=0
        )
        if num_outputs > 1:
            ys[:, i, 1] = data

    if to_torch:
        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        ys_noisy = torch.from_numpy(ys_noisy.astype(np.float32))

        return xs.to(device), ys.to(device), ys.to(device)
    else:
        return xs, ys, ys
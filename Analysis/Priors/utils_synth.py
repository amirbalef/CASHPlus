
from scipy.stats import truncnorm
from scipy.stats import skewnorm
from scipy.stats import lognorm
from scipy.stats import powerlaw

import numpy as np
import torch


def truncated_normal_dist(a_trunc, b_trunc, seq_len):
    # loc = np.random.normal(0.8, 0.1)
    # uniform
    loc = np.random.uniform(0.2, 0.8)
    scale = np.random.uniform(0.0, 0.3)
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=seq_len)


def truncated_left_skew_normal_dist(a_trunc, b_trunc, seq_len):
    a = 1 - 1/ np.arange(1, seq_len + 1)
    skewness = np.random.uniform(-a*20, 0.2)
    loc = np.random.uniform(0.5, 1.0)
    scale = np.random.uniform(0.2, 0.4)

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
    maximum = np.random.uniform(0.5, 0.8)
    scale = np.random.uniform(0.0, 0.4)
    loc = 1 - (
        maximum + np.maximum.accumulate(np.random.normal(0.0, 0.2 * scale, seq_len))
    )
    a_values = np.random.uniform(0.0, 0.3)
    quantile1 = powerlaw.cdf(a_trunc, a_values, loc=loc, scale=scale)
    quantile2 = powerlaw.cdf(b_trunc, a_values, loc=loc, scale=scale)
    uniform_samples = np.random.uniform(quantile1, quantile2)
    return 1 - powerlaw.ppf(uniform_samples, a_values, loc=loc, scale=scale)



def truncated_custom_mixture_dist(a_trunc, b_trunc, seq_len):
    initial_length = int(np.random.uniform(0.05, 0.3)* seq_len)
    loc = 1 - np.random.uniform(0.6, 0.9) # 1 - maximum
    scale = np.random.uniform(0, 0.4)
    a_values = np.random.uniform(0.1, 0.4) 

    quantile1 = powerlaw.cdf(a_trunc, a_values, loc=loc, scale=scale)
    quantile2 = powerlaw.cdf(b_trunc, a_values, loc=loc, scale=scale)
    uniform_samples = np.random.uniform(
        quantile1, quantile2, size=seq_len - initial_length
    )
    res1 = 1 - powerlaw.ppf(uniform_samples, a_values, loc=loc, scale=scale)

    b = 1 - loc
    a = b - scale
    res2 = np.random.uniform(a, b, size=initial_length)

    return np.concatenate([res2, res1], axis=0)


# def truncated_skew_normal_dist(a_trunc, b_trunc, seq_len):
#     skewness = 100
#     loc = np.random.uniform(0.0, 1.0)
#     scale = np.random.uniform(0.0, 0.2)

#     quantile1 = skewnorm.cdf(a_trunc, skewness, loc=loc, scale=scale)
#     quantile2 = skewnorm.cdf(b_trunc, skewness, loc=loc, scale=scale)

#     return skewnorm.ppf(
#         np.random.uniform(quantile1, quantile2, size=seq_len),
#         skewness,
#         loc=loc,
#         scale=scale,
#     )

def truncated_skew_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = -100
    loc = np.random.uniform(0.0, 1.0)
    scale = np.random.uniform(0.0, 0.3)

    samples = skewnorm.rvs(a=skewness, loc=loc, scale=scale, size=seq_len)
    # Ensure values are within [0,1]
    min_val = np.min(samples)
    if min_val < 0 :
        samples = (samples - min_val)
    max_val = np.max(samples)
    if max_val > 1:
        samples = samples/max_val
    return samples


def priors(a_trunc, b_trunc, seq_len):
    loc = np.random.uniform(0.5, 0.95)
    scale = np.random.uniform(0.0, 0.05)
    initial_length = int(np.random.uniform(0.05, 0.1) * seq_len)
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    initial_seq = truncnorm.rvs(a, b, loc=loc, scale=scale, size=initial_length)


    scale = np.random.uniform(0.0, 0.2)
    loc = 1 - np.max(initial_seq) 

    a_values =np.random.uniform(0.2, 0.8) # skewness

    quantile1 = powerlaw.cdf(a_trunc, a_values, loc=loc, scale=scale)
    quantile2 = powerlaw.cdf(b_trunc, a_values, loc=loc, scale=scale)
    uniform_samples = np.random.uniform(
        quantile1, quantile2, size=seq_len - initial_length
    )
    res1 = 1 - powerlaw.ppf(uniform_samples, a_values, loc=loc, scale=scale)


    return np.concatenate([initial_seq, res1], axis=0)


def get_batch_func(
    batch_size,
    seq_len,
    num_features,
    device="cpu",
    hyperparameters=None,
    noisy_target=True,
    **_,
):
    num_features = 1  # t, x, v
    num_outputs = 1

    list_of_dist_functions = [
        # truncated_normal_dist,
        # truncated_left_skew_normal_dist,
        # truncated_neg_log_normal_dist,
        # truncated_neg_powerlaw_dist,
        #truncated_skew_normal_dist,
        priors,
    ]
    a_trunc, b_trunc = 0, 1

    xs = np.zeros((seq_len, batch_size, num_features))
    ys = np.zeros((seq_len, batch_size, num_outputs))
    y_raw = np.zeros((seq_len, batch_size, num_outputs))

    for i in range(batch_size):
        xs[:, i, 0] = np.arange(1, seq_len + 1)
        datas =[]
        for func in list_of_dist_functions:
        #func = list_of_dist_functions[-1] #np.random.choice(list_of_dist_functions)
            datas.append(func(a_trunc, b_trunc, seq_len))
        data = np.mean(datas, axis = 0)
        # max_so_far = np.maximum.accumulate(data)  # Running max
        # improvement = np.zeros_like(data)  # Pre-allocate output array
        # improvement[1:] = np.maximum(0, data[1:] - max_so_far[:-1])

        ys[:, i, 0] =  np.maximum.accumulate(data) #improvement
        y_raw[:, i, 0] = data

    xs = torch.from_numpy(xs.astype(np.float32))
    ys = torch.from_numpy(ys.astype(np.float32))

    y_raw = torch.from_numpy(y_raw.astype(np.float32))

    return xs.to(device), ys.to(device), y_raw.to(device)
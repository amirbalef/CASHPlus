import numpy as np
import torch
from scipy.stats import skewnorm
from scipy.stats import lognorm

def paper_flat_dist(a_trunc, b_trunc, seq_len):
    sigma1 = 0.1
    sigma2 = 0.001
    skewness = np.random.uniform(-100.0, -20.0)
    loc1 = np.random.uniform(0.0, 1.0)
    scale1 = np.random.uniform(0.0, sigma1)

    quantile1 = skewnorm.cdf(0.0, a=skewness, loc=loc1, scale=scale1)
    quantile2 = skewnorm.cdf(1.0, a=skewness, loc=loc1, scale=scale1)
    samples_1 = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=loc1,
        scale=scale1,
    )

    skewness = np.random.uniform(-100.0, -20.0)
    scale2 = np.random.uniform(sigma2 * (1 - loc1), sigma2)
    quantile1 = skewnorm.cdf(0.0, a=skewness, loc=1.0, scale=scale2)
    quantile2 = skewnorm.cdf(1.0, a=skewness, loc=1.0, scale=scale2)
    samples_3 = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=1.0,
        scale=scale2,
    )
    samples_3 = np.sort(samples_3)
    mixed_samples = samples_1 * samples_3
    mixed_samples[mixed_samples > 1.0] = 1.0
    mixed_samples[mixed_samples < 0.0] = 0.0
    return mixed_samples


def paper_semi_flat_dist(a_trunc, b_trunc, seq_len):
    sigma1 = 0.2
    sigma2 = 0.01
    skewness = np.random.uniform(-100.0, -20.0)
    loc1 = np.random.uniform(0.0, 1.0)
    scale1 = np.random.uniform(0.0, sigma1)

    quantile1 = skewnorm.cdf(0.0, a=skewness, loc=loc1, scale=scale1)
    quantile2 = skewnorm.cdf(1.0, a=skewness, loc=loc1, scale=scale1)
    samples_1 = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=loc1,
        scale=scale1,
    )

    skewness = np.random.uniform(-100.0, -20.0)
    scale2 = np.random.uniform(sigma2 * (1 - loc1), sigma2)
    quantile1 = skewnorm.cdf(0.0, a=skewness, loc=1.0, scale=scale2)
    quantile2 = skewnorm.cdf(1.0, a=skewness, loc=1.0, scale=scale2)
    samples_3 = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=1.0,
        scale=scale2,
    )
    samples_3 = np.sort(samples_3)
    mixed_samples = samples_1 * samples_3
    mixed_samples[mixed_samples > 1.0] = 1.0
    mixed_samples[mixed_samples < 0.0] = 0.0
    return mixed_samples


def paper_curved_dist(a_trunc, b_trunc, seq_len):
    sigma1 = 0.2
    sigma2 = 0.1
    skewness = np.random.uniform(-100.0, -20.0)
    loc1 = np.random.uniform(0.0, 1.0)
    scale1 = np.random.uniform(0.0, sigma1)

    quantile1 = skewnorm.cdf(0.0, a=skewness, loc=loc1, scale=scale1)
    quantile2 = skewnorm.cdf(1.0, a=skewness, loc=loc1, scale=scale1)
    samples_1 = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=loc1,
        scale=scale1,
    )

    skewness = np.random.uniform(-100.0, -20.0)
    scale2 = np.random.uniform(sigma2 * (1 - loc1), sigma2)
    quantile1 = skewnorm.cdf(0.0, a=skewness, loc=1.0, scale=scale2)
    quantile2 = skewnorm.cdf(1.0, a=skewness, loc=1.0, scale=scale2)
    samples_3 = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=1.0,
        scale=scale2,
    )
    samples_3 = np.sort(samples_3)
    mixed_samples = samples_1 * samples_3
    mixed_samples[mixed_samples > 1.0] = 1.0
    mixed_samples[mixed_samples < 0.0] = 0.0
    return mixed_samples


def output_func(output, data):
    if output == "max":
        return np.maximum.accumulate(data, axis=0)
    if output == "min":
        return np.minimum.accumulate(data, axis=0)
    if output == "raw":
        return data


def get_batch_func(
    batch_size,
    seq_len,
    num_features=1,
    device="cpu",
    hyperparameters=None,
    noisy_target=True,
    outputs=["max"],
    function_index=0,
    max_seq_len=200,
    **_,
):
    list_of_dist_functions = [
        paper_flat_dist,  # 0
        paper_semi_flat_dist,  # 1
        paper_curved_dist,  # 2
    ]
    a_trunc, b_trunc = 0, 1

    xs = np.zeros((seq_len, batch_size, num_features))
    ys = np.zeros((seq_len, batch_size, len(outputs)))

    for i in range(batch_size):
        start_index = 0 if seq_len == max_seq_len else np.random.randint(0, max_seq_len - seq_len + 1)
        xs[:, i, 0] = np.arange(1, max_seq_len + 1)[start_index : start_index + seq_len]
        func = list_of_dist_functions[function_index]
        data = func(a_trunc, b_trunc, max_seq_len)
        for o, output in enumerate(outputs):
            ys[:, i, o] = output_func(output, data)[start_index : start_index + seq_len]

    xs = torch.from_numpy(xs.astype(np.float32))
    ys = torch.from_numpy(ys.astype(np.float32))

    return xs.to(device), ys.to(device), ys.to(device)

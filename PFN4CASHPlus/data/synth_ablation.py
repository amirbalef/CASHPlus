import numpy as np
import torch
from scipy.stats import skewnorm
from scipy.stats import lognorm

def neg_skew_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(-100.0, -20.0)
    loc = np.random.uniform(0.01, 0.99)
    scale = min((loc) / 5, np.random.uniform(0.0, 0.2))

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


def limited_neg_skew_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(-100.0, -20.0)
    loc = np.random.uniform(0.7, 0.99)
    scale = min((loc) / 5, np.random.uniform(0.0, 0.2))

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




def pos_skew_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(20.0, 100.0)
    loc = np.random.uniform(0.01, 0.99)
    scale = min((1 - loc) / 5, np.random.uniform(0.0, 0.2))

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
    loc = np.random.uniform(0.01, 0.99)
    scale = min((1 - loc) / 5, (loc) / 5, np.random.uniform(0.0, 0.2))

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



def mixed_mixture_skew_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(-100.0, -20.0)
    loc1 = np.random.uniform(0.2, 1.0)
    scale1 = np.random.uniform(0.0, 0.2)

    quantile1 = skewnorm.cdf(0.0, a=skewness, loc=loc1, scale=scale1)
    quantile2 = skewnorm.cdf(1.0, a=skewness, loc=loc1, scale=scale1)
    samples_1 = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=loc1,
        scale=scale1,
    )

    skewness = np.random.uniform(-100.0, -20.0)
    scale2 = np.random.uniform(0.0, 0.2)
    quantile1 = skewnorm.cdf(0.0, a=skewness, loc=1.0, scale=scale2)
    quantile2 = skewnorm.cdf(1.0, a=skewness, loc=1.0, scale=scale2)
    samples_3 = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=1.0,
        scale=scale2,
    )
    samples_3 = np.sort(samples_3)

    skewness = np.random.uniform(-100.0, -20.0)
    loc2 = np.random.uniform(loc1 - 0.2, loc1)
    quantile1 = skewnorm.cdf(0.0, a=skewness, loc=loc2, scale=scale2)
    quantile2 = skewnorm.cdf(1.0, a=skewness, loc=loc2, scale=scale2)
    samples_2 = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a=skewness,
        loc=loc2,
        scale=scale2,
    )

    mixed_samples = np.concatenate([samples_2, samples_1])
    np.random.shuffle(mixed_samples)
    mixed_samples = mixed_samples[:seq_len] * samples_3
    mixed_samples[mixed_samples > 1.0] = 1.0
    mixed_samples[mixed_samples < 0.0] = 0.0
    return mixed_samples


def neg_skew_plus_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(-100.0, -20.0)
    loc = np.random.uniform(0.0,1.0)
    scale = min((1 - loc) / 3, (loc) / 3, np.random.uniform(0.0, 0.3))

    quantile1 = skewnorm.cdf(0.001, a=skewness, loc=loc, scale=scale)
    quantile2 = skewnorm.cdf(0.999, a=skewness, loc=loc, scale=scale)

    samples = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len)**(1/ np.linspace(1,  np.random.randint(1, seq_len + 1), seq_len)),
        a=skewness,
        loc=loc,
        scale=scale,
    )
    samples[samples > 0.999] = 0.999
    samples[samples < 0.001] = 0.001
    return samples

def mix_plus_normal_dist(a_trunc, b_trunc, seq_len):
    skewness = np.random.uniform(-100.0, 100.0)
    loc = np.random.uniform(0.0,1.0)
    scale = min((1 - loc) / 3, (loc) / 3, np.random.uniform(0.0, 0.3))

    quantile1 = skewnorm.cdf(0.001, a=skewness, loc=loc, scale=scale)
    quantile2 = skewnorm.cdf(0.999, a=skewness, loc=loc, scale=scale)

    samples = skewnorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len)**(1/ np.linspace(1,  np.random.randint(1, seq_len + 1), seq_len)),
        a=skewness,
        loc=loc,
        scale=scale,
    )
    samples[samples > 0.999] = 0.999
    samples[samples < 0.001] = 0.001
    return samples

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
        neg_skew_normal_dist,
        pos_skew_normal_dist,
        mix_skew_normal_dist,
        limited_neg_skew_normal_dist,
        mixed_mixture_skew_normal_dist,
        neg_skew_plus_normal_dist,
        mix_plus_normal_dist,
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
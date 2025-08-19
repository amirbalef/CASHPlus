import numpy as np
import torch

import pandas as pd


def get_dataset(dataset_path = "../../Bandits/datasets/" +  "classification_TabRepoRaw" + ".csv"):
    dataset = pd.read_csv(dataset_path)
    instances = sorted(dataset["instance"].unique())
    all_arm_index_list = dataset["arm_index"].unique()
    valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
    number_of_arms = len(valid_arm_index_list)
    number_of_trails = len(dataset["repetition"].unique())
    horizon_time = len(dataset["iteration"].unique())

    df = dataset[(dataset["arm_index"] >= 0)]
    df = df[["instance", "arm_index", "repetition", "iteration", "loss"]]
    real_data = df.sort_values(by=["instance", "arm_index", "repetition", "iteration"])[
        "loss"
    ].values.reshape(len(instances), number_of_arms, number_of_trails, horizon_time)
    real_data = 1 - real_data.reshape(-1, horizon_time)
    real_data[real_data < 0] = 0
    return real_data


def output_func(output, data):
    if output == "max":
        return np.maximum.accumulate(data, axis=0)
    if output == "min":
        return np.minimum.accumulate(data, axis=0)
    if output == "raw":
        return data


def get_batch_func(
    real_data,
    batch_size,
    seq_len,
    num_features=1,
    device="cpu",
    hyperparameters=None,
    noisy_target=True,
    outputs=["max"],
    max_seq_len=200,
    **_,
):

    xs = np.zeros((seq_len, batch_size, num_features))
    ys = np.zeros((seq_len, batch_size, len(outputs)))

    for i in range(batch_size):
        start_index = np.random.randint(0, max_seq_len - seq_len + 1)
        xs[:, i, 0] = np.arange(1, max_seq_len + 1)[start_index : start_index + seq_len]

        data = real_data[np.random.randint(real_data.shape[0])]
        for o, output in enumerate(outputs):
            ys[:, i, o] = output_func(output, data)[start_index : start_index + seq_len]


    xs = torch.from_numpy(xs.astype(np.float32))
    ys = torch.from_numpy(ys.astype(np.float32))

    return xs.to(device), ys.to(device), ys.to(device)
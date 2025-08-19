
import numpy as np
import os
import time


def get_data(dataset_name, task_id, max_attempts=1):
    dataset = {}
    from tabularlab.utils import openml_datasets
    attempts = 0
    while attempts < max_attempts:
        try:
            openml_dataset = openml_datasets.OpenMLDataset()
            dataset["dataset_name"] = dataset_name
            openml_dataset.set_task(task_id)
            (X, y), k_fold_split_indx = openml_dataset.get_dataset()
            dataset["task_type"] = openml_dataset.task_type
            dataset["task_id"] = task_id
            break
        except Exception as e:
            print(f"Attempt {attempts + 1}: Error loading dataset:", e)
            attempts += 1
            if attempts == max_attempts:
                print("Failed to load dataset after 5 attempts.")
                exit()
            else:
                time.sleep(10)

    dataset["k_fold_split_indx"] = k_fold_split_indx
    dataset["X"] = X
    dataset["y"] = y
    time.sleep(0.1)
    return dataset


def get_data_splits(data, rng):
    dataset_info = {}
    dataset_info["task_id"] = data["task_id"]
    dataset_info["task_type"] = data["task_type"]
    dataset_info["k_fold_split_indx"] = data["k_fold_split_indx"]
    dataset_info["test_folds"] = [0]

    folds_order = np.arange(1, len(data["k_fold_split_indx"]), dtype=int)
    folds_permutation = rng.permutation(folds_order)
    dataset_info["folds_permutation"] = folds_permutation

    val_folds = folds_permutation[-2:]
    dataset_info["val_folds"] = val_folds

    train_folds = folds_permutation[:-2]
    dataset_info["train_folds"] = train_folds
    
    return (data["X"], data["y"]), dataset_info
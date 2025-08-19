import os
import pickle
import numpy as np
import pandas as pd


def fetch_results(policy_algorithms, result_directory, dataset_name):
    fetched_results = {}
    for alg_name, alg in policy_algorithms.items():
        if not os.path.exists(result_directory + dataset_name + "/" + alg_name):
            print(result_directory + dataset_name + "/" + alg_name)
            print("Error: please first run main_reproducing_results.py for " + alg_name)
            exit()
        else:
            with open(
                result_directory + dataset_name + "/" + alg_name + "/result.pkl", "rb"
            ) as file:
                result = pickle.load(file)
                fetched_results[alg_name] = result
    return fetched_results

def fetch_pullings(policy_algorithms, result_directory, dataset_name):
    fetched_results = {}
    for alg_name, alg in policy_algorithms.items():
        if not os.path.exists(result_directory + dataset_name + "/" + alg_name):
            print(result_directory + dataset_name + "/" + alg_name)
            print("Error: please first run main_reproducing_results.py for " + alg_name)
            exit()
        else:
            with open(
                result_directory + dataset_name + "/" + alg_name + "/pulled_arms.pkl",
                "rb",
            ) as file:
                result = pickle.load(file)
                fetched_results[alg_name] = result
    return fetched_results


def run_expriment(alg, data):
    number_of_arms,number_of_trails,  horizon_time = data.shape
    result = []
    result_pulled_arms = []

    for trial in range(number_of_trails):
        pulled_arms = np.zeros(number_of_arms, dtype=int)
        np.random.seed(trial)
        policy = alg(number_of_arms, T=horizon_time)
        result_reward_error = [np.nan]

        result_iteration = [0]
        pulled_arms_list = []
        t = 0
        while t < horizon_time:
            arm = policy.play()
            loss = data[
                arm, trial, pulled_arms[arm]
            ]
            policy.update_loss(loss)
            pulled_arms_list.append(arm)
            pulled_arms[arm] += 1
            t += 1
            result_iteration.append(t)
            result_reward_error.append(loss)

        result_timeseries = pd.Series(result_reward_error, index=result_iteration)
        pulled_arms_timeseries = pd.Series(
            pulled_arms_list, index=result_iteration[: len(pulled_arms_list)]
        )
        result.append(result_timeseries)
        result_pulled_arms.append(pulled_arms_timeseries)
    return result, result_pulled_arms

def run_fake_expriment(alg, data):
    number_of_arms, number_of_trails, horizon_time = data.shape
    oracle_arm = np.argmin(np.min(data, axis=-1), axis=0)
    result = []
    result_pulled_arms = []
    for trial in range(number_of_trails):
        result_reward_error = [np.nan]
        result_iteration = [0]
        t = 0
        pulled_arms_list = []
        while t < horizon_time:
            arm = oracle_arm[trial]
            reward_error = data[arm, trial, t]
            t += 1
            result_reward_error.append(reward_error)
            result_iteration.append(t)
            pulled_arms_list.append(arm)

        result_timeseries = pd.Series(result_reward_error, index=result_iteration)
        pulled_arms_timeseries = pd.Series(
            pulled_arms_list, index=result_iteration[: len(pulled_arms_list)]
        )
        result.append(result_timeseries)
        result_pulled_arms.append(pulled_arms_timeseries)
    return result, result_pulled_arms

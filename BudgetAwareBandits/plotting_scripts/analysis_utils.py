import numpy as np 
import pandas as pd
from scipy.stats import rankdata
import os
import pickle
import pandas as pd


def fetch_results(policy_algorithms, result_directory, dataset_name, test_results = False):
    if (test_results):
        f_name = "result_test.pkl"
    else:
        f_name = "result.pkl" 
    fetched_results = {}
    for alg_name, alg in policy_algorithms.items():
        if not os.path.exists(result_directory + dataset_name + "/" + alg_name):
            print(result_directory + dataset_name + "/" + alg_name)
            print("Error: please first run main_reproducing_results.py for " + alg_name)
            exit()
        else:
            with open(
                result_directory + dataset_name + "/" + alg_name + "/"+ f_name , "rb"
            ) as file:
                result = pd.read_pickle(file)
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

def get_ranks_per_instance_MC(
    all_result,
    horizon_time,
    number_of_arms,
    instances,
    number_of_trails,
    num_samples=100,
    method="average",
):
    
    error_results = np.zeros(
        (len(all_result), len(instances), number_of_trails, horizon_time)
    )
    for instance_num in range(len(instances)):
        for seed in range(number_of_trails):
            error = get_error(all_result, number_of_arms, instances, instance_num, seed)
            error_results[:, instance_num, seed, :] = error.T.to_numpy()[:, 1:]
    list_of_trials = np.arange(number_of_trails)
    mean_ranks = []
    for _ in range(num_samples):
        selected_trails = np.random.choice(
            list_of_trials, size=len(list_of_trials), replace=True
        )
        error_selected_trails = error_results[:, :, selected_trails, :]
        m_err_selected_trails = np.ma.masked_invalid(np.nanmean(error_selected_trails, axis=2)) 
        ranks = rankdata(m_err_selected_trails, axis=0)
        ranks = ranks.mean(axis = 1)
        mean_ranks.append(ranks)
    mean_ranks = np.asarray(mean_ranks)
    return mean_ranks.mean(axis=0), mean_ranks.std(axis=0)

def get_error( all_result, number_of_arms, instances, instance_num, seed, normalize=False):
    one_sample_results = list(all_result.values())[0][instance_num][seed]
    baseline_value = np.median(one_sample_results.values[1 : 1 + number_of_arms])
    list_timeseries = []
    top_error_value = np.inf
    for k, d in all_result.items():
        data = d[instance_num][seed]
        data = data.sort_index()  # Ensure the index is sorted in ascending order
        data = data[~data.index.isna()]  # Remove NaN index
        #data.loc[12.0] = np.nan
        data.index = pd.to_datetime(data.index, unit='h')  # Convert index to DatetimeIndex
        data = data.resample("20s").ffill().cummin()
        data = data[data.index <= (data.index[0] + pd.Timedelta(hours=1))]
        list_timeseries.append(data.rename(k))
        for item in d[instance_num]:
            top_error_value = min(
                top_error_value, np.nanmin(item)
            ) # we do not want to put weights on some seeds! 
    #top_error_value = max(  1e-3, top_error_value)
    df = pd.concat(list_timeseries, axis=1).sort_index()
    for key in df.keys():
        df[key] = df[key].fillna(value=np.max(df.dropna().max()))
    #if("Oracle_Arm" in df.keys()):
    #    df["Oracle_Arm"] = df["Oracle_Arm"].fillna(value=np.max(df.dropna().max()))
    df = df.ffill().cummin()
    denominator = max(1e-3, baseline_value - top_error_value)
    if(normalize):
        return (df - top_error_value) / (denominator)
    else:
        return df



def get_error_per_instance_time(
    all_result, number_of_arms, instances, number_of_trails, time, normalize= False
):
    mean_res_per_instance = []
    for instance_num in range(len(instances)):
        result_one_instance = []
        for seed in range(number_of_trails):
            normallized_error = get_error(
                all_result, number_of_arms, instances, instance_num, seed, normalize
            )
            result_one_instance.append(normallized_error)
        df = pd.concat(result_one_instance, axis=1).sort_index().ffill()
        df = df.loc[time]
        res = df.groupby(level=0).agg(lambda x: list(x))
        res = res.reindex(list(all_result.keys()))
        mean_res_per_instance.append(res)
    return mean_res_per_instance


def get_normallized_error_per_instance_time(
    all_result, number_of_arms, instances, number_of_trails, time):
    mean_res_per_instance = []
    for instance_num in range(len(instances)):
        result_one_instance = []
        for seed in range(number_of_trails):
            normallized_error = get_error(
                all_result,
                number_of_arms,
                instances,
                instance_num,
                seed,
                normalize=True)
            result_one_instance.append(normallized_error)
        df = pd.concat(result_one_instance, axis=1).sort_index().ffill()
        df = df.iloc[time]
        res = df.groupby(level=0).agg(lambda x: list(x))
        res = res.reindex(list(all_result.keys()))
        mean_res_per_instance.append(res)
    return mean_res_per_instance

def mean_if_no_nans(x):
    return x.mean(axis=1, skipna=True)


def get_error_per_seed(
    all_result, dataset, number_of_arms, instances, number_of_trails, normalize = True):
    mean_res_per_seed = []
    for seed in range(number_of_trails):
        result_all_instance = []
        for instance_num in range(len(instances)):
            error = get_error(
                all_result,
                number_of_arms,
                instances,
                instance_num,
                seed,
                normalize=normalize,
            )
            result_all_instance.append(error)
        df = pd.concat(result_all_instance, axis=1).sort_index().ffill().fillna(1)
        mean_res = df.groupby(level=0, axis=1).apply(mean_if_no_nans).fillna(np.inf)
        mean_res_per_seed.append(mean_res)
    return mean_res_per_seed



def get_ranks(
    all_result,
    horizon_time,
    number_of_arms,
    instances,
    number_of_trails,
    method="average",
):
    error_results = np.zeros(
        (len(all_result), len(instances), number_of_trails, horizon_time)
    )
    for instance_num in range(len(instances)):
        for seed in range(number_of_trails):
            error = get_error(all_result, number_of_arms, instances, instance_num, seed)
            error_results[:, instance_num, seed, :] = error.T.to_numpy()[:, 1:]
    ranks = rankdata(np.mean(error_results, axis=2), axis=0)
    ranks = ranks.mean(axis=1)
    return ranks.T

def get_ranks_per_instance_time(
    all_result,
    horizon_time,
    number_of_arms,
    instances,
    number_of_trails,
    time,
    method="average",
):
    all_ranks = []
    for instance_num in range(len(instances)):
        error_results = np.zeros((len(all_result), number_of_trails))
        for seed in range(number_of_trails):
            error = get_error(all_result, number_of_arms, instances, instance_num, seed)
            error_results[:, seed] = error.T.to_numpy()[:, time]
        ranks = rankdata(np.mean(error_results, axis=1), axis=0)
        all_ranks.append(ranks.T)
    return all_ranks
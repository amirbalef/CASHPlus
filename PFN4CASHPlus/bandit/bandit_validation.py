import os
import numpy as np 
import pandas as pd
import pickle 
from functools import partial
import wandb
from cycler import cycler

from .PFN_PS import PFN_PS
from . import exp_utils
from . import plt_utils

multiprocess = "joblib"
multiprocess = " "
if(multiprocess == "joblib"):
     import joblib


def run_expriment(model, device, dataset_name, outputs):
    data_info ={}
    dataset = pd.read_csv("../../Bandits/datasets/" + dataset_name + ".csv")
    instances = sorted(dataset["instance"].unique())
    all_arm_index_list = dataset["arm_index"].unique()
    valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
    number_of_arms = len(valid_arm_index_list)
    number_of_trails = len(dataset["repetition"].unique())
    horizon_time = len(dataset["iteration"].unique())
    combined_search_algorithms = list(
        dataset[dataset["arm_index"] < 0]["optimizer"].unique()
    )
    data_info["number_of_arms"] = number_of_arms
    data_info["instances"] = instances
    data_info["number_of_trails"] = number_of_trails
    data_info["horizon_time"] = horizon_time
    data_info["combined_search_algorithms"] = combined_search_algorithms
    data_info["dataset_name"] = dataset_name
    data_info["dataset"] = dataset

    df = dataset[(dataset["arm_index"]>=0)]
    df = df[["instance", "arm_index", "repetition", "iteration", "loss"]]
    data = df.sort_values(by=["instance", "arm_index", "repetition", "iteration"])[
        "loss"
    ].values.reshape(len(instances), number_of_arms, number_of_trails, horizon_time)

    if(model==None):
        return {}, {}, data_info

    alg = partial(PFN_PS, model=model, device=device, outputs=outputs)
    if(multiprocess=='joblib'):
        result, result_pulled_arms = zip(
            *joblib.Parallel(n_jobs=-1)(
                joblib.delayed(partial(exp_utils.run_expriment, alg))(data[instance_num])
                for instance_num in range(len(instances))
            )
        )
    else:
        result = []
        result_pulled_arms = []
        for instance_num in range(len(instances)):
            res, res_pulled = exp_utils.run_expriment(alg, data[instance_num])
            result.append(res)
            result_pulled_arms.append(res_pulled)

    return result, result_pulled_arms, data_info


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




def validate_model(model, device, dataset_name, name, outputs):
    print("Validating model for ", dataset_name)
    result, result_pulled_arms, data_info = run_expriment(
        model, device, dataset_name, outputs
    )
    policy_algorithms = {}
    policy_algorithms["MaxUCB"] = 1
    policy_algorithms["Rising_Bandit"] = 1

    number_of_policies = len(policy_algorithms.keys()) + 1

    if len(data_info["combined_search_algorithms"]) > 0:
        data_info["combined_search_algorithms"].append(
            data_info["combined_search_algorithms"].pop(0)
        )
        for algorithm in data_info["combined_search_algorithms"]:
            policy_algorithms[algorithm] = 1
    policy_algorithms["Oracle_Arm"] = 1

    result_directory = "./bandit/results/"
    results = fetch_results(policy_algorithms, result_directory, dataset_name)
    
    all_algorithm = list(policy_algorithms.keys())[: number_of_policies - 1] + ["PFN_PS"] + list(policy_algorithms.keys())[ number_of_policies - 1:]
    results["PFN_PS"] = result

    for key in all_algorithm:
        results[key] = results.pop(key)

    data_info["all_result"] = results

    CB_color_cycle = [
        "#377eb8",
        "#4daf4a",
        "#ff7f00",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#e41a1c",
        "#dede00",
        "#999999",
    ]
    colors = CB_color_cycle[:number_of_policies] 
    all_cyclers = cycler(color=colors) * cycler(linestyle=["-"]) 
    if dataset_name != "Reshuffling":
            colorcycler = cycler(color=["black"])
            lines = ["-", ":"]
            if dataset_name == "TabRepo":
                lines = [":"]
            if "SMAC_NoInit" in policy_algorithms:
                lines = ["-", "--", ":"]
            linecycler = cycler(linestyle=lines)
            all_cyclers = all_cyclers.concat(colorcycler * linecycler)

    if "Oracle_Arm" in policy_algorithms:
        colorcycler = cycler(color=["grey"])
        lines = ["-"]
        linecycler = cycler(linestyle=lines)
        all_cyclers = all_cyclers.concat(colorcycler * linecycler)

    data_info["cyclers"] = all_cyclers
    data_info["name"] = name

    data_info["plot_type"] = "Ranking"
    data_info["ylabel"] = "Ranking"
    fig1 = plt_utils.plot_averaged_on_datasets(data_info)
    wandb.log({"Ranking " + name: wandb.Image(fig1)})

    data_info["plot_type"] = "Normalized loss"
    data_info["ylabel"] = "Normalized loss"
    fig_2 = plt_utils.plot_averaged_on_datasets(data_info)
    wandb.log({"Regret " + name: wandb.Image(fig_2)})

    fig_3 = plt_utils.plt_heatmap(data_info)
    wandb.log({"Heatmap " + name: wandb.Image(fig_3)})
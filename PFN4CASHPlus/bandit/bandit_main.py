import os
import numpy as np 
import pandas as pd
import pickle 
from functools import partial

from MaxUCB import MaxUCB
from Rising_Bandit import Rising_Bandit
import exp_utils

multiprocess = "joblib"
#multiprocess = " "
if(multiprocess == "joblib"):
     import joblib

dataset_name = "non_stationary_YaHPOGym_distilled"
dataset_name = "classification_TabRepoRaw_distilled"

dataset = pd.read_csv("../../../Bandits/datasets/" + dataset_name + ".csv")
instances = sorted(dataset["instance"].unique())
all_arm_index_list = dataset["arm_index"].unique()
valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
number_of_arms = len(valid_arm_index_list)
number_of_trails = len(dataset["repetition"].unique())
horizon_time = len(dataset["iteration"].unique())
classes = dataset["classifier"].unique()
combined_search_algorithms = dataset[dataset["arm_index"] < 0]["optimizer"].unique()


result_directory = "./results/" + dataset_name + "/"
###########################
policy_algorithms = {}
policy_algorithms["MaxUCB"] = MaxUCB
policy_algorithms["Rising_Bandit"] = Rising_Bandit
#policy_algorithms["PFN_PS"] = PFN_PS
policy_algorithms["Oracle_Arm"] =  None


df = dataset[(dataset["arm_index"]>=0)]
df = df[["instance", "arm_index", "repetition", "iteration", "loss"]]
data = df.sort_values(by=["instance", "arm_index", "repetition", "iteration"])[
    "loss"
].values.reshape(len(instances), number_of_arms, number_of_trails, horizon_time)

for alg_name,alg in policy_algorithms.items():
    if(alg_name=="Oracle_Arm"):
        run_expriment = exp_utils.run_fake_expriment
    else:
        run_expriment = exp_utils.run_expriment
    if not os.path.exists(result_directory + alg_name):
        if(multiprocess=='joblib'):
            print(alg_name)
            result, result_pulled_arms = zip(
                *joblib.Parallel(n_jobs=-1)(  #, backend="multiprocessing"
                    joblib.delayed(partial(run_expriment, alg))(
                        data[instance_num]
                    )
                    for instance_num in range(len(instances))
                )
            )
        else:
            result = []
            result_pulled_arms = []
            for instance_num in range(len(instances)):
                print(alg_name, instances[instance_num])
                res, res_pulled = run_expriment(alg, data[instance_num])
                result.append(res)
                result_pulled_arms.append(res_pulled)

        os.makedirs(result_directory + alg_name)
        with open(result_directory + alg_name + "/result.pkl", "wb") as file:
            pickle.dump(result,file )
        with open(result_directory + alg_name + "/pulled_arms.pkl", "wb") as file:
            pickle.dump(result_pulled_arms,file )
    else:
        print(alg_name +" does exist")

for alg_name in combined_search_algorithms:
    df = dataset[(dataset["arm_index"] < 0) & (dataset["optimizer"] == alg_name)]
    df = df[["instance", "arm_index", "repetition", "iteration", "loss"]]
    data = df.sort_values(by=["instance", "arm_index", "repetition", "iteration"])[
        "loss"
    ].values.reshape(len(instances), 1, number_of_trails, horizon_time)

    if not os.path.exists(result_directory + alg_name):
        if(multiprocess=='joblib'):
            result, _ = zip(
                *joblib.Parallel(n_jobs=-1, backend="multiprocessing")(
                    joblib.delayed(partial(exp_utils.run_fake_expriment, alg_name))(
                        data[instance_num]
                    )
                    for instance_num in range(len(instances))
                )
            )
        else:
            result = []
            for instance_num in range(len(instances)):
                res, _ = exp_utils.run_fake_expriment(alg_name, data[instance_num])
                result.append(res)

        os.makedirs(result_directory + alg_name)
        with open(result_directory + alg_name + "/result.pkl", "wb") as file:
            pickle.dump(result,file )
    else:
        print(alg_name +" does exist")

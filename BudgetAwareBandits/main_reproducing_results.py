import os
import numpy as np 
import pandas as pd
import pickle 
from functools import partial
import exp_utils
import sys
sys.path.append("../")

from Policies.MaxUCB import MaxUCB
from Policies.Rising_Bandit import Rising_Bandit
from Policies.Rising_Bandit_CA import Rising_Bandit_CA
from Policies.Random import Random
from Policies.UCB import UCB
from Policies.TS import TS

from Policies.PS_Max import PS_Max
from Policies.PS_Max_CA import PS_Max_CA
from Policies.PS_PFN_CA import PS_PFN_CA
from Policies.PS_PFNs_CA import PS_PFNs_CA
from Policies.PS_PFNs import PS_PFNs
from Policies.PS_PFN import PS_PFN


import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    nargs="?",
    default="",
    help="dataset name",
)

parser.add_argument(
    "--policy",
    nargs="?",
    default="all",
    help="dataset name",
)

args = parser.parse_args()

multiprocess = "joblib"
#multiprocess = " "
if(multiprocess == "joblib"):
     import joblib



dataset_name = "TabRepo"
dataset_name = "TabRepoRaw"
#dataset_name = "YaHPOGym"
#dataset_name = "Complex"
#dataset_name = "openml"
#dataset_name = "Grinsztajn"
#dataset_name = "TabRepoRaw_30"
#dataset_name = "YaHPOGym_100"
#dataset_name = "Complex"
#dataset_name = "more"
#dataset_name = "all"


dataset_name = dataset_name if args.dataset=="" else args.dataset

dataset = pd.read_csv("../Datasets/" + dataset_name + ".csv")

instances = sorted(dataset["instance"].unique())
print(instances, len(instances))

all_arm_index_list = dataset["arm_index"].unique()
valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
number_of_arms = len(valid_arm_index_list)
number_of_trails = len(dataset["repetition"].unique())
max_horizon_time = len(dataset["iteration"].unique())
combined_search_algorithms = dataset[dataset["arm_index"] < 0]["optimizer"].unique()

# if dataset_name == "more" or dataset_name == "Complex":   
#     number_of_trails = 2

###########################
policy_algorithms = {}
policy_algorithms["Oracle_Arm"] = "Oracle_Arm"
policy_algorithms["Random"] = Random
policy_algorithms["MaxUCB"] = MaxUCB
policy_algorithms["Rising_Bandit"] = Rising_Bandit
policy_algorithms["Rising_Bandit_CA"] = Rising_Bandit_CA
policy_algorithms["UCB"] = UCB
policy_algorithms["TS"] = TS


policy_algorithms["PS_Max"] = PS_Max
policy_algorithms["PS_Max_CA"] = PS_Max_CA


policy_algorithms["PS_Max_R"] = partial(
        PS_Max,
        max_target="remaining_steps",
    )

policy_algorithms["PS_Max_CA_R"] = partial(
        PS_Max_CA,
        max_target="remaining_steps",
    )



if (
    dataset_name == "openml"
    or dataset_name == "Grinsztajn"
    or dataset_name == "Complex"
    or dataset_name == "more"
    or dataset_name == "all"
):
    models_per_arm = {
        "XTab": "semi_flat",
        "FLAML": "semi_flat",
        "RealMLP": "flat",
        "TabForestPFN": "semi_flat",
        "TabPFN_v2": "semi_flat",
    }

if dataset_name == "TabRepoRaw_30":
    models_per_arm = {
        "CatBoost": "semi_flat",
        "ExtraTrees": "semi_flat",
        "LightGBM": "curved",
        "NeuralNet(FastAI)": "curved",
        "NeuralNet(Torch)": "curved",
        "RandomForest": "semi_flat",
        "XGBoost": "flat",
    }
if dataset_name == "YaHPOGym_100":
    models_per_arm = {
        "AKNN": "curved",
        "GLMNet": "semi_flat",
        "Ranger": "curved",
        "RPart": "semi_flat",
        "SVM": "curved",
        "XGBoost": "curved",
    }

policy_algorithms["PS_PFN"] = PS_PFN
policy_algorithms["PS_PFN_CA"] = PS_PFN_CA

policy_algorithms["PS_PFN_R"] = partial(
        PS_PFN,
        max_target="remaining_steps",
    )

policy_algorithms["PS_PFN_CA_R"] = partial(
        PS_PFN_CA,
        max_target="remaining_steps",
    )


policy_algorithms["PS_PFNs"] = partial(
    PS_PFNs,
    models_per_arm=list(models_per_arm.values()),
)

policy_algorithms["PS_PFNs_CA"] = partial(
    PS_PFNs_CA,
    models_per_arm=list(models_per_arm.values()),
)

policy_algorithms["PS_PFNs_R"] = partial(
    PS_PFNs,
    models_per_arm=list(models_per_arm.values()),
    max_target="remaining_steps",
)

policy_algorithms["PS_PFNs_CA_R"] = partial(
    PS_PFNs_CA,
    models_per_arm=list(models_per_arm.values()),
    max_target="remaining_steps",
)


policy_algorithms["PS_PFN_curve"] =  partial(
        PS_PFN,
        model_name="../Analysis/PFN/trained_models/curved.pt",
    )
    
policy_algorithms["PS_PFN_curve_CA_R"] = partial(
        PS_PFN_CA,
        model_name="../Analysis/PFN/trained_models/curved.pt",
        max_target="remaining_steps",
    )

policy_algorithms["PS_PFN_flat"] =  partial(
        PS_PFN,
        model_name="../Analysis/PFN/trained_models/flat.pt",
    )
policy_algorithms["PS_PFN_flat_CA_R"] = partial(
        PS_PFN_CA,
        model_name="../Analysis/PFN/trained_models/flat.pt",
        max_target="remaining_steps",
    )

###########################

if args.policy != "all":
    policy_algorithms = {args.policy: policy_algorithms[args.policy]}
else:
    del policy_algorithms["PS_PFN_CA_R"]
    del policy_algorithms["PS_PFNs_CA_R"]
    del policy_algorithms["PS_PFN"]
    del policy_algorithms["PS_PFNs"]
    del policy_algorithms["PS_PFNs_CA"]
    del policy_algorithms["PS_PFN_CA"]
    del policy_algorithms["PS_PFN_R"]
    del policy_algorithms["PS_PFNs_R"]
    del policy_algorithms["PS_Max_R"]
    del policy_algorithms["PS_Max_CA"]


df = dataset[(dataset["arm_index"]>=0)]
df = df[
    [
        "instance",
        "arm_index",
        "repetition",
        "iteration",
        "loss",
        "eval_time",
        "test_loss",
    ]
]

# Calculate the average of eval_time sums per instance
if dataset_name == "Complex" or dataset_name == "openml":
    # eval_time_sums = (
    #     df.groupby(["instance", "arm_index", "repetition"])["eval_time"]
    #     .sum()
    #     .reset_index()
    # )
    # eval_time_avg_per_instance = (
    #     eval_time_sums.groupby("instance")
    #     .apply(lambda group: group.groupby("repetition")["eval_time"].mean().min())
    #     .to_dict()
    # )
    #print(eval_time_avg_per_instance)
    eval_time_avg_per_instance = {instance: 2 * 3600 for instance in instances}
elif dataset_name == "Grinsztajn":
    eval_time_avg_per_instance = {instance: 2 * 3600 for instance in instances}
else:
    eval_time_sums = df.groupby(["instance", "arm_index", "repetition"])["eval_time"].sum().reset_index()
    eval_time_avg_per_instance = eval_time_sums.groupby("instance").apply(
        lambda group: group.groupby("repetition")["eval_time"].mean().min()
    ).to_dict()


# Function to get a single element based on instance, arm_index, and iteration
def get_single_element(instance, arm_index, iteration):
    return df[
        (df["instance"] == instance) &
        (df["arm_index"] == arm_index) &
        (df["iteration"] == iteration)
    ]
data_info = {}
data_info["number_of_arms"] = number_of_arms
data_info["instances"] = instances
data_info["number_of_trails"] = number_of_trails
data_info["max_horizon_time"] = max_horizon_time
data_info["combined_search_algorithms"] = combined_search_algorithms
data_info["max_budget"] = eval_time_avg_per_instance


#data_info["number_of_trails"] = 4
result_directory = (
    "./results_" + str(data_info["number_of_trails"]) + "/" + dataset_name + "/"
)


for alg_name,alg in policy_algorithms.items():
    if(alg_name=="Oracle_Arm"):
        run_expriment = exp_utils.run_fake_expriment
    else:
        run_expriment = exp_utils.run_expriment
    if not os.path.exists(result_directory + alg_name):
        if(multiprocess=='joblib'):
            print(alg_name)
            result, result_test, result_pulled_arms = zip(
                *joblib.Parallel(n_jobs=-1)(  # , backend="multiprocessing"
                    joblib.delayed(
                        partial(run_expriment, alg=alg, data_info=data_info)
                    )(
                        data = df[(df["instance"] == instance)],
                        max_budget=eval_time_avg_per_instance[instance],
                    )
                    for instance in instances
                )
            )
        else:
            result = []
            result_test = []
            result_pulled_arms = []
            for instance in instances:
                print(alg_name, instance)
                res, res_test, res_pulled = run_expriment(
                    alg,
                    data_info=data_info,
                    data=df[(df["instance"] == instance)],
                    max_budget=eval_time_avg_per_instance[instance],
                )
                result.append(res)
                result_test.append(res_test)
                result_pulled_arms.append(res_pulled)

        os.makedirs(result_directory + alg_name)
        with open(result_directory + alg_name + "/result.pkl", "wb") as file:
            pickle.dump(result,file )
        with open(result_directory + alg_name + "/result_test.pkl", "wb") as file:
            pickle.dump(result_test,file )
        with open(result_directory + alg_name + "/pulled_arms.pkl", "wb") as file:
            pickle.dump(result_pulled_arms,file )
    else:
        print(alg_name +" does exist")



for alg_name in combined_search_algorithms:
    df = dataset[(dataset["arm_index"] < 0) & (dataset["optimizer"] == alg_name)]
    df = df[
        [
            "instance",
            "arm_index",
            "repetition",
            "iteration",
            "loss",
            "eval_time",
            "test_loss",
        ]
    ]
    if not os.path.exists(result_directory + alg_name):
        if(multiprocess=='joblib'):
            print(alg_name)
            result, result_test, _ = zip(
                *joblib.Parallel(n_jobs=-1, backend="multiprocessing")(
                    joblib.delayed(
                        partial(
                            exp_utils.run_fake_expriment,
                            alg=alg_name,
                            data_info=data_info,
                        )
                    )(
                        data=df[(df["instance"] == instance)],
                        max_budget=eval_time_avg_per_instance[instance],
                    )
                    for instance in instances
                )
            )
        else:
            result = []
            result_test = []
            for instance in instances:
                print(alg_name, instance)
                res, res_test, res_pulled = exp_utils.run_fake_expriment(
                    alg_name,
                    data_info=data_info,
                    data=df[(df["instance"] == instance)],
                    max_budget=eval_time_avg_per_instance[instance],
                )
                result.append(res)
                result_test.append(res_test)

        os.makedirs(result_directory + alg_name)
        with open(result_directory + alg_name + "/result.pkl", "wb") as file:
            pickle.dump(result,file )
        with open(result_directory + alg_name + "/result_test.pkl", "wb") as file:
            pickle.dump(result_test, file)
    else:
        print(alg_name +" does exist")

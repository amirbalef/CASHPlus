import os
import numpy as np
import pandas as pd
import pickle
from experiment.dataset import get_data

dataset_name = "Complex"

res_dir = "./results/"
root_dir = res_dir + dataset_name + "/"

if dataset_name == "Complex" or dataset_name == "more":
    piplines = [
        "xtab.finetuning",
        "flaml.automl",
        "realmlp.smac",
        "tabforestpfn.finetuning",
        "tabpfn_v2_phe.phe",
    ]
    number_of_arms = len(piplines)
    number_of_trails = 4 # each trail has 3 folds,  together 12 repitions
    root_dir = res_dir + dataset_name + "/"


instances = [
    instance
    for instance in os.listdir(root_dir)
    if (os.path.isdir(root_dir + instance) and instance[0] != "_")
]
print(instances)

results_list = []
errors = []
for instace in instances:
    dataset_info = get_data(dataset_name, int(instace))
    print(dataset_info["task_type"] )
    if dataset_info["task_type"] == "multiclass":
        metric = "logloss"
    elif dataset_info["task_type"] == "binary":
        metric = "1-auc_roc"
    else:
        metric = "rmse"
    try:
        instace_dir = root_dir + instace + "/"
        arm_index_method_list = [
            (
                int(arm_index),
                piplines[arm_index].split(".")[1],
                piplines[arm_index].split(".")[0],
                piplines[arm_index],
            )
            for arm_index in range(number_of_arms)
        ]

        for arm_index, optimizer, classifier_name, optimizer_method in arm_index_method_list:
            print(instace, arm_index, optimizer, classifier_name, optimizer_method)
            for r in range(3):# each trail has 3 folds
                for trial in range(number_of_trails):
                    result =[]
                    result_path = instace_dir + optimizer_method + "/" + str(trial) + "/result.pkl"
                    if os.path.exists(result_path) :
                        result = pd.read_pickle(result_path)
                    if(len(result)==0):
                        result = pd.read_pickle(
                            instace_dir + optimizer_method + "/" + str(trial) + "/history.pkl"
                        )
                        if("ptarl.finetuning"==optimizer_method):
                            for item in result:
                                item.config["d_layers"] ="" #"["+' '.join(str(x) for x in item.config["d_layers"]) +"]"
                        if "realmlp.smac" == optimizer_method:
                            for item in result:
                                item.config["Model:hidden_sizes"] ="["+' '.join(str(x) for x in item.config["Model:hidden_sizes"]) +"]"
                        result = result.df()
                        with open(result_path, 'wb') as f:
                            pickle.dump(result, f)
                    
                    try:
                        if r == 0:
                            losses = result["summary:val_loss0"].to_numpy()
                        if r == 1:
                            losses = result["summary:val_loss1"].to_numpy()
                        if r == 2:
                            losses = result["summary:test_loss0"].to_numpy()
  
                        time_duration_s =result["profile:cross-validate:time:duration"].to_numpy()
                        test_losses = result["summary:test_loss0"].to_numpy()
                        test_time_duration_s = result['profile:cross-validate:predictions:time:duration'].to_numpy() 

                        if len(result) < 10:
                            print(
                                "Warning!",
                                optimizer_method,
                                instace,
                                trial,
                                len(result),
                                np.sum(time_duration_s),
                            )

                    except Exception as e:
                        print("error", optimizer_method, instace, trial, arm_index)
                        print(result.columns)
                        print(e)
                        exit()
                    for iteration, (
                        loss,
                        time_duration,
                        test_loss,
                        test_time,
                    ) in enumerate(
                        zip(
                            losses,
                            time_duration_s,
                            test_losses,
                            test_time_duration_s,
                        )
                    ):
                        dict1 = {
                            "instance": instace,
                            "metric": metric,
                            "repetition": 3*trial+r,
                            "arm_index": arm_index,
                            "arm_name": classifier_name,
                            "optimizer": optimizer,
                            "iteration": iteration,
                            "loss": loss,
                            "eval_time": time_duration,
                            "test_loss": test_loss,
                            "inference_time": test_time,
                        }
                        results_list.append(dict1)

    except Exception as e:
        print(e)
        errors.append(instace)
        pass

print(errors)
df = pd.DataFrame(results_list)
df = df[~df["instance"].isin(errors)]
df.to_csv("../Datasets/" + dataset_name + ".csv", index=False)
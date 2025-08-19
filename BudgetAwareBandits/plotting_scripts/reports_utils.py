import plotting_utils
import pandas as pd
import copy
import analysis_utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.colors import LogNorm
import sys

sys.path.append("../../")
from Datasets import algorithms_data
from collections import defaultdict


def get_all_results(
    dataset_name,
    policy_algorithms,
    base_path="../../Datasets/",
    result_directory="../results/",
    test_results = False,
):
    policy_algorithms = copy.deepcopy(policy_algorithms)
    number_of_policies = len(policy_algorithms.keys())
    dataset = pd.read_csv(base_path + dataset_name + ".csv")

    instances = sorted(dataset["instance"].unique())
    all_arm_index_list = dataset["arm_index"].unique()
    valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
    number_of_arms = len(valid_arm_index_list)
    number_of_trails = len(dataset["repetition"].unique())
    horizon_time = 180
    combined_search_algorithms = list(
        dataset[dataset["arm_index"] < 0]["optimizer"].unique()
    )

    if len(combined_search_algorithms) > 0:
        combined_search_algorithms.append(combined_search_algorithms.pop(0))
        for algorithm in combined_search_algorithms:
            # if algorithm != "SMAC_NoInit":
            policy_algorithms[algorithm] = 1
    #policy_algorithms["Oracle_Arm"] = 1
    
    all_result = analysis_utils.fetch_results(
        policy_algorithms, result_directory, dataset_name, test_results
    )

    from cycler import cycler


    grouped = defaultdict(lambda: {"base": [], "ca": []})

    for elem in list(policy_algorithms.keys())[:number_of_policies]:
        if "_CA" in elem:
            base = elem.split("_CA")[0]  # Take everything before _CA
            grouped[base]["ca"].append(elem)
        else:
            base = elem
            grouped[base]["base"].append(elem)

    colors = plotting_utils.CB_color_cycle
    all_cyclers = None

    for i, (base, versions) in enumerate(grouped.items()):
        colorcycler = cycler(color=[colors[i]]) * cycler(linestyle=["-", "--", ":"][0:len(versions["base"])+len(versions["ca"])])
        if all_cyclers is None:
            all_cyclers = colorcycler
        else:
            all_cyclers = all_cyclers.concat(colorcycler)

    if (
        dataset_name == "YaHPOGym"
        or dataset_name == "YaHPOGym_100"
        or dataset_name == "TabRepoRaw"
        or dataset_name == "TabRepoRaw_30"
    ):
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


    data = {}
    data["all_result"] = all_result
    data["dataset"] = dataset
    data["horizon_time"] = horizon_time
    data["number_of_arms"] = number_of_arms
    data["instances"] = instances
    data["number_of_trails"] = number_of_trails
    data["saving_path"] = None
    data["dataset_name"] = dataset_name
    data["cyclers"] = all_cyclers
    data["test_data"] = -1
    data["fig_size"] = (10, 8)
    data["legend"] = "inside"
    data["tilte"] = False
    data["ylabel"] = None
    data["plot_confidence_type"] = "mean_std"
    data["max_budget"] = 12 * 3600

    data["set_ylim"] = None
    if data["dataset_name"] == "TabRepo":
        data["set_ylim"] = (1, 0.05)
    if data["dataset_name"] == "Reshuffling":
        data["set_ylim"] = (1, 0.02)
    if data["dataset_name"] == "TabRepoRaw" or data["dataset_name"] == "TabRepoRaw_30":
        data["set_ylim"] = (0.7, 0.25)
        data["max_budget"] =  3600

    if data["dataset_name"] == "YaHPOGym" or data["dataset_name"] == "YaHPOGym_100":
        data["set_ylim"] = (0.7, 0.1)
        data["max_budget"] = 3600
        
    if data["dataset_name"] == "Complex":
        data["set_ylim"] = (0.5, 0.25)
        data["max_budget"] = 2 * 3600
        data["alpha"] = 0.01

    if (
        data["dataset_name"] == "synth_ablation"
        or data["dataset_name"] == "synth_ablation_distilled"
    ):
        data["set_ylim"] = (1, 0.1)

    return data 


def plt_heatmap(data, normalized = False):
    res_per_instance = analysis_utils.get_normallized_error_per_instance_time(
        data["all_result"],
        data["number_of_arms"],
        data["instances"],
        data["number_of_trails"],
        data["horizon_time"],
    )

    df = pd.DataFrame()
    for j, key in enumerate(data["all_result"].keys()):
        k = algorithms_data.printing_name_dict[key]
        if(normalized):
            df[k] = [
                (
                    np.mean(res_per_instance[i][j])
                    - np.nanmin([np.min(z) for z in res_per_instance[i]])
                )
                / np.nanmin([np.mean(z) + 0.01 for z in res_per_instance[i]])
                for i, item in enumerate(data["instances"])
            ]
        else:
            df[k] = [
                np.mean(res_per_instance[i][j])
                for i, item in enumerate(data["instances"])
            ]
    df_rank = df.T 
    
    fig, ax = plt.subplots(
        figsize=(0.4*len(data["instances"]), 0.5 * len(data["all_result"].keys()))
    )
    title_size = 15

    cmap = matplotlib.colormaps.get_cmap("RdYlGn_r")
    cmap.set_bad("white")

    ax = sns.heatmap(
        df_rank,
        cmap=cmap,
        yticklabels=True,
        norm=LogNorm(),
        xticklabels=data["instances"],
    )

    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.tick_params(labelsize=10)

    #plt.xticks([])
    plt.xlabel(
        algorithms_data.printing_name_dict[data["dataset_name"]],
        fontdict={"size": title_size},
    )
    return fig


def plot_pulls(data, dataset_name, policy_algorithms, result_directory="../results/",sort = True):
    policy_algorithms = copy.deepcopy(policy_algorithms)

    all_pulls_result = analysis_utils.fetch_pullings(
        policy_algorithms, result_directory, dataset_name
    )

    df = data["dataset"][(data["dataset"]["arm_index"] >= 0)]
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

    eval_time_sums = df.groupby(["instance", "arm_index", "repetition"])["eval_time"].sum().reset_index()
    eval_time_avg_per_instance = eval_time_sums.groupby("instance").apply(
        lambda group: group.groupby("repetition")["eval_time"].mean().min()
    ).to_dict()

    pull_s = []
    for intance_num, intance in enumerate(data["instances"]):
        pull_list = []
        for trial in range(data["number_of_trails"]):
            arm_mins =[]
            for arm in range(data["number_of_arms"]):
                arm_data = df[
                    (df["repetition"] == trial)
                    & (df["arm_index"] == arm)
                    & (df["instance"] == intance)
                ]
                cumulative_eval_time = arm_data["eval_time"].cumsum()
                valid_indices = cumulative_eval_time[
                    cumulative_eval_time <= eval_time_avg_per_instance[intance]
                ].index
                arm_mins.append(arm_data.loc[valid_indices, "loss"].min())
            if sort:
                selected_arms = np.argsort(arm_mins)
            else:
                selected_arms = np.arange(data["number_of_arms"])

            list_pulls = []
            for key, alg_pulls in all_pulls_result.items():
                pulled_arms = np.bincount(
                    alg_pulls[intance_num][trial][: data["horizon_time"]],
                    minlength=data["number_of_arms"],
                )
                list_pulls.append(pulled_arms[selected_arms])

            pull_list.append(list_pulls)

        pull_s.append(pull_list)
    pulls = np.asarray(pull_s)


    dfs = []
    for alg_num, alg_name in enumerate(all_pulls_result.keys()):
        _data = {}
        for i in range(0, data["number_of_arms"]):
            _data[str(i)] = pulls[:, :, alg_num, i].flatten()
        df = pd.DataFrame(_data)
        df["Algorithm"] = algorithms_data.printing_name_dict[alg_name]
        dfs.append(df)

    combined_df = pd.concat(dfs)
    # Melt the combined DataFrame
    melted_df = pd.melt(
        combined_df, id_vars=["Algorithm"], var_name="arms", value_name="Value"
    )

    plt.rcParams["text.usetex"] = True
    # Fig size
    plt.rcParams["figure.figsize"] = 10, 6
    plt.rcParams.update({"font.size": 26})

    fig, ax = plt.subplots()
    my_pal = plotting_utils.CB_color_cycle[: len(all_pulls_result.keys())]

    g1 = sns.boxplot(
        x="arms",
        y="Value",
        hue="Algorithm",
        data=melted_df,
        linewidth=0.4,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2},
        showcaps=False,
        # linecolor="#137",
        palette=my_pal,
        gap=0.2,
        boxprops=dict(facecolor="white", alpha=0.3, edgecolor="black"),
    )
    g1.get_legend().remove()

    g2 = sns.barplot(
        x="arms",
        y="Value",
        hue="Algorithm",
        errorbar=None,
        data=melted_df,
        palette=my_pal,
        saturation=0.75,
        gap=0.0,
    )

    # plt.yscale('log')
    # axins.set_xlim(0.95, 1.05)
    plt.xticks(range(data["number_of_arms"]))
    plt.ylabel("Number of pulls")
    plt.xlabel("arms")

    # extract the existing handles and labels
    h, l = g2.get_legend_handles_labels()

    ax.legend(
        h[len(all_pulls_result.keys()) :],
        l[len(all_pulls_result.keys()) :],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    return fig



def number_of_pulls(
    data,
    dataset_name,
    policy_algorithms,
    result_directory="../results/",
):
    policy_algorithms = copy.deepcopy(policy_algorithms)

    all_pulls_result = analysis_utils.fetch_pullings(
        policy_algorithms, result_directory, dataset_name
    )


    return all_pulls_result


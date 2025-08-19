from . import analysis_utils
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as plticker
import seaborn as sns



def plt_heatmap(data_info):
    res_per_instance = analysis_utils.get_normallized_error_per_instance_time(
        data_info["all_result"],
        data_info["number_of_arms"],
        data_info["instances"],
        data_info["number_of_trails"],
        data_info["horizon_time"],
    )
    df = pd.DataFrame()
    for j, key in enumerate(data_info["all_result"].keys()):
            df[key] = [
                np.mean(res_per_instance[i][j])
                for i, item in enumerate(data_info["instances"])
            ]
    df_rank = df.T 
    #df_rank = df_rank.reindex(columns=indexes)

    fig, ax = plt.subplots(figsize=(16, 0.5 * len(data_info["all_result"].keys())))
    title_size = 10

    cmap = matplotlib.colormaps.get_cmap("RdYlGn_r")
    cmap.set_bad("white")

    ax = sns.heatmap(
        df_rank,
        cmap=cmap,
        yticklabels=True,
        norm=LogNorm(),
        xticklabels=data_info["instances"],
    )

    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.tick_params(labelsize=title_size)

    #plt.xticks([])
    plt.xlabel(
        data_info["dataset_name"],
        fontdict={"size": title_size},
    )
    return fig

def plot_averaged_on_datasets(data):
    linewidth = 3
    if data["plot_type"] == "Ranking":
        alpha = 0.3
        mean, std = analysis_utils.get_ranks_per_instance_MC(
            data["all_result"],
            data["horizon_time"],
            data["number_of_arms"],
            data["instances"],
            data["number_of_trails"],
        )
    if data["plot_type"] == "Normalized loss":
        alpha = 0.1
        res = analysis_utils.get_error_per_seed(
            data["all_result"],
            data["dataset"],
            data["number_of_arms"],
            data["instances"],
            data["number_of_trails"],
        )
        df = pd.concat(res, axis=1).sort_index().ffill()
        mean = (
            df.groupby(level=0, axis=1)
            .mean()[data["all_result"].keys()]
            .values.T[:, 1:]
        )
        std = (
            df.groupby(level=0, axis=1).std()[data["all_result"].keys()].values.T[:, 1:]
        )
    index = np.arange(data["horizon_time"])

    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_prop_cycle(data["cyclers"])

    for i, item in enumerate(data["all_result"].keys()):
        zorder = len(data["all_result"].keys()) - i
        ax.plot(
            index,
            mean[i],
            label=item,
            linewidth=linewidth,
            zorder=zorder,
        )  # , marker=i,markevery=10)
        ax.fill_between(index, mean[i] - std[i], mean[i] + std[i], alpha=alpha)

    ax.set(xlabel="Iteration", ylabel=data["ylabel"])
    if data["plot_type"] == "Normalized loss":
        ax.ticklabel_format(style="plain")
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(plticker.NullFormatter())
        ax.yaxis.set_ticks(ax.get_yticks())
        ax.set_yticklabels([str(x) for x in ax.get_yticks()])
        ax.set_ylim(bottom=0.05, top=1.0)

    ax.legend(
        fontsize=14,
    )
    plt.title(data["name"])

    return fig
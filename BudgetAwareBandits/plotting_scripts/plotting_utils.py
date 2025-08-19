import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import analysis_utils
import matplotlib.ticker as plticker
from matplotlib.colors import LogNorm
import pylab
import sys
sys.path.append("../../")
from Datasets import algorithms_data


plt.rcParams["text.usetex"] = True

# ploting
colors_6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
colors_8 = ["#1845fb", "#ff5e02", "#c91f16", "#c849a9", "#adad7d", "#86c8dd", "#578dff", "#656364"]
colors_10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
CB_color_cycle = ['#377eb8', '#ff7f00', '#f781bf' ,
                  '#4daf4a', '#a65628', '#984ea3',
                  '#e41a1c', '#dede00','#999999']

colorblind_friendly = ["#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7","#E69F00"] #A set of 8 colorblind-friendly colors from Bang Wong’s Nature Methods paper https://www.nature.com/articles/nmeth.1618.pdf

def plot_averaged_on_datasets(data):
    linewidth = 3
    alpha = 0.1
    if "linewidth" in data:
        linewidth = data["linewidth"]

    if data["plot_type"] == "Ranking":
        alpha = 0.3
        mean, std = analysis_utils.get_ranks_per_instance_MC(
            data["all_result"],
            data["horizon_time"],
            data["number_of_arms"],
            data["instances"],
            data["number_of_trails"],
            num_samples=4*data["number_of_trails"],
        )
    if data["plot_type"] == "Normalized loss":
        if "alpha" in data:
            alpha = data["alpha"]
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
    # setting font sizeto 30
    plt.rcParams.update({"font.size": 26})
    if "fig_size" in data.keys():
        fig, ax = plt.subplots(figsize=data["fig_size"])
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_prop_cycle(data["cyclers"])

    for i, item in enumerate(data["all_result"].keys()):
        zorder = len(data["all_result"].keys()) - i
        ax.plot(
            index,
            mean[i],
            label=algorithms_data.printing_name_dict[item],
            linewidth=linewidth,
            zorder=zorder,
        )  # , marker=i,markevery=10)
        ax.fill_between(index, mean[i] - std[i], mean[i] + std[i], alpha=alpha)

    if data["dataset_name"] == "Complex" or data["dataset_name"] == "Grinsztajn":
        ax.set(xlabel="Wall-clock time (h)", ylabel=data["ylabel"])
    else:
        ax.set(xlabel="Normalized wall-clock time", ylabel=data["ylabel"])

    if data["plot_type"] == "Normalized loss":
        ax.ticklabel_format(style="plain")
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(plticker.NullFormatter())
        ax.yaxis.set_ticks(ax.get_yticks())
        ax.set_yticklabels([str(x) for x in ax.get_yticks()])
        ax.set_ylim(bottom=data["set_ylim"][1], top=data["set_ylim"][0])
    if data["max_budget"]== 12*3600:
        ax.set_xticks(
            [x for x in range(0, data["horizon_time"], data["horizon_time"] // 6)]
        )
    if data["max_budget"] == 8 * 3600:
        ax.set_xticks(
            [x for x in range(0, data["horizon_time"] + 1, data["horizon_time"] // 4)]
        )
    if data["max_budget"] == 2 * 3600:
        ax.set_xticks(
            [x for x in range(0, data["horizon_time"] + 1, data["horizon_time"] // 2)]
        )
        minor_ticks = [
            x for x in range(0, data["horizon_time"] + 1, data["horizon_time"] // 4)
        ]
        ax.set_xticks(minor_ticks, minor=True)
    if data["max_budget"] == 3600:
        ax.set_xticks([0, data["horizon_time"]])
        minor_ticks = [
            x for x in range(0, data["horizon_time"] + 1, data["horizon_time"] // 4)
        ]
        ax.set_xticks(minor_ticks, minor=True)

    ax.set_xticklabels(
        [
            str(data["max_budget"] / 3600 * (x / data["horizon_time"]))
            for x in ax.get_xticks()
        ]
    )

    if data["legend"] == "inside": 
        loc = "center right"
        if "fig_size" in data:
            bbox_to_anchor = (0.65 * 8 / data["fig_size"][0], 0.5, 1, 0.1)
        else:
            bbox_to_anchor = (0.65, 0.5, 1, 0.1)
        ax.legend(
            loc=loc,
            ncol=1,
            fontsize=22,
            bbox_to_anchor=bbox_to_anchor,
            handletextpad=0.15,
            handlelength=1.5,
            frameon=False,
        )
    if data["tilte"]:
        plt.title(algorithms_data.printing_name_dict[data["dataset_name"]])
    # plt.savefig(data["saving_path"] +"/"+data["dataset_name"]+"_" + data["saving_name"] +".png", dpi=600, bbox_inches='tight')
    # plt.tight_layout()
    plt.grid(True, which='both', axis="x", zorder=0)
    if data["saving_path"] == None:
        return fig

    if data["saving_path"] == "legend":
        figlegend = pylab.figure(figsize=(3, 8))
        pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left")
        return figlegend
    else:
        plt.savefig(
            data["saving_path"]
            + "/"
            + data["dataset_name"]
            + "_"
            + data["saving_name"]
            + ".pdf",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
        if data["legend"] == "seperate":
            figlegend = pylab.figure(figsize=(3, 8))
            pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left")
            figlegend.savefig(
                data["saving_path"] + "/" + data["saving_name"] + "_legend.pdf",
                dpi=600,
                bbox_inches="tight",
            )


def plot_heatmap(data):
    cmap = "RdYlGn_r"
    if "cmap" in data:
        cmap = data["cmap"]
    if data["plot_type"] == "Ranking":
        m_res = analysis_utils.get_ranks(
            data["all_result"],
            data["horizon_time"],
            data["number_of_arms"],
            data["instances"],
            data["number_of_trails"],
        )

    elif data["plot_type"] == "Normalized loss":
        res = analysis_utils.get_error_per_seed(
            data["all_result"],
            data["dataset"],
            data["number_of_arms"],
            data["instances"],
            data["number_of_trails"],
        )
        df = pd.concat(res, axis=1).sort_index().ffill()
        if data["plot_confidence_type"] == "mean_std":
            m_res = df.groupby(level=0, axis=1).mean()
            if("clip_upper" in data):
                m_res = m_res.clip(upper=np.nanmax(m_res.to_numpy()) * data["clip_upper"]) 

        if data["plot_confidence_type"] == "median_quantile":
            m_res = df.groupby(level=0, axis=1).median()
    # setting font sizeto 30
    plt.rcParams.update({"font.size": 26})
    fig, ax = plt.subplots(figsize=(10, 8))
    if data["plot_type"] == "Normalized Error":
        sns.heatmap(
            m_res.T[: len(data["third_axis"])],
            linewidths=0,
            cmap=cmap,
            norm=LogNorm(),
            rasterized=True,
        )
    else:
        sns.heatmap(
            m_res.T[: len(data["third_axis"])],
            linewidths=0,
            cmap=cmap,
            rasterized=True,
        )
    ax.set_yticks(
        np.arange(len(data["third_axis"])), labels=data["third_axis"], rotation=0
    )
    loc = plticker.MultipleLocator(
        base=2.0
    )  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)

    ax.set_xticks(
        np.arange(m_res.shape[0]), labels=np.arange(m_res.shape[0]), rotation=0
    )
    loc = plticker.MultipleLocator(
        base=data["horizon_time"] / 10.0
    )  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.set(xlabel="Iteration", ylabel="$\\alpha$")

    if data["legend"] == "inside": 
        loc = "center right"
        if "fig_size" in data:
            bbox_to_anchor = (0.65 * 8 / data["fig_size"][0], 0.5, 1, 0.1)
        else:
            bbox_to_anchor = (0.65, 0.5, 1, 0.1)
        ax.legend(
            loc=loc,
            ncol=1,
            fontsize=22,
            bbox_to_anchor=bbox_to_anchor,
            handletextpad=0.15,
            handlelength=1.5,
            frameon=False,
        )

    plt.title(
        data["plot_type"]
        + " - "
        + algorithms_data.printing_name_dict[data["dataset_name"]]
    )
    if data["saving_path"] ==None:
        return plt 
    else:
        plt.savefig(
            data["saving_path"]
            + "/"
            + data["dataset_name"]
            + "_"
            + data["saving_name"]
            + "_heatmap.pdf",
            dpi=600,
            bbox_inches="tight",
        )

        plt.close()
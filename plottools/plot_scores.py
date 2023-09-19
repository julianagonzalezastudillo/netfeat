import os
import json
import numpy as np
from scipy.stats import t
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (
    collapse_session_scores,
    combine_effects,
    combine_pvalues
)


path = os.getcwd()
jsonfile = "params.json"
jsonfullpath = os.path.join(path, jsonfile)
with open(jsonfullpath) as jsonfile:
    parameterDict = json.load(jsonfile)
PALETTE = parameterDict['palette']


def score_plot(data, pipelines=None):
    """Plot scores for all pipelines and all datasets

    Parameters
    ----------
    data: output of Results.to_dataframe()
        results on datasets
    pipelines: list of str | None
        pipelines to include in this plot

    Returns
    -------
    fig: Figure
        Pyplot handle
    color_dict: dict
        Dictionary with the facecolor
    """
    data = collapse_session_scores(data)
    data["dataset"] = data["dataset"].apply(moabb_plt._simplify_names)
    if pipelines is not None:
        data = data[data.pipeline.isin(pipelines)]
    fig = plt.figure(figsize=(8, 9), dpi=300)
    ax = fig.add_subplot(111)

    # marker
    marker_path, attributes = svg2paths('plottools/bonhomme.svg')
    my_marker = parse_path(attributes[0]['d'])

    _, idx = np.unique(data['pipeline'], return_index=True)
    pipe_list = data['pipeline'].iloc[np.sort(idx)].to_list()

    data['pipeline'] = pd.Categorical(data['pipeline'],
                                      categories=pipe_list,
                                      ordered=True)
    colors = []
    for pipe in pipe_list:
        colors.append(PALETTE[pipe])

    sns.stripplot(
        data=data,
        y="dataset",
        x="score",
        jitter=0.1,
        palette=colors,
        hue="pipeline",
        dodge=True,
        ax=ax,
        alpha=0.4,
        marker=my_marker,
        s=22,
    )
    sns.stripplot(
        y=data.groupby(['dataset', 'pipeline'], as_index=False)['score'].mean()['dataset'],
        x=data.groupby(['dataset', 'pipeline'], as_index=False)['score'].mean()['score'],
        jitter=0.15,
        palette=colors,
        hue=data.groupby(['dataset', 'pipeline'], as_index=False)['score'].mean()['pipeline'],
        dodge=True,
        ax=ax,
        marker=my_marker,
        s=35,
        edgecolors='k',
        linewidth=1,
    )
    text_dt = data.groupby(['dataset', 'pipeline'], as_index=False)['score'].mean()['dataset']
    text_pipe = data.groupby(['dataset', 'pipeline'], as_index=False)['score'].mean()['pipeline']
    text_score = data.groupby(['dataset', 'pipeline'], as_index=False)['score'].mean()['score']
    text_sd = data.groupby(['dataset', 'pipeline'], as_index=False)['score'].std()['score']
    # print(text_dt)
    # print(text_pipe)
    # print('mean')
    # for i in np.arange(0, len(text_score)):
    #     print("%.2f" % (text_score[i]*100))
    # print('std')
    # for i in np.arange(0, len(text_sd)):
    #     print("%.2f" % (text_sd[i]*100))
    ax.set_xlim([0, 1])
    ax.axvline(0.5, linestyle="--", color="k", linewidth=2)
    ax.axvline(0.7, linestyle="--", color="grey", linewidth=2)
    ax.set_title("Scores per dataset and algorithm")
    handles, labels = ax.get_legend_handles_labels()
    color_dict = {lb: h.get_facecolor()[0] for lb, h in zip(labels, handles)}
    # ax.legend(labels[0:int(len(labels)/2)])
    end = None
    ax.legend(handles[len(np.unique(data.pipeline)):end],
              labels[len(np.unique(data.pipeline)):end])
    # ax.get_legend().remove()
    plt.tight_layout()
    return fig, ax


def _simplify_names(x):
    if len(x) > 10:
        return x.split(" ")[0]
    else:
        return x

def meta_analysis_plot(stats_df, alg1, alg2, alg1_new_name, alg2_new_name):  # noqa: C901
    """Meta-analysis to compare two algorithms across several datasets

    A meta-analysis style plot that shows the standardized effect with
    confidence intervals over all datasets for two algorithms.
    Hypothesis is that alg1 is larger than alg2

    Parameters
    ----------
    stats_df: DataFrame
        DataFrame generated by compute_dataset_statistics
    alg1: str
        Name of first pipeline
    alg2: str
        Name of second pipeline

    Returns
    -------
    fig: Figure
        Pyplot handle
    """

    def _marker(pval):
        if pval < 0.001:
            return "$***$", 100
        elif pval < 0.01:
            return "$**$", 70
        elif pval < 0.05:
            return "$*$", 30
        else:
            raise ValueError("insignificant pval {}".format(pval))

    assert alg1 in stats_df.pipe1.unique()
    assert alg2 in stats_df.pipe1.unique()
    df_fw = stats_df.loc[(stats_df.pipe1 == alg1) & (stats_df.pipe2 == alg2)]
    df_fw = df_fw.sort_values(by="pipe1")
    df_bk = stats_df.loc[(stats_df.pipe1 == alg2) & (stats_df.pipe2 == alg1)]
    df_bk = df_bk.sort_values(by="pipe1")
    dsets = df_fw.dataset.unique()
    ci = []
    fig = plt.figure(dpi=300)
    gs = gridspec.GridSpec(1, 5)
    sig_ind = []
    pvals = []
    ax = fig.add_subplot(gs[0, :-1])
    ax.set_yticks(np.arange(len(dsets) + 1))
    ax.set_yticklabels(["Meta-effect"] + [_simplify_names(d) for d in dsets])
    pval_ax = fig.add_subplot(gs[0, -1], sharey=ax)
    plt.setp(pval_ax.get_yticklabels(), visible=False)
    _min = 0
    _max = 0
    for ind, d in enumerate(dsets):
        nsub = float(df_fw.loc[df_fw.dataset == d, "nsub"])
        t_dof = nsub - 1
        ci.append(t.ppf(0.95, t_dof) / np.sqrt(nsub))
        v = float(df_fw.loc[df_fw.dataset == d, "smd"])
        if v > 0:
            p = df_fw.loc[df_fw.dataset == d, "p"].item()
            if p < 0.05:
                sig_ind.append(ind)
                pvals.append(p)
        else:
            p = df_bk.loc[df_bk.dataset == d, "p"].item()
            if p < 0.05:
                sig_ind.append(ind)
                pvals.append(p)
        _min = _min if (_min < (v - ci[-1])) else (v - ci[-1])
        _max = _max if (_max > (v + ci[-1])) else (v + ci[-1])
        ax.plot(
            np.array([v - ci[-1], v + ci[-1]]), np.ones((2,)) * (ind + 1), c="tab:grey"
        )
    _range = max(abs(_min), abs(_max))
    ax.set_xlim((0 - _range, 0 + _range))
    final_effect = combine_effects(df_fw["smd"], df_fw["nsub"])
    ax.scatter(
        pd.concat([pd.Series([final_effect]), df_fw["smd"]]),
        np.arange(len(dsets) + 1),
        s=np.array([50] + [30] * len(dsets)),
        marker="D",
        c=["k"] + ["tab:grey"] * len(dsets),
    )
    for i, p in zip(sig_ind, pvals):
        m, s = _marker(p)
        ax.scatter(df_fw["smd"].iloc[i], i + 1.4, s=s, marker=m, color="r")
    # pvalues axis stuf
    pval_ax.set_xlim([-0.1, 0.1])
    pval_ax.grid(False)
    pval_ax.set_title("p-value", fontdict={"fontsize": 10})
    pval_ax.set_xticks([])
    for spine in pval_ax.spines.values():
        spine.set_visible(False)
    for ind, p in zip(sig_ind, pvals):
        pval_ax.text(
            0,
            ind + 1,
            horizontalalignment="center",
            verticalalignment="center",
            s="{:.2e}".format(p),
            fontsize=8,
        )
    if final_effect > 0:
        p = combine_pvalues(df_fw["p"], df_fw["nsub"])
        if p < 0.05:
            m, s = _marker(p)
            ax.scatter([final_effect], [-0.4], s=s, marker=m, c="r")
            pval_ax.text(
                0,
                0,
                horizontalalignment="center",
                verticalalignment="center",
                s="{:.2e}".format(p),
                fontsize=8,
            )
    else:
        p = combine_pvalues(df_bk["p"], df_bk["nsub"])
        if p < 0.05:
            m, s = _marker(p)
            ax.scatter([final_effect], [-0.4], s=s, marker=m, c="r")
            pval_ax.text(
                0,
                0,
                horizontalalignment="center",
                verticalalignment="center",
                s="{:.2e}".format(p),
                fontsize=8,
            )

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, linestyle="--", c="k")
    ax.axhline(0.5, linestyle="-", linewidth=3, c="k")

    title = "< {} better{}\n{}{} better >".format(
        alg2_new_name, " " * (45 - len(alg2_new_name)), " " * (45 - len(alg1_new_name)), alg1_new_name
    )
    ax.set_title(title, ha="left", ma="right", loc="left")
    ax.set_xlabel("Standardized Mean Difference")
    fig.tight_layout()

    return fig,ax

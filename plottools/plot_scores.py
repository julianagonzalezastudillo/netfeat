import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (
    collapse_session_scores
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
    print(text_dt)
    print(text_pipe)
    print('mean')
    for i in np.arange(0, len(text_score)):
        print("%.2f" % (text_score[i]*100))
    print('std')
    for i in np.arange(0, len(text_sd)):
        print("%.2f" % (text_sd[i]*100))
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

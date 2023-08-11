""" This file is responsible for defining globals and initializing them """
import pandas as pd


def init():
    global n_cv
    n_cv = 1

    column_names = ["dataset", "subject", "session", "pipeline", "metric", "cv", "ch", "t-val"]
    global df_select
    df_select = pd.DataFrame(columns=column_names)

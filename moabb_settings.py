""" This file contains auxiliary functions """

import gzip
import pandas as pd
import pickle
import save_features
from config import ConfigPath


def add_attributes(obj, **kwargs):
    """
    Adds attributes to an object dynamically. Created to add necessary attributes to pipelines.
    """
    for key, value in kwargs.items():
        setattr(obj, key, value)


def save_global(params):
    if params.t_val:
        params.t_val = [round(val, 2) for val in params.t_val]
    selec = {
        "dataset": params.dataset.code,
        "subject": params.dataset.sub,
        "session": params.session,
        "pipeline": params.pipeline,
        "metric": params.metric,
        "cv": save_features.n_cv,  # n_cv,
        "ch": params.subelec_names,
        "t-val": params.t_val,
    }
    # save_features.df_select = save_features.df_select.append(selec, ignore_index=True)
    # Convert selec to a DataFrame
    selec_df = pd.DataFrame([selec])

    # Concatenate the DataFrame
    save_features.df_select = pd.concat(
        [save_features.df_select, selec_df], ignore_index=True
    )

    if (
        int(params.dataset.sub) == params.dataset.subject_list[-1]
        and save_features.n_cv == params.cv_splits
        and params.session == params.sessions_name[-1]
    ):
        df_save = save_features.df_select[
            (save_features.df_select["dataset"] == params.dataset.code)
            & (save_features.df_select["pipeline"] == params.pipeline)
        ]
        df_save.to_csv(
            ConfigPath.RES_DIR
            / f"select_features/select_features_{params.dataset.code}_{params.pipeline}.csv",
            index=False,
        )
    save_features.n_cv = save_features.n_cv + 1


def create_info_file(dataset, subject, paradigm):
    """
    Create info file with subject information in order to fasten when loading information.
    """
    info_path = ConfigPath.RES_DIR / "info"
    info_path.mkdir(parents=True, exist_ok=True)

    info_file = info_path / f"info_{dataset.code}_sub{str(subject).zfill(3)}.gz"

    if info_file.exists():
        with gzip.open(info_file, "r") as f:
            info = pickle.load(f)
    else:
        X, y, metadata = paradigm.get_data(
            dataset=dataset, subjects=[subject], return_epochs=True
        )
        info = {
            "sub": subject,
            "dataset": dataset.code,
            "ch_names": X.ch_names,
            "classes": y,
            "session": metadata.session,
            "event_id": X.event_id,
        }
        with gzip.open(info_file, "w") as f:  # save
            pickle.dump(info, f)
    return info

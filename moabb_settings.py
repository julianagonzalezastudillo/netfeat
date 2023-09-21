""" This file contains auxiliary functions """
import save_features
from pathlib import Path
import pickle
import gzip
import os


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
    save_features.df_select = save_features.df_select.append(selec, ignore_index=True)

    if params.dataset.sub == params.dataset.subject_list[-1] \
            and save_features.n_cv == params.cv_splits \
            and params.session == params.sessions_name[-1]:
        df_save = save_features.df_select[(save_features.df_select["dataset"] == params.dataset.code) &
                                          (save_features.df_select["pipeline"] == params.pipeline)]
        df_save.to_csv("./results/select_features/select_features_{0}_{1}.csv"
                       .format(params.dataset.code, params.pipeline), index=False)
    save_features.n_cv = save_features.n_cv + 1


def create_info_file(dataset, subject, paradigm):
    """
    Create info file with subject information in order to fasten when loading information.
    """
    info_path = Path(os.path.join(os.getcwd(), 'results', 'info'))
    if not os.path.exists(info_path):
        os.makedirs(info_path)

    info_file = Path(os.path.join(info_path, 'info_{0}_sub{1}.gz'.format(dataset.code, str(subject).zfill(3))))
    if os.path.exists(info_file):
        with gzip.open(info_file, "r") as f:
            info = pickle.load(f)
    else:
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
        info = {'sub': subject,
                'dataset': dataset.code,
                'ch_names': X.ch_names,
                'classes': y,
                'session': metadata.session,
                'event_id': X.event_id
                }
        with gzip.open(info_file, "w") as f:  # save
            pickle.dump(info, f)
    return info

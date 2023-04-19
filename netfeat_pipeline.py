"""
==========================================
 Network Metrics & Functional Connectivity
==========================================
This module is design to compute network metrics based on functional connectivity for MOABB datasets
"""

import numpy as np
import pandas as pd
from time import time
from tools import bcolors, suppress_stdout

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)

from sklearn.preprocessing import LabelEncoder

from moabb.evaluations.base import BaseEvaluation
from moabb_settings import add_attributes
import save_features
save_features.init()


class WithinSessionEvaluation_netfeat(BaseEvaluation):
    def is_valid(self, dataset):
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, dataset, pipelines):
        print('=' * 100)
        print(bcolors.HEADER + "dataset: {0}".format(type(dataset).__name__) + bcolors.ENDC)

        dataset_res = list()
        for subject in dataset.subject_list:
            run_pipes = self.results.not_yet_computed(pipelines, dataset, subject)
            if len(run_pipes) == 0:
                continue

            print('-' * 100)
            print(bcolors.HEADER + "sub: {0}".format(subject) + bcolors.ENDC)

            X, y, metadata = self.paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
            dataset.sessions = np.unique(metadata.session)

            for session in dataset.sessions:
                ix = metadata.session == session
                X_session = np.array(X[ix])
                le = LabelEncoder()  # to change string labels to numerical ones
                y_session = le.fit_transform(y[ix])
                nchan = np.array(X).shape[1]

                for name, clf in pipelines.items():
                    t_start = time()
                    cv = StratifiedKFold(5, shuffle=False, random_state=None)
                    [add_attributes(clf[clf.steps[i][0]], subject=subject, dataset=dataset, session=session,
                                    pipeline=name, ch_names=X.ch_names, cv_splits=cv.n_splits) for i in range(len(clf.steps))]

                    with suppress_stdout():
                        acc = cross_val_score(clf,
                                              X_session,
                                              y_session,
                                              cv=cv,
                                              scoring=self.paradigm.scoring,
                                              n_jobs=self.n_jobs,
                                              error_score = self.error_score
                                              )
                    score = acc.mean()
                    score_std = acc.std()
                    duration = time() - t_start
                    res = {
                        "dataset": dataset.code,
                        "subject": subject,
                        "session": session,
                        "pipeline": name,
                        "score": score,
                        "std": score_std,
                        "n_samples": len(y_session),  # not training sample
                        "n_channels": nchan,
                        "time": duration / 5.0,  # 5 fold CV
                    }
                    if "n_cv" in globals():
                        global n_cv
                        n_cv = 1
                    save_features.n_cv = 1
                    dataset_res.append(res)
        return pd.DataFrame(dataset_res)

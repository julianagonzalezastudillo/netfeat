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
import os
import pickle
import gzip
from copy import deepcopy

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)

from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

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

            for name, clf_preproc in pipelines.items():
                t_start = time()
                cv = StratifiedKFold(5, shuffle=False, random_state=None)
                [add_attributes(clf_preproc[clf_preproc.steps[i][0]], subject=subject, dataset=dataset, pipeline=name,
                                ch_names=X.ch_names, cv_splits=cv.n_splits) for i in range(len(clf_preproc.steps))]

                X, clf = self.fc_net(X, clf_preproc)

                for session in dataset.sessions:
                    ix = metadata.session == session
                    X_session = np.array(X[ix])
                    le = LabelEncoder()  # to change string labels to numerical ones
                    y_session = le.fit_transform(y[ix])
                    nchan = np.array(X).shape[1]
                    [add_attributes(clf[clf.steps[i][0]], session=session) for i in range(len(clf.steps))]

                    # with suppress_stdout():
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

    def fc_net(self, X, clf_preproc):
        """
        For optimisation functional connectivity and network metrics are computed and saved outside the
        cross-validation. If the files contained this data already exists, they are loaded.
        :return:
        X: nd.array
        clf: pipeline
        """
        clf = deepcopy(clf_preproc)
        content = [clf.steps[i][0] for i in range(len(clf.steps))]

        for estimator in ['functional_connectivity', 'net_metric']:
            if estimator in content:
                method = clf[estimator].method
                subject = clf.named_steps[estimator].subject
                dataset = clf.named_steps[estimator].dataset

                file_path = os.path.join(os.getcwd(), "results/{0}/".format(estimator))
                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                file_name = "{0}_{1}_{2}_{3}_{4}.gz"\
                    .format(estimator, dataset.code, str(subject).zfill(3), "-".join(list(X.event_id)), method)
                fc_file = os.path.join(file_path, file_name)
                if os.path.exists(fc_file):
                    print("Loading {0} ...".format(estimator))
                    # load
                    with gzip.open(fc_file, "r") as f:
                        Xfc = pickle.load(f)
                    X._data = Xfc[method]
                else:
                    print("Computing {0} ...".format(estimator))
                    preproc = clone(clf.named_steps[estimator])
                    # compute estimator
                    X._data = preproc.transform(X=X)
                    # save
                    with gzip.open(fc_file, "w") as f:  # save
                        Xfc = {method: X._data}
                        pickle.dump(Xfc, f)
                step_index = [i for i, (name, _) in enumerate(clf.steps) if name == estimator][0]
                clf.steps.pop(step_index)
            # else:
                # concatenate connectivity matrix

        return X._data, clf

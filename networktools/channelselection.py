"""Code for network features selection."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from itertools import compress
from scipy import stats
from sklearn.metrics import accuracy_score
from moabb_settings import save_global
import networktools.net_topo as net_topo


class NetSelection(TransformerMixin, BaseEstimator):

    def __init__(self, dataset=None, subject=None, name=None, session=None, sessions_name=None, metric=None,
                 ch_names=None, montage_name=None, pipeline=None, cv_splits=None):
        self.dataset = dataset
        self.subject = subject
        self.name = name
        self.session = session
        self.sessions_name = sessions_name
        self.metric = metric
        self.ch_names = ch_names
        self.montage_name = montage_name
        self.scoring = 'roc_auc'
        self.pipeline = pipeline
        self.cv_splits = cv_splits

    def fit(self, X, y):
        self.selection = self._get_selection(X, y)

        self.subelec_names = [row[1] for row in self.selection]
        self.t_val = [row[0] for row in self.selection]
        metric = self.metric
        if self.pipeline == 'fusion+SVM':
            self.metric = [row[3] for row in self.selection]
        save_global(self)
        self.metric = metric

        return self

    def transform(self, X_cv):
        selec_idx = [row[2] for row in self.selection]

        X_selec = X_cv[:, selec_idx]
        # print('cv: {0}, ch: {1}, size: {2}'.format(n_cv, [row[1] for row in self.selection], np.shape(X_selec)))
        return X_selec

    def _sort_selection(self, X, y):
        # Separate data into two classes based on labels
        class1 = X[y == np.unique(y)[0]]
        class2 = X[y == np.unique(y)[1]]

        t_val = []
        # Calculate t-test values for each feature
        for n in np.arange(0, np.size(class1, 1)):
            n_t_val, _ = stats.ttest_ind(class1[:, n], class2[:, n], equal_var=False)
            t_val.append(n_t_val)

        ch_list = []
        metric_list = []
        for metric in self.metric:
            # Check if it's a lateralization metric
            net_metric_lateralization = ['local_laterality',
                                         'segregation',
                                         'integration',
                                         ]
            if metric in net_metric_lateralization:
                ch_pos = net_topo.positions_matrix(self.montage_name, self.ch_names)
                rh_idx, lh_idx, _, _ = net_topo.channel_idx(self.ch_names, ch_pos)
                ch_list += list(np.array(self.ch_names)[lh_idx])
                metric_list += [metric] * len(lh_idx)
            else:
                ch_list += list(np.array(self.ch_names))
                metric_list += [metric] * len(self.ch_names)

        # Create a list of tuples with t-test values, channel names, indices, and metric names
        sort_selection = sorted(zip(abs(np.array(t_val)), np.array(ch_list), np.arange(len(t_val)), np.array(metric_list)), reverse=True)

        # Get back t-test sign in the sorted list
        for i in np.arange(len(sort_selection)):
            sort_selection[i] = list(sort_selection[i])
            sort_selection[i][0] = t_val[sort_selection[i][2]]

        return sort_selection

    def _get_selection(self, X, y):
        # rank and select based on classification score

        # Split data for cross-validation
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)  # in order to avoid overfitting

        # Initialize StratifiedKFold
        skf = StratifiedKFold(5, shuffle=False, random_state=None)

        best_features = []
        for train_cv_idx, val_cv_idx in skf.split(X_train, y_train):
            # hand-made cross-val
            X_train_cv, X_val_cv = X_train[train_cv_idx], X_train[val_cv_idx]
            y_train_cv, y_val_cv = y_train[train_cv_idx], y_train[val_cv_idx]

            # Rank features
            sort_selection = self._sort_selection(X_train_cv, y_train_cv)

            score = 0
            candidate_mask = np.zeros(shape=np.shape(X_train)[1], dtype=bool)
            features_mask = np.zeros(shape=np.shape(X_train)[1], dtype=bool)

            # Select the top 10 features
            clf = SVC(kernel='linear')
            # for feature, i in zip(sort_selection, np.arange(len(sort_selection))):
            for feature, i in zip(sort_selection, np.arange(0, 10)):
                # Include the feature/channel in the candidate set
                candidate_mask[feature[2]] = True

                # Update data with candidate features
                X_train_cv_new = X_train_cv[:, candidate_mask]
                X_val_cv_new = X_val_cv[:, candidate_mask]

                # Train and validate a linear SVM classifier
                clf.fit(X_train_cv_new, y_train_cv)
                new_score = roc_auc_score(y_val_cv, clf.decision_function(X_val_cv_new))

                # Compare with the previous best score and update the feature mask
                if new_score > score:
                    features_mask[i] = True
                    score = new_score
                else:
                    candidate_mask[feature[2]] = False

            # Extend the list of best_features with selected features of this fold
            best_features.extend(list(compress(sort_selection, features_mask)))

        # Remove duplicates based on feature index
        indixes = np.unique([row[2] for row in best_features], return_index=True)
        selection = [best_features[i] for i in indixes[1]]

        return selection


    def _net_train_test(self, X, y):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Forward selection using scikit-learn and MNE-Python with SVM
        selected_features = []
        remaining_features = list(range(X_train.shape[1]))
        while remaining_features:
            best_feature = None
            best_score = -1
            for feature in remaining_features:
                candidate_features = selected_features + [feature]
                X_train_selected = X_train[:, candidate_features]
                # Create a SVM classifier and fit it on the selected features
                classifier = SVC(kernel='linear')
                classifier.fit(X_train_selected, y_train)
                # Evaluate the classifier on the testing set
                X_test_selected = X_test[:, candidate_features]
                y_pred = classifier.predict(X_test_selected)
                score = accuracy_score(y_test, y_pred)
                if score > best_score:
                    best_score = score
                    best_feature = feature
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:
                break

        return selected_features
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


LATERALIZATION_METRIC = ["local_laterality", "segregation", "integration"]


class NetSelection(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        dataset=None,
        name=None,
        session=None,
        sessions_name=None,
        metric=None,
        ch_names=None,
        montage_name=None,
        pipeline=None,
        cv_splits=None,
    ):
        self.selection = None
        self.subelec_names = None
        self.t_val = None
        self.rank = "t-test"
        self.dataset = dataset
        self.name = name
        self.session = session
        self.sessions_name = sessions_name
        self.metric = metric
        self.ch_names = ch_names
        self.montage_name = montage_name
        self.scoring = "roc_auc"
        self.pipeline = pipeline
        self.cv_splits = cv_splits

    def fit(self, X, y):
        """Fit the model to the training data.

        Parameters
        ----------
        X : {array-like} of shape (n_trials, n_channels)
            The

        y : {array-like}, shape (n_samples,)
            The target values.

        Returns
        -------
        self : Returns an instance of self.
        """
        # Calculate feature selection
        self.selection = self._get_selection(X, y)

        # Extract names and t-values from the selection
        self.subelec_names = [row[1] for row in self.selection]
        self.t_val = [row[0] for row in self.selection]

        # For testing network metrics fusion
        original_metric = self.metric
        if self.pipeline is "fusion+SVM":
            self.metric = [row[3] for row in self.selection]

        # Save cross validation selected features
        save_global(self)

        # Restore the original metric attribute
        self.metric = original_metric

        return self

    def transform(self, X_cv):
        """Transform the input data based on the selected features.

        Parameters
        ----------
        X_cv : {array-like} of shape (n_trials, n_channels)
               The training data of network properties.

        Returns
        -------
        X_selec : {array-like} of shape (n_trials, n_selected_channels)
                  The selected subset of network properties.
        """
        selec_idx = [row[2] for row in self.selection]

        X_selec = X_cv[:, selec_idx]
        # print('cv: {0}, ch: {1}, size: {2}'.format(n_cv, [row[1] for row in self.selection], np.shape(X_selec)))
        return X_selec

    def _get_selection(self, X, y):
        """Rank and select based on classification score

        Parameters
        ----------
        X : {array-like} of shape (n_trials, n_channels)
            The

        y : {array-like}, shape (n_samples,)
            The target values.

        Returns
        -------
        selection : {array-like} of shape (n_trials, n_selected_channels)

        """
        # Split data for cross-validation
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )  # in order to avoid over fitting

        # Initialize StratifiedKFold
        skf = StratifiedKFold(5, shuffle=False, random_state=None)

        best_features = []
        for train_cv_idx, val_cv_idx in skf.split(X_train, y_train):
            # cross-val
            X_train_cv, X_val_cv = X_train[train_cv_idx], X_train[val_cv_idx]
            y_train_cv, y_val_cv = y_train[train_cv_idx], y_train[val_cv_idx]

            # Rank features
            sort_selection = self._rank_features(X_train_cv, y_train_cv)
            features_mask = self._select_features(
                sort_selection, X_train_cv, X_val_cv, y_train_cv, y_val_cv
            )

            # Extend the list of best_features with selected features of this fold
            best_features.extend(list(compress(sort_selection, features_mask)))

        return self._remove_duplicates(best_features)

    def _rank_features(self, X, y):
        """
        Rank features based on their discriminative power.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,)
            Target labels.

        Returns:
        --------
        sort_selection : list of tuples
            Sorted list of features based on their discriminative power,
            containing tuples of (t-test value, channel name, index, metric name).
        """
        if self.rank == "t-test":
            # Separate data into two classes based on labels
            class1 = X[y == np.unique(y)[0]]
            class2 = X[y == np.unique(y)[1]]

            # Calculate t-test values for each feature
            rank_param = stats.ttest_ind(class1, class2, equal_var=False)[0]

        elif self.rank == "score":
            # Oder by individual classification performance
            scores = []
            for channel_idx in range(np.shape(X)[1]):
                # Select data corresponding to the current channel
                X_channel = X[:, channel_idx].reshape(-1, 1)

                # Split, train and test
                X_train, X_test, y_train, y_test = train_test_split(
                    X_channel, y, test_size=0.2, random_state=0, stratify=y
                )
                clf = SVC(kernel="linear")
                clf.fit(X_train, y_train)
                score = roc_auc_score(y_test, clf.decision_function(X_test))
                scores.append(score)
            rank_param = scores

        # Generate lists of channel names and metric names based on self.metric
        ch_list = []
        metric_list = []
        for metric in self.metric:
            # Check if it's a lateralization metric
            if metric in LATERALIZATION_METRIC:
                # Selection get reduce to half of the channels because of redundancy
                ch_pos = net_topo.positions_matrix(self.montage_name, self.ch_names)
                rh_idx, lh_idx, _, _ = net_topo.channel_idx(self.ch_names, ch_pos)
                ch_list += list(np.array(self.ch_names)[lh_idx])
                metric_list += [metric] * len(lh_idx)
            else:
                ch_list += list(np.array(self.ch_names))
                metric_list += [metric] * len(self.ch_names)

        # Create a list of tuples with t-test values, channel names, indices, and metric names
        sort_selection = sorted(
            zip(rank_param, ch_list, range(len(rank_param)), metric_list),
            key=lambda x: abs(x[0]),  # Sort by absolute value of the rank_param
            reverse=True,
        )

        return sort_selection

    def _select_features(
        self, sort_selection, X_train_cv, X_val_cv, y_train_cv, y_val_cv
    ):
        """Select top features based on classification score.

        Parameters
        ----------
        sort_selection : list of tuples
            Sorted list of features based on their discriminative power,
            containing tuples of (t-test value, channel name, index, metric name)

        X_train_cv : {array-like} of shape (n_samples, n_channels)
            Training data for cross-validation.

        X_val_cv : {array-like} of shape (n_samples, n_channels)
            Validation data for cross-validation.

        y_train_cv : {array-like} of shape (n_samples,)
            Target labels for training data.

        y_val_cv : {array-like} of shape (n_samples,)
            Target labels for validation data.

        Returns
        -------
        features_mask : {array-like} of shape (n_features)
            Boolean mask indicating selected features.
        """
        features_mask = np.zeros(shape=np.shape(X_train_cv)[1], dtype=bool)
        candidate_mask = np.zeros(shape=np.shape(X_train_cv)[1], dtype=bool)
        clf = SVC(kernel="linear")
        score = 0  # Initialize score variable

        # Include the feature/channel in the candidate set
        for feature, i in zip(sort_selection, range(10)):
            # candidate_mask = np.zeros(shape=np.shape(X_train_cv)[1], dtype=bool)
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

        return features_mask

    def _remove_duplicates(self, best_features):
        """Remove duplicate features.

        Parameters
        ----------
        best_features : list of tuples
            Sorted list of features based on their discriminative power,
            containing tuples of (t-test value, channel name, index, metric name)
        Returns
        -------
        unique_features : {array-like} of shape (n_features)
                          The list of selected unique features.
        """
        indexes = np.unique([row[2] for row in best_features], return_index=True)
        return [best_features[i] for i in indexes[1]]

    def _net_train_test(self, X, y):
        """
        Perform feature selection using forward selection.

        Parameters:
        X (array-like): Input data.
        y (array-like): Target labels.

        Returns:
        selected_features (list): List of selected features.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
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
                classifier = SVC(kernel="linear")
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

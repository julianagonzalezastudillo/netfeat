"""Code for network features selection."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from itertools import compress

from src.moabb_settings import save_global
from src.config import N_FEATURES


class PSDSelection(TransformerMixin, BaseEstimator):
    """Feature selection for PSD .

    Parameters
        ----------
        dataset : List of Dataset instance
        name : str or None, default=None
            Name of the model.
        session : int or None, default=None
            Session number.
        sessions_name : list or None, default=None
            Session name.
        metric : str or None, default=None
            Nework metric name.
        ch_names : list or None, default=None
            List of channel names.
        montage_name : str or None, default=None
            Montage name.
        pipeline : str or None, default=None
            Pipeline name.
        cv_splits : int or None, default=None
            Number of cross-validation splits.
    References
    ----------

    [1] Dominguez, L. G. (2009). “On the risk of extracting relevant information from random data”.
    In: Journal of neural engineering 6.5, p. 058001.
    """

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
        self.rank = "score"  # "t-test" or "score"
        self.dataset = dataset
        self.name = name
        self.session = session
        self.sessions_name = sessions_name
        self.metric = "psd"
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

        # Save cross validation selected features
        save_global(self)
        # print(f"fit")
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
        # print(f"X_selec = {X_selec}")
        return X_selec

    def _get_selection(self, X, y):
        """Rank and select based on classification score

        Parameters
        ----------
        X : {array-like} of shape (n_trials, n_channels)
            Input data.

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
        # print(f"best_features = {self._remove_duplicates(best_features)}")
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
        # Oder by individual classification performance
        scores = []
        for channel_idx in range(np.shape(X)[1]):
            # Select data corresponding to the current channel
            x_channel = X[:, channel_idx].reshape(-1, 1)

            # Split, train and test
            x_train, x_test, y_train, y_test = train_test_split(
                x_channel, y, test_size=0.2, random_state=0, stratify=y
            )
            clf_rank = LinearSVC()
            clf_rank.fit(x_train, y_train)
            score = roc_auc_score(y_test, clf_rank.decision_function(x_test))
            scores.append(score)
        rank_param = scores

        # Generate lists of channel names and metric names based on self.metric
        ch_list = []
        ch_list += list(np.array(self.ch_names))

        # Create a list of tuples with t-test values, channel names, indices, and metric names
        sort_selection = sorted(
            zip(rank_param, ch_list, range(len(rank_param))),
            key=lambda x: abs(x[0]),  # Sort by absolute value of the rank_param
            reverse=True,
        )
        # print(f"sort_selection = {sort_selection}")
        return sort_selection

    @staticmethod
    def _select_features(sort_selection, x_train_cv, x_val_cv, y_train_cv, y_val_cv):
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
        features_mask = np.zeros(shape=np.shape(x_train_cv)[1], dtype=bool)
        candidate_mask = np.zeros(shape=np.shape(x_train_cv)[1], dtype=bool)
        clf_select = SVC(kernel="linear")
        score = 0  # Initialize score variable

        # Include the feature/channel in the candidate set
        for feature, i in zip(sort_selection, range(N_FEATURES)):
            # candidate_mask = np.zeros(shape=np.shape(X_train_cv)[1], dtype=bool)
            candidate_mask[feature[2]] = True

            # Update data with candidate features
            X_train_cv_new = x_train_cv[:, candidate_mask]
            X_val_cv_new = x_val_cv[:, candidate_mask]

            # Train and validate a linear SVM classifier
            clf_select.fit(X_train_cv_new, y_train_cv)
            new_score = roc_auc_score(
                y_val_cv, clf_select.decision_function(X_val_cv_new)
            )

            # Compare with the previous best score and update the feature mask
            if new_score > score:
                features_mask[i] = True
                score = new_score
            else:
                candidate_mask[feature[2]] = False
        # print(f"features_mask = {features_mask}")
        return features_mask

    @staticmethod
    def _remove_duplicates(best_features):
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
        # print(f"_remove_duplicates = {[best_features[i] for i in indexes[1]]}")
        return [best_features[i] for i in indexes[1]]

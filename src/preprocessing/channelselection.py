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
from src.moabb_settings import save_global
import networktools.net_topo as net_topo
from src.config import LATERALIZATION_METRIC
import traceback


class FeaturesSelection(TransformerMixin, BaseEstimator):
    """Feature selection for network properties.

    We implemented an embedded approach to select the best discriminant features. We use a sequential forward feature
    selection. Within a nested cross-validation framework, this algorithm adds features to form a feature subset. To
    limit the research complexity, at each stage we rank our features from the training folds according to their
    discrimination power (t-test or classification score). Then we perform a bottom up search procedure which
    conditionally includes new features to the selected set based on the cross-validation score.

    Within a cross-validation framework, this approach uses a nested 5-fold cross-validated SVM on a sub-training set
    (80% of the original cv training) [1], to obtain a subset of selected features, N′ = 10. For each nested cv
    iteration, features are ranked according to their discriminant power between classes. In a forward sequential order,
    a feature is going to be retained and accumulated in the selected set, if its accuracy is higher than the previous
    set. The output of this nested cv is a group of 10 selected features on which the original cv validation set is
    going to be tested. This is repeated for each iteration in the original CV.

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
        rank="t-test",  # "t-test" or "score"
    ):
        self.selection = None
        self.subelec_names = None
        self.t_val = None
        self.rank = rank
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

        # Save cross validation selected features
        save_global(self)
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
        try:
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
                    clf_rank = SVC(kernel="linear")
                    clf_rank.fit(X_train, y_train)
                    score = roc_auc_score(y_test, clf_rank.decision_function(X_test))
                    scores.append(score)
                rank_param = scores

            # Generate lists of channel names and metric names based on self.metric
            ch_list = []
            metric_list = []
            # for metric in self.metric:
            # Check if it's a lateralization metric
            if self.metric in LATERALIZATION_METRIC:
                # Selection get reduce to half of the channels because of redundancy
                if self.dataset.code == "Grosse-Wentrup 2009":
                    rh_idx, lh_idx, ch_idx, ch_bis_idx = net_topo.channel_idx_MunichMI()
                else:
                    ch_pos = net_topo.positions_matrix("standard_1005", self.ch_names)
                    rh_idx, lh_idx, _, _ = net_topo.channel_idx(self.ch_names, ch_pos)
                ch_list += list(np.array(self.ch_names)[lh_idx])
                metric_list += [self.metric] * len(lh_idx)
            else:
                ch_list += list(np.array(self.ch_names))
                metric_list += [self.metric] * len(self.ch_names)

            # Create a list of tuples with t-test values, channel names, indices, and metric names
            sort_selection_ = sorted(
                zip(rank_param, ch_list, range(len(rank_param)), metric_list),
                key=lambda x: abs(x[0]),  # Sort by absolute value of the rank_param
                reverse=True,
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
        return sort_selection_

    @staticmethod
    def _select_features(sort_selection, X_train_cv, X_val_cv, y_train_cv, y_val_cv):
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
        for feature, i in zip(sort_selection, range(len(sort_selection))):
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

            if sum(features_mask) == np.shape(X_train_cv)[0]:
                print(f"maximum number of features reached: {sum(features_mask)}")
                break
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
        return [best_features[i] for i in indexes[1]]

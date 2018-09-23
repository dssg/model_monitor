import numpy as np
import sklearn.metrics as skmetrics


# --------------------------------------------------------------------------------------------------------------------
# Confusion matrix loss functions
# --------------------------------------------------------------------------------------------------------------------

def gmean(cm, prior):
    """
    Computes geometric mean loss from confusion matrix and prior

    :param cm: np.ndarray
    :param prior: np.array
    :return: float
    """
    return 1 - np.sqrt(np.prod(np.diag(cm) / prior))


def hmean(cm, prior):
    """
    Computes harmonic mean loss from confusion matrix and prior

    :param cm: np.ndarray
    :param prior: np.array
    :return: float
    """
    return 1 - 2 * (np.sum(prior / np.diag(cm))) ** (-1)


def qmean(cm, prior):
    """
    Computes q-mean loss from confusion matrix and prior

    :param cm: np.ndarray
    :param prior: np.array
    :return: float
    """
    return np.sqrt(1 / 2 * np.sum((1 - np.diag(cm) / prior) ** 2))


def minimax(cm, prior):
    """
    Computes minimax loss from confusion matrix and prior

    :param cm: np.ndarray
    :param prior: np.array
    :return: float
    """
    return np.max(1 - np.diag(cm) / prior)


# --------------------------------------------------------------------------------------------------------------------
# Prediction preprocessors (loss handler)
# --------------------------------------------------------------------------------------------------------------------


class PredictionPreprocessor(object):
    """
    Class to handle preprocessing of predictions table results
    """

    def __init__(self, prediction_df):
        """
        Constructor

        :param prediction_df: pd.DataFrame
        """
        self._raw_df = prediction_df
        self.n = len(prediction_df)

        # precision-recall breakpoints
        self._precision, self._recall, self._thresholds = skmetrics.precision_recall_curve(
            prediction_df['label'],
            prediction_df['score']
        )

        # prior
        prior1 = np.average(prediction_df['label'])
        self.prior = [1. - prior1, prior1]

    def raw_scores(self):
        """
        Raw scores from prediction DataFrame

        :return: pd.Series
        """
        return self._raw_df['score']

    def raw_labels(self):
        """
        Raw labels from prediction DataFrame

        :return: pd.Series
        """
        return self._raw_df['label']

    # ---------------------------------------------------------------------------------------------------------------
    # Precision filtering
    # ---------------------------------------------------------------------------------------------------------------

    def decision_boundary_at_precision(self, p):
        """
        Decision boundary at a fixed precision

        :param p: float, precision
        :return: float
        """
        return self._thresholds[np.argmax(self._precision >= p)]

    def predictions_at_precision(self, p):
        """
        Predictions for a fixed precision

        :param p: float, precision
        :return: np.array
        """
        return np.where(self.raw_scores() > self.decision_boundary_at_precision(p), 1, 0)

    def confusion_matrix_at_precision(self, p, normalize=True):
        """
        Confusion matrix at fixed precision

        :param p: float, precision
        :param normalize: bool, if True then normalize matrix
        :return: np.ndarray
        """
        m = skmetrics.confusion_matrix(self.raw_labels(), self.predictions_at_precision(p))
        if normalize:
            return m / float(self.n)
        return m

    # ---------------------------------------------------------------------------------------------------------------
    # Extremal entity filtering
    # ---------------------------------------------------------------------------------------------------------------

    def decision_boundary_at_top_n(self, n):
        """
        Decision boundary at a fixed top n entities

        :param n: int, top n entities
        :return: float
        """
        return np.array(self.raw_scores())[-n]

    def predictions_at_top_n(self, n):
        """
        Predictions for a fixed top n entities

        :param n: int, top n entities
        :return: np.array
        """
        return np.where(self.raw_scores() > self.decision_boundary_at_top_n(n), 1, 0)

    def confusion_matrix_at_top_n(self, n, normalize=True):
        """
        Confusion matrix at fixed top n entities

        :param n: int, top n entities
        :param normalize: bool, if True then normalize matrix
        :return: np.ndarray
        """
        m = skmetrics.confusion_matrix(self.raw_labels(), self.predictions_at_top_n(n))
        if normalize:
            return m / float(self.n)
        return m


class CategoryPredictionPreprocessor(object):
    """
    Prediction variable preprocessor for categorical subsets
    """

    def __init__(self, prediction_df, category_column):
        """
        Constructor

        :param prediction_df: pd.DataFrame
        :param category_column: str, column in pd.DataFrame with k categories, must contain int values 0 through k-1
        """

        # preprocess all categories individually
        self.all_entities_handler = PredictionPreprocessor(prediction_df)
        self.unique_categories = np.unique(prediction_df[category_column])
        self.category_group_handlers = [PredictionPreprocessor(prediction_df[prediction_df[category_column] == v])
                                        for v in self.unique_categories]

    def parity_score(self, p, parity_type='positive_label', return_most_biased_category=False):
        """
        Parity score compared against categorical variables

        Possible component calculations (expressed in terms of their constituent components)
            - 'positive_odds': C[1,1]
            - 'negative_odds': C[0, 1]
            - 'positive_label': C[0, 1] + C[1, 1]
            - 'negative_label': C[0, 0] + C[1, 0]

        :param p: float, precision
        :param parity_type: str, one of ['positive_odds', 'negative_odds', 'positive_label', 'negative_label']
        :param return_most_biased_category: if True, return category associated with max bias, else return score
        :return: float
        """

        assert parity_type in ['positive_odds', 'negative_odds', 'positive_label', 'negative_label']

        # get category confusion matrix for each group handler
        category_cms = [pre.confusion_matrix_at_precision(p) for pre in self.category_group_handlers]

        # apply parity calculation
        if parity_type == 'positive_label':
            category_components = np.array([cm[0, 1] + cm[1, 1] for cm in category_cms])
        elif parity_type == 'negative_label':
            category_components = np.array([cm[0, 0] + cm[1, 0] for cm in category_cms])
        elif parity_type == 'positive_odds':
            category_components = np.array([cm[1, 1] for cm in category_cms])
        else:
            category_components = np.array([cm[0, 1] for cm in category_cms])

        score_deviations = np.abs(category_components - np.mean(category_components))

        # return most biased category if specified, else max value (score)
        return np.argmax(score_deviations) if return_most_biased_category else np.max(score_deviations)


def transition_matrix(v0, v1, normalize=True):
    """
    Matrix of transition rates from 0->1

    :param v0: np.array
    :param v1: np.array
    :param normalize: bool, if True then normalize matrix
    :return: np.ndarray
    """
    m = np.unique(np.array([v0, v1]), axis=1, return_counts=True)[1].reshape((2, 2))

    if normalize:
        return m / float(len(v0))
    return m

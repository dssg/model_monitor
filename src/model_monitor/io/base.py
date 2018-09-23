from abc import ABC, abstractmethod


class BaseResultsExtractor(ABC):
    """
    Base class for extracting model results
    """

    @abstractmethod
    def select_training_matrices(self):
        """
        Query training matrices table

        :return: pd.DataFrame
        """

    @abstractmethod
    def select_testing_matrices(self):
        """
        Query training matrices table

        :return: pd.DataFrame
        """

    @abstractmethod
    def select_models(self):
        """
        Query model table

        :return: pd.DataFrame
        """
        pass

    @abstractmethod
    def select_model_groups(self):
        """
        Query model groups table

        :return: pd.DataFrame
        """
        pass

    @abstractmethod
    def select_predictions(self):
        """
        Query predictions table

        :return: pd.DataFrame
        """
        pass

    @abstractmethod
    def select_feature_importances(self):
        """
        Query feature importances table

        :return: pd.DataFrame
        """
        pass


class BaseFeatureExtractor(ABC):

    @abstractmethod
    def load_matrix_by_hash(self, matrix_hash):
        """
        Method to load matrix by matrix hash

        :param matrix_hash: str, matrix UUID
        :return: pd.DataFrame
        """
        pass

    @abstractmethod
    def load_features_by_hash(self, matrix_hash, feature):
        """
        Method to load individual feature by matrix hash

        :param matrix_hash: str, matrix UUID
        :param feature: str, feature name
        :return: pd.Series
        """
        pass

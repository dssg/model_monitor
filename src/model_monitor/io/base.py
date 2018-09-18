from abc import ABC, abstractmethod


class BaseResultsExtractor(ABC):
    """
    Base class for extracting model results
    """

    @abstractmethod
    def select_predictions(self, target_date):
        """
        Query predictions table

        :param target_date: str, format YYYY-MM-DD
        :return: pd.DataFrame
        """
        pass

    @abstractmethod
    def select_feature_importances(self, target_date):
        """
        Query feature importances table

        :param target_date: str, format YYYY-MM-DD
        :return: pd.DataFrame
        """
        pass

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

import os
import datetime
import pandas as pd

from model_monitor.io.base import BaseResultsExtractor, BaseFeatureExtractor
from model_monitor.io.shared import get_sqlalchemy_engine_by_conn_id, get_s3_client_by_conn_id


class TriageResultsExtractor(BaseResultsExtractor):
    """
    Results extractor for default triage project
    """
    def __init__(self, conn_id, mm_config):
        """
        Constructor

        :param conn_id: str, connection ID (used by Airflow to construct engine)
        :param mm_config: dict
        """

        self._engine = get_sqlalchemy_engine_by_conn_id(conn_id)

        # argument parsing - nulls need separate clauses for array-like SQL arguments

        # default arguments (NB: lists are unused dummy parameters)
        self.model_group_id_params = {
            'included_model_group_ids': [0, ],
            'filter_included_model_group_ids': False,
            'excluded_model_group_ids': [0, ],
            'filter_excluded_model_group_ids': False,
        }
        self.model_type_params = {
            'included_model_types': ["", ],
            'filter_included_model_types': False,
            'excluded_model_types': ["", ],
            'filter_excluded_model_types': False,
        }

        self.model_id_params = {
            'included_model_ids': [0, ],
            'filter_included_model_ids': False,
            'excluded_model_ids': [0, ],
            'filter_excluded_model_ids': False,
        }

        # update to include / exclude model groups
        included_model_group_ids = mm_config['model_targets']['included_model_group_ids']
        if included_model_group_ids:
            self.model_group_id_params['included_model_group_ids'] = included_model_group_ids
            self.model_group_id_params['filter_included_model_group_ids'] = True

        excluded_model_group_ids = mm_config['model_targets']['excluded_model_group_ids']
        if excluded_model_group_ids:
            self.model_group_id_params['excluded_model_group_ids'] = excluded_model_group_ids
            self.model_group_id_params['filter_excluded_model_group_ids'] = True

        # include / exclude model types
        included_model_types = mm_config['model_targets']['included_model_types']
        if included_model_types:
            self.model_type_params['included_model_types'] = included_model_types
            self.model_type_params['filter_included_model_types'] = True

        excluded_model_types = mm_config['model_targets']['excluded_model_types']
        if included_model_types:
            self.model_type_params['excluded_model_types'] = excluded_model_types
            self.model_type_params['filter_included_model_types'] = True

        # include / exclude models
        included_model_ids = mm_config['model_targets']['included_model_ids']
        if included_model_ids:
            self.model_id_params['included_model_ids'] = included_model_ids
            self.model_id_params['filter_included_model_ids'] = True

        excluded_model_ids = mm_config['model_targets']['excluded_model_ids']
        if excluded_model_ids:
            self.model_id_params['excluded_model_ids'] = excluded_model_ids
            self.model_id_params['filter_excluded_model_ids'] = True

    def _read_triage_query(self, fname):
        """
        Read default query from file in "triage_queries" folder

        :param fname: filename
        :return: str, query text
        """
        cdir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cdir, 'triage_queries', fname), mode='r') as f:
            return f.read()

    # ----------------------------------------------------------------------------------------------------------------
    # Base table queries
    # ----------------------------------------------------------------------------------------------------------------

    def select_model_groups(self):
        """
        Select model groups according to configuration filters

        :return: pd.DataFrame
        """
        query = self._read_triage_query('model_groups.sql')

        params = self.model_group_id_params.copy()
        params.update(self.model_type_params)

        return pd.read_sql(query, params=params, con=self._engine)

    def select_models(self):
        """
        Select models according to configuration filters

        :return: pd.DataFrame
        """
        query = self._read_triage_query('models.sql')

        params = self.model_group_id_params.copy()
        params.update(self.model_type_params)
        params.update(self.model_id_params)

        return pd.read_sql(query, params=params, con=self._engine)

    def select_predictions(self, target_date):
        """
        Select predictions according to configuration filters

        :param target_date: str, format "YYYY-MM-DD"
        :return: pd.DataFrame
        """
        query = self._read_triage_query('predictions.sql')

        params = {'target_date': datetime.datetime.strptime(target_date, "%Y-%m-%d")}
        params.update(self.model_group_id_params)
        params.update(self.model_type_params)
        params.update(self.model_id_params)

        return pd.read_sql(query, con=self._engine, params=params)

    def select_feature_importances(self, target_date):
        """
        Select feature importances according to configuration filters

        :param target_date: str, format "YYYY-MM-DD"
        :return: pd.DataFrame
        """
        query = self._read_triage_query('feature_importances.sql')

        params = {'target_date': datetime.datetime.strptime(target_date, "%Y-%m-%d")}
        params.update(self.model_group_id_params)
        params.update(self.model_type_params)
        params.update(self.model_id_params)

        return pd.read_sql(query, con=self._engine, params=params)


# --------------------------------------------------------------------------------------------------------------------
# Custom view results extractor
# --------------------------------------------------------------------------------------------------------------------


class CustomViewTriageResultsExtractor(TriageResultsExtractor):
    """
    Results extractor for "modified" triage-style project

    This class allows the user to define drop-in view replacements for different tables in results-schema. Use this
    extractor if you already use postgres and your schema is similar, but not an exact match, to the current version
    of results-schema in triage.
    """
    def __init__(self,
                 conn_id,
                 mm_config):
        """
        Constructor

        :param conn_id: str, connection ID (used by Airflow to construct engine)
        :param mm_config: dict
        """

        TriageResultsExtractor.__init__(self, conn_id, mm_config)

        # define drop-in view replacements for existing SQL code

        self.view_override_sources = mm_config['runtime_settings']['results_extractor_args']['view_override_sources']

        self._view_from_overrides = {
            "FROM {source}": "FROM {source}_vw".format(source=source)
            for source in self.view_override_sources
        }

        self._view_join_overrides = {
            "JOIN {source}": "JOIN {source}_vw".format(source=source)
            for source in self.view_override_sources
        }

    def _read_triage_query(self, fname):
        """
        Load triage query, referencing external views where specified in view_override_sources

        :param fname: filename
        :return: str, query text
        """

        # get original triage query
        original_query = TriageResultsExtractor._read_triage_query(self, fname)

        # update from statements with view overrides
        for orig, replace in self._view_from_overrides.items():
            original_query = original_query.replace(orig, replace)

        # update join statements with view overrates
        for orig, replace in self._view_from_overrides.items():
            original_query = original_query.replace(orig, replace)

        return original_query


# ----------------------------------------------------------------------------------------------------------------
# Triage s3 feature extractor
# ----------------------------------------------------------------------------------------------------------------

class TriageS3FeatureExtractor(BaseFeatureExtractor):
    """
    Base class for extracting from an s3-stored feature matrix
    """

    def __init__(self,
                 conn_id,
                 mm_config):
        """
        Constructor
        :param conn_id: str, connection ID (used by Airflow to create boto3 client)
        :param mm_config: dict
        """

        super(TriageS3FeatureExtractor, self).__init__()

        self.client = get_s3_client_by_conn_id(conn_id)
        self._bucket = mm_config['runtime_settings']['s3_bucket']
        self._root_key = mm_config['runtime_settings']['s3_root_key']
        self._initialized_matrices = set()

    def _initialize(self, matrix_hash):
        """
        Initialize matrix from s3 with temporary location

        :param matrix_hash: str, uuid
        :return: None
        """

        full_key = '{}/{}.h5'.format(self._root_key, matrix_hash)
        tmp_loc = '/tmp/{}.h5'.format(matrix_hash)

        self.client.Object(Bucket=self._bucket, key=full_key).put(Body=open('/tmp/{}.h5'.format(matrix_hash), 'rb'))
        self._initialized_matrices.add(tmp_loc)

    def load_matrix_by_hash(self, matrix_hash):
        """
        Load matrix by hash

        :param matrix_hash: str, uuid
        :return: pd.DataFrame
        """

        # check if need to pull from s3 to tmp location
        tmp_loc = '/tmp/{}.h5'.format(matrix_hash)
        if tmp_loc not in self._initialized_matrices:
            self._initialize(matrix_hash)

        # read matrix
        with pd.HDFStore(tmp_loc, mode='r') as stor:
            return stor['/{}'.format(matrix_hash)]

    def load_features_by_hash(self, matrix_hash, feature):
        """
        Load matrix feature by hash

        :param matrix_hash: str, uuid
        :param feature: str, feature name
        :return: pd.Series
        """

        with pd.HDFStore('/tmp/{}.h5', mode='r') as stor:
            return stor.select('/{}'.format(matrix_hash), columns=feature)

    def cleanup(self, matrix_hash=None):
        """
        Cleanup temporary resources

        :param matrix_hash: str, uuid, if None then removes all initialized matrices in /tmp
        :return: None
        """

        # if removing only one initialized matrix
        if matrix_hash:
            fname = '/tmp/{}.h5'.format(matrix_hash)
            if os.path.exists(fname):
                os.remove(fname)

            self._initialized_matrices.remove(fname)

        # else, remove all initialized matrices
        else:
            for fname in self._initialized_matrices:
                if os.path.exists(fname):
                    os.remove(fname)

                self._initialized_matrices.remove(fname)

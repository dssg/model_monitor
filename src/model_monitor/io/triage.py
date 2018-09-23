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

        epoch_start = datetime.datetime.fromtimestamp(0).strftime("%Y-%m-%d")
        today = datetime.datetime.today().strftime("%Y-%m-%d")

        self.default_args = {
            'included_model_group_ids': [0,],
            'excluded_model_group_ids': [0, ],
            'included_model_types': ["", ],
            'excluded_model_types': ["", ],
            'included_model_ids': [0, ],
            'excluded_model_ids': [0, ],
            'train_as_of_date_start': epoch_start,
            'train_as_of_date_end': today,
            'test_as_of_date_start': epoch_start,
            'test_as_of_date_end': today,
            'filter_included_model_group_ids': False,
            'filter_excluded_model_group_ids': False,
            'filter_included_model_types': False,
            'filter_excluded_model_types': False,
            'filter_included_model_ids': False,
            'filter_excluded_model_ids': False
        }

        model_targets = mm_config['model_targets']

        # model subset parsing
        if model_targets['included_model_group_ids']:
            self.default_args['incldued_model_group_ids'] = model_targets['included_model_group_ids']
            self.default_args['filter_included_model_group_ids'] = True

        if model_targets['excluded_model_group_ids']:
            self.default_args['excldued_model_group_ids'] = model_targets['excluded_model_group_ids']
            self.default_args['filter_excluded_model_group_ids'] = True

        if model_targets['included_model_types']:
            self.default_args['incldued_model_types'] = model_targets['included_model_types']
            self.default_args['filter_included_model_types'] = True

        if model_targets['excluded_model_types']:
            self.default_args['excldued_model_types'] = model_targets['excluded_model_types']
            self.default_args['filter_excluded_model_types'] = True

        if model_targets['included_model_ids']:
            self.default_args['incldued_model_ids'] = model_targets['included_model_ids']
            self.default_args['filter_included_model_ids'] = True

        if model_targets['excluded_model_ids']:
            self.default_args['excldued_model_ids'] = model_targets['excluded_model_ids']
            self.default_args['filter_excluded_model_ids'] = True

        # model date parsing
        if model_targets['train_as_of_date_start']:
            self.default_args['train_as_of_date_start'] = model_targets['train_as_of_date_start']

        if model_targets['train_as_of_date_start']:
            self.default_args['train_as_of_date_start'] = model_targets['train_as_of_date_start']

        if model_targets['test_as_of_date_end']:
            self.default_args['train_as_of_date_end'] = model_targets['train_as_of_date_end']

        if model_targets['test_as_of_date_end']:
            self.default_args['train_as_of_date_end'] = model_targets['train_as_of_date_end']

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
        params = self.default_args.copy()
        return pd.read_sql(query, params=params, con=self._engine)

    def select_models(self):
        """
        Select models according to configuration filters

        :return: pd.DataFrame
        """
        query = self._read_triage_query('models.sql')
        params = self.default_args.copy()
        return pd.read_sql(query, params=params, con=self._engine)

    def select_predictions(self):
        """
        Select predictions according to configuration filters

        :return: pd.DataFrame
        """
        query = self._read_triage_query('predictions.sql')
        params = self.default_args.copy()
        return pd.read_sql(query, params=params, con=self._engine)

    def select_feature_importances(self):
        """
        Select feature importances according to configuration filters

        :return: pd.DataFrame
        """
        query = self._read_triage_query('feature_importances.sql')
        params = self.default_args.copy()

        # drop unused params
        del params['test_as_of_date_start']
        del params['test_as_of_date_end']

        return pd.read_sql(query, params=params, con=self._engine)

    def select_training_matrices(self):
        """
        Select training matrices according to configuration filters

        :return: pd.DataFrame
        """
        query = self._read_triage_query('training_matrices.sql')
        params = self.default_args.copy()

        # drop unused params
        del params['test_as_of_date_start']
        del params['test_as_of_date_end']

        return pd.read_sql(query, params=params, con=self._engine)

    def select_testing_matrices(self):
        """
        Select training matrices according to configuration filters

        :return: pd.DataFrame
        """
        query = self._read_triage_query('testing_matrices.sql')
        params = self.default_args.copy()
        return pd.read_sql(query, params=params, con=self._engine)


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

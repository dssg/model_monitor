import os
import re
import inspect
import itertools
import traceback

import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

from scipy import stats as stats
from sklearn import cluster as cluster

from model_monitor.io.shared import _n_dirname
from model_monitor.io.base import BaseResultsExtractor, BaseFeatureExtractor
from model_monitor.io.triage import TriageResultsExtractor, CustomViewTriageResultsExtractor, TriageS3FeatureExtractor

DEFAULT_RESULTS_EXTRACTORS = {
    'TriageResultsExtractor': TriageResultsExtractor,
    'CustomViewTriageResultsExtractor': CustomViewTriageResultsExtractor
}
DEFAULT_FEATURE_EXTRACTORS = {
    'TriageS3FeatureExtractor': TriageS3FeatureExtractor
}


class BadConfigurationError(ValueError):
    """
    Exception for improperly specified mm_config.yaml
    """
    pass


# --------------------------------------------------------------------------------------------------------------------
# Extractor factories
# --------------------------------------------------------------------------------------------------------------------


def candidate_extractors():
    """
    Collect all implemented candidate classes for ResultsExtractor and FeatureExtractor
    """
    results_extractors = DEFAULT_RESULTS_EXTRACTORS.copy()
    feature_extractors = DEFAULT_FEATURE_EXTRACTORS.copy()

    # scan custom implementation directory
    custom_dir = os.path.join(_n_dirname(os.path.abspath(__file__), 4), 'custom')
    custom_fs = [f[:-3] for f in os.listdir(custom_dir) if f[-3:] == '.py' and f != '__init__.py']

    for f in custom_fs:
        mod = SourceFileLoader(f, "{}/{}.py".format(custom_dir, f))
        mod_loaded = mod.load_module()

        for name, obj in inspect.getmembers(mod_loaded):
            # ignore existing base objects
            if 'Base' in name:
                continue

            if issubclass(obj, BaseResultsExtractor) and name not in results_extractors.keys():
                results_extractors.update({name: obj})

            if issubclass(obj, BaseFeatureExtractor) and name not in feature_extractors.keys():
                feature_extractors.update({name: obj})

    return results_extractors, feature_extractors


def _results_extractor_factory(mmc, results_extractors):
    """
    Construct results extractor from runtime settings

    :param mmc: dict, from mm_config.yaml
    :param results_extractors: dict, from candidate_extractors()
    :return: BaseResultsExtractor
    """

    runtime_settings = mmc['runtime_settings']

    # get required arguments for class instatiation
    if runtime_settings['results_extractor'] not in results_extractors:
        raise BadConfigurationError("Extractor '{}' not available".format(runtime_settings['results_extractor']))
    results_extractor_class = results_extractors[runtime_settings['results_extractor']]
    required_args = {arg for arg in list(inspect.getfullargspec(results_extractor_class.__init__))[0]
                     if arg not in ['mm_config', 'self']}

    # if arguments required
    if required_args:
        for req_arg in required_args:
            if req_arg not in runtime_settings['results_extractor_args'].keys():
                raise BadConfigurationError("Results extractor '{}' missing argument '{}'".format(
                    runtime_settings['results_extractor'], req_arg
                ))
    # construct results extractor
    try:
        results_extractor_args = runtime_settings['results_extractor_args'].copy()
        results_extractor_args['mm_config'] = mmc
        return results_extractor_class(**results_extractor_args)

    except Exception as e:
        raise BadConfigurationError("Results extractor constructor failed, traceback: \n {}".format(e))


def _feature_extractor_factory(mmc, feature_extractors):
    """
    Construct results extractor from runtime settings

    :param mmc: dict, from mm_config.yaml
    :param feature_extractors: dict, from candidate_extractors()
    :return: BaseFeatureExtractor
    """

    runtime_settings = mmc['runtime_settings']

    # get required arguments for class instatiation
    if runtime_settings['feature_extractor'] not in feature_extractors:
        raise BadConfigurationError("Extractor '{}' not available".format(runtime_settings['feature_extractor']))
    feature_extractor_class = feature_extractors[runtime_settings['feature_extractor']]
    required_args = {arg for arg in list(inspect.getfullargspec(feature_extractor_class.__init__))[0]
                     if arg not in ['mm_config', 'self']}

    # if arguments required
    if required_args:
        for req_arg in required_args:
            if req_arg not in runtime_settings['feature_extractor_args'].keys():
                raise BadConfigurationError("Feature extractor '{}' missing argument '{}'".format(
                    runtime_settings['feature_extractor'], req_arg
                ))

    # construct feature extractor
    try:
        feature_extractor_args = runtime_settings['feature_extractor_args'].copy()
        feature_extractor_args['mm_config'] = mmc
        return feature_extractor_class(**feature_extractor_args)
    except Exception as e:
        raise BadConfigurationError("Feature extractor constructor failed, traceback: \n {}".format(e))


# --------------------------------------------------------------------------------------------------------------------
# Feature name parsing
# --------------------------------------------------------------------------------------------------------------------


TIME_AGG_PATTERN = '_([0-9]+[dwmy])_'
TIME_AGG_REGEX = re.compile(TIME_AGG_PATTERN)
TIME_AGG_REGEX_TERM = re.compile(TIME_AGG_PATTERN[:-1] + '$')

AGG_FUNCS = ['min', 'max', 'mode', 'sum', 'avg', 'var', 'variance', 'iqr', 'count', 'rate', 'stddev']
AGG_FUNC_PATTERN = '_({})_'.format('|'.join(AGG_FUNCS))
AGG_FUNC_REGEX = re.compile(AGG_FUNC_PATTERN)
AGG_FUNC_REGEX_TERM = re.compile(AGG_FUNC_PATTERN[:-1] + '$')

DEFAULT_SOURCE_TABLE_REGEX = re.compile('^(.*_id)_')


def _parse_agg_func_type(agg, default=''):
    """
    Map aggregation function to behavior

    :param agg: str
    :param default: default value
    :return: str
    """

    if agg in ['avg', 'mode']:
        return 'central'
    elif agg in ['var', 'variance', 'stddev', 'iqr']:
        return 'variation'
    elif agg in ['max', 'sum', 'count']:
        return 'extremal_upper'
    elif agg == 'min':
        return 'extremal_lower'
    elif agg == 'rate':
        return 'rate'
    else:
        return default


def _parse_default_triage_feature_name(feat):
    """
    Unpack feature name into components using the standard triage structure

    :param feat: str, feature name
    :return: dict
    """
    try:
        # attempt to match (source_table, time_agg, feature_name, agg_func)
        feat_remainder, agg_func = [i for i in AGG_FUNC_REGEX_TERM.split(feat) if i]
        source_table, time_agg, feature_name = TIME_AGG_REGEX.split(feat_remainder)
        return {
            'source_table': source_table,
            'time_agg': time_agg,
            'feature_name': feature_name,
            'agg_func': agg_func
        }
    except:
        try:
            # attempt to match (source_table, feature_name, agg_func)
            feat_remainder, agg_func = [i for i in AGG_FUNC_REGEX_TERM.split(feat) if i]
            _, source_table, feature_name = DEFAULT_SOURCE_TABLE_REGEX.split(feat_remainder)
            return {
                'source_table': source_table,
                'time_agg': '',
                'feature_name': feature_name,
                'agg_func': agg_func
            }
        except:
            return {
                'source_table': '',
                'time_agg': '',
                'feature_name': '',
                'agg_func': ''
            }


# --------------------------------------------------------------------------------------------------------------------
# Random variable definitions
# --------------------------------------------------------------------------------------------------------------------

PREDICTION_RV_REGEX = re.compile('^(score|label)_metrics$')
PREDICTION_AT_PRECISION_RV_REGEX = re.compile('^(prediction_at_precision_[0-9]+)_metrics$')
FEATURE_REGEX = re.compile('^feature_(.*)_metrics$')


def _parse_random_variable_name(rv_name):
    """
    Parse random variable name from config file section

    :param rv_name: str, config file section name
    :return: dict, random variable row definition
    """
    rv_row = {
        'source_table': None,
        'latent_variable_name': None,
        'agg_func': None,
        'time_agg': None
    }

    if PREDICTION_RV_REGEX.match(rv_name):
        rv_row.update({
            'rv_name': PREDICTION_RV_REGEX.match(rv_name).groups()[0],
            'rv_type': 'prediction_raw',
            'source_table': 'predictions'
        })
    elif PREDICTION_AT_PRECISION_RV_REGEX.match(rv_name):
        rv_row.update({
            'rv_name': PREDICTION_AT_PRECISION_RV_REGEX.match(rv_name).groups()[0],
            'rv_type': 'prediction_at_precision',
            'source_table': 'predictions'
        })

    elif FEATURE_REGEX.match(rv_name):
        # try parse additional info
        feature_full_name = FEATURE_REGEX.match(rv_name).groups()[0]
        rv_row.update({
            'rv_name': feature_full_name,
            'rv_type': 'feature'
        })

        # try parse additional info
        rv_row.update(_parse_default_triage_feature_name(feature_full_name))

    else:
        raise ValueError("Failed to match RV regex for RV name: '{}'".format(rv_name))

    return rv_row


# --------------------------------------------------------------------------------------------------------------------
# Metric definitions
# --------------------------------------------------------------------------------------------------------------------


def _unpack_metric_args(metric_args):
    """
    Helper function to unpack metric definition arguments

    :param metric_args: dict()
    :return: pd.DataFrame
    """

    # unpack metric args
    metric_args_ordered = list(metric_args.items())

    # expand cartesian product of all arg combinations in block
    column_names = [metric_args_ordered[ix][0] for ix in range(4)]
    column_vals = list(itertools.product(*[metric_args_ordered[ix][1] for ix in range(4)]))

    # return as table
    return pd.DataFrame.from_records(
        column_vals,
        columns=column_names
    )


def tabulate_metric_defs(mm_config):
    """
    Tabulate metric definition from mm_config.yaml
    See the configuration documentation for notes on how to properly specify metric definitions

    :param mm_config: dict
    :return: dict<str: pd.DataFrame>
    """

    metric_sections = [k for k in mm_config.keys() if '_metrics' in k]
    metric_tables = []

    # for each metric section
    for ix, metric_section in enumerate(metric_sections):

        metric_section_tables = []
        metric_config = mm_config[metric_section]

        # initialize arguments
        default_metric_args = {
            'metric_name': [],
            'compare_interval': [],
            'subset_name': [],
            'subset_threshold': [np.NaN]
        }

        global_metric_args = metric_config['global_metrics']

        # if global settings to override
        if global_metric_args:
            # override them
            default_metric_args.update(global_metric_args)

        # if block keys to parse
        block_keys = [k for k in metric_config.keys() if 'block_' in k]

        if block_keys:

            # apply block key overrides to each argument
            for block_key in block_keys:

                block_metric_args = default_metric_args.copy()
                block_config = metric_config[block_key]
                block_metric_args.update(block_config)

                # check to make sure block arguments are complete
                for k, v in block_metric_args.items():
                    if k != 'subset_threshold':
                        if not v:
                            raise BadConfigurationError(
                                "Missing metric def component: '{}', '{}', '{}'".format(metric_section, block_key, k)
                            )

                # then add block arguments to metric_tables
                metric_section_tables.append(_unpack_metric_args(block_metric_args))

            # concatenate metric section tables and append
            metric_tables.append(pd.concat(metric_section_tables))  # type: pd.DataFrame

        else:
            # only global settings specified- check to make sure config is properly specified
            for k, v in default_metric_args.items():
                if k != 'subset_threshold':
                    if not v:
                        raise BadConfigurationError(
                            "Missing metric def component: '{}', 'global', '{}'".format(metric_section, k)
                        )

            # if so, then append
            metric_tables.append(_unpack_metric_args(global_metric_args))

    # return mapping of random variables to metric tables
    return dict(zip(metric_sections, metric_tables))


# --------------------------------------------------------------------------------------------------------------------
# Distribution metadata
# --------------------------------------------------------------------------------------------------------------------


class BadMetadataError(ValueError):
    """
    Exception class for misconfigured metadata
    """


class DistributionMetadata(object):
    __slots__ = [
        'distribution_metadata_id',
        'is_discrete',
        # preprocessing arguments
        'is_nullable',
        'default_type',
        'default_value',
        'use_default_value_on_unsafe_cast',
        'support_minimum',
        'support_maximum',
        'remove_samples_out_of_support',
        'is_online',
        'warn_on_online_support_change',
        'interpolation_mode',
        # CDF tracking arguments
        'tracking_mode',
        # quantile arguments
        'n_quantiles',
        'n_lower_tail_quantiles',
        'n_upper_tail_quantiles',
        'custom_quantiles',
        # histogram arguments
        'histogram_min',
        'histogram_max',
        'n_histogram_bins',
        'custom_histogram_bins',
        # clustering arguments
        'n_clusters',
        'clustering_algorithm',
        'clustering_algorithm_kwargs',
        'clustering_parametric_family',
        # parametric arguments
        'parametric_family',
    ]

    def __init__(self,
                 is_discrete,
                 is_nullable=False,
                 default_type=None,
                 default_value=None,
                 use_default_value_on_unsafe_cast=None,
                 support_minimum=None,
                 support_maximum=None,
                 remove_samples_out_of_support=None,
                 is_online=None,
                 warn_on_online_support_change=None,
                 interpolation_mode=None,
                 tracking_mode=None,
                 n_quantiles=None,
                 n_lower_tail_quantiles=None,
                 n_upper_tail_quantiles=None,
                 custom_quantiles=None,
                 histogram_min=None,
                 histogram_max=None,
                 n_histogram_bins=None,
                 custom_histogram_bins=None,
                 n_clusters=None,
                 clustering_algorithm=None,
                 clustering_algorithm_kwargs=None,
                 clustering_parametric_family=None,
                 parametric_family=None):
        """
        Constructor

        Validates distribution metadata for illegal configurations before constructing metadata instance.

        :param is_discrete: bool
        :param is_nullable: bool
        :param default_type: str, numpy-castable dtype
        :param default_value: numeric, default
        :param use_default_value_on_unsafe_cast: bool, if cast fails then use default value specified above
        :param support_minimum: numeric
        :param support_maximum: numeric
        :param remove_samples_out_of_support: bool
        :param is_online: bool, allow multiple sample updates
        :param warn_on_online_support_change: bool, warn if online and registered support changes
        :param interpolation_mode: str, one of ['empirical', 'nearest', 'linear', 'pchip']
        :param tracking_mode: str, tracking mode (required if is_discrete = False)
        :param n_quantiles: int, number of quantiles
        :param n_lower_tail_quantiles: int, number of lower tail quantiles (in powers of ten)
        :param n_upper_tail_quantiles: int, number of upper tail quantiles (in powers of ten)
        :param custom_quantiles: list of quantiles
        :param histogram_min: numeric
        :param histogram_max: numeric
        :param n_histogram_bins: int, number of evenly spaced histogram bins
        :param custom_histogram_bins: list of histogram bins
        :param n_clusters: int, number of clusters
        :param clustering_algorithm: str, clustering algorithm
        :param clustering_algorithm_kwargs: dict, clustering algorithm keyword arguments
        :param clustering_parametric_family: str, clustering parametric family name
        :param parametric_family: str, parametric family to fit distribution
        """

        # if default_type specified, check it is numerically tractable
        if default_type:
            try:
                np.array([1]).astype(default_type, casting='safe')
            except TypeError:
                raise BadMetadataError("Default type '{}' either not numeric or not castable".format(default_type))

        # check default value is not None if necessary
        if use_default_value_on_unsafe_cast:
            if not default_value:
                raise BadMetadataError("Default value used on unsafe cast, but no default value specified")

        # check support is numeric and sensible
        if support_minimum and support_maximum:
            if support_minimum > support_maximum:
                raise BadMetadataError("Support minimum larger than support maximum")

        # if filtering on support, check specified
        if remove_samples_out_of_support:
            if not support_minimum and not support_maximum:
                raise BadMetadataError("No support specified, but remove_samples_out_of_support=True")

        # check online support change
        if warn_on_online_support_change:
            if not is_online:
                raise BadMetadataError("Distribution is not online, but warn_on_online_support_change=True")

        # continuous variable setting validation
        if not is_discrete:
            if not tracking_mode:
                raise BadMetadataError("Continuous distribution does not have a tracking mode specified")

            if tracking_mode in ['histogram', 'quantile'] and not interpolation_mode:
                raise BadMetadataError("Nonparametric continuous distribution does not have an interpolation mode")

            # quantile tracking settings
            if tracking_mode == 'quantile':
                # if custom quantiles, check for validity
                if custom_quantiles:
                    try:
                        qs = [float(q) for q in custom_quantiles]
                        for q in qs:
                            if q < 0. or q > 1.:
                                raise BadMetadataError("Custom quantile {} not between 0 and 1".format(q))

                    except (TypeError, ValueError):
                        raise BadMetadataError(
                            "Failed to convert custom quantiles '{}' to array".format(custom_quantiles)
                        )
                else:
                    # else check n_quantiles
                    if not n_quantiles:
                        raise BadMetadataError("Quantile tracking specified, but no quantiles specified")

            # histogram tracking settings
            elif tracking_mode == 'histogram':

                # check bounds
                if not histogram_max and not support_maximum:
                    raise BadMetadataError("Histogram tracking specified, but no upper bound found")

                if not histogram_min and not support_minimum:
                    raise BadMetadataError("Histogram tracking specified, but no lower bound found")

                # if custom bins, check for validity
                if custom_histogram_bins:
                    try:
                        hbs = [float(q) for q in custom_histogram_bins]
                        hmin = histogram_min if histogram_min else support_minimum
                        hmax = histogram_max if histogram_max else support_maximum

                        for hb in hbs:
                            if hb < hmin or hb > hmax:
                                raise BadMetadataError("Custom histogram bin {} not contained in specified support")

                    except (TypeError, ValueError):
                        raise BadMetadataError(
                            "Failed to convert custom histogram bins '{}' to array".format(custom_histogram_bins)
                        )

                else:
                    # else, check n_histogram_bins
                    if not n_histogram_bins:
                        raise BadMetadataError("Histogram tracking specified, but no histogram bins specified")

            # cluster tracking settings
            elif tracking_mode == 'cluster':
                # check necessary parameters
                if not n_clusters:
                    raise BadMetadataError("Cluster tracking specified, but n_clusters unspecified")

                if not clustering_algorithm:
                    raise BadMetadataError("Cluster tracking specified, but n_clusters unspecified")

                # check valid algorithm
                if not hasattr(cluster, clustering_algorithm):
                    raise BadMetadataError("Clustering algorithm '{}' not found".format(clustering_algorithm))

                # check constructor accepts valid arguments
                if not clustering_algorithm_kwargs:
                    clustering_algorithm_kwargs = {'n_clusters': n_clusters}
                else:
                    clustering_algorithm_kwargs.update({'n_clusters': n_clusters})
                try:
                    getattr(cluster, clustering_algorithm)(**clustering_algorithm_kwargs)
                except Exception:
                    tb = traceback.format_exc()
                    raise BadMetadataError("Failed to construct clustering algorithm, exception below: \n\n{}".format(
                        tb
                    ))

            # parametric tracking settings
            elif tracking_mode == 'parametric':
                # check parametric family exists
                if not parametric_family:
                    raise BadMetadataError("Parametrc tracking specified, but parametric_family unspecified")

                # check parametric family is valid
                try:
                    getattr(getattr(stats, parametric_family), 'fit')
                except AttributeError:
                    raise BadMetadataError("Invalid parametric family '{}'".format(parametric_family))

        # construct if final type casting is valid
        try:
            self.is_discrete = bool(is_discrete)
            self.is_nullable = bool(is_nullable)
            self.default_type = str(default_value) if default_value else None
            self.default_value = default_value
            self.use_default_value_on_unsafe_cast = bool(use_default_value_on_unsafe_cast)
            self.support_minimum = support_minimum if support_minimum else None
            self.support_maximum = support_maximum if support_maximum else None
            self.remove_samples_out_of_support = bool(remove_samples_out_of_support)
            self.is_online = bool(is_online)
            self.warn_on_online_support_change = bool(warn_on_online_support_change)
            self.interpolation_mode = interpolation_mode
            self.tracking_mode = str(tracking_mode) if tracking_mode else None
            self.n_quantiles = int(n_quantiles) if n_quantiles else 0
            self.n_lower_tail_quantiles = int(n_lower_tail_quantiles) if n_lower_tail_quantiles else 0
            self.n_upper_tail_quantiles = int(n_upper_tail_quantiles) if n_upper_tail_quantiles else 0
            self.custom_quantiles = [float(q) for q in custom_quantiles] if custom_quantiles else None
            self.histogram_min = histogram_min if histogram_min else None
            self.histogram_max = histogram_max if histogram_max else None
            self.n_histogram_bins = int(n_histogram_bins) if n_histogram_bins else 0
            self.custom_histogram_bins = [float(h) for h in custom_histogram_bins] if custom_histogram_bins else None
            self.n_clusters = int(n_clusters) if n_clusters else 0
            self.clustering_algorithm = clustering_algorithm
            self.clustering_algorithm_kwargs = clustering_algorithm_kwargs
            self.clustering_parametric_family = str(clustering_parametric_family) if clustering_parametric_family else None
            self.parametric_family = parametric_family
        except TypeError as e:
            raise BadMetadataError("Failed to convert parameter '{}'".format(e.args))


def read_distribution_metadata(mm_config):
    """
    Tabulate metric definition from mm_config.yaml
    See the configuration documentation for notes on how to properly specify metric definitions

    :param mm_config: dict
    :return: dict<str: DistributionMetadata>
    """

    metric_sections = [k for k in mm_config.keys() if '_metrics' in k]
    return {ms: DistributionMetadata(**mm_config[ms]['distribution_metadata'])
            for ms in metric_sections}

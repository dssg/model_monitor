import os
import inspect
import itertools
import pandas as pd
from importlib.machinery import SourceFileLoader

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
            'subset_threshold': []
        }

        global_metric_args = metric_config['global']

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
                    if not v:
                        raise BadConfigurationError(
                            "Missing metric def component: '{}', '{}', '{}'".format(metric_section, block_key, k)
                        )

                # then add block arguments to metric_tables
                metric_section_tables.append(_unpack_metric_args(block_metric_args))

            # concatenate metric section tables and append
            metric_tables[ix] = pd.concat(metric_section_tables)  # type: pd.DataFrame

        else:
            # only global settings specified- check to make sure config is properly specified
            for k, v in default_metric_args.items():
                if not v:
                    raise BadConfigurationError(
                        "Missing metric def component: '{}', 'global', '{}'".format(metric_section, k)
                    )

            # if so, then append
            metric_tables[ix] = _unpack_metric_args(global_metric_args)

    # return mapping of random variables to metric tables
    rv_names = [k[:-8] for k in metric_sections]
    return dict(zip(rv_names, metric_tables))

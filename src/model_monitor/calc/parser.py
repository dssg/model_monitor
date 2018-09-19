import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.spatial as spatial

from model_monitor.io.shared import DistributionMetadata

from model_monitor.calc.distribution import distribution_factory
from model_monitor.calc.difference_metrics import empirical_cdf_distance, integrated_lp_cdf_distance, cdf_merge

# independent calculations
_MOMENT_RE = re.compile('^m([0-9])$')
_NTILE_RE = re.compile('^q([0-9]*)$')
_POINT_PDF_RE = re.compile('^pdf_([0-9]*)$')
_POINT_CDF_RE = re.compile('^cdf_([0-9]*)$')

# calculations requiring join on entity ID
_LP_ENTITY_RE = re.compile('^l([0-9])_entity$')
_RANK_CORR_RE = re.compile('^(dcorr|spearman|kendalltau|wilcoxon|mannwhitney)$')
_RANK_CORR_P_RE = re.compile('^(spearman|kendalltau|wilcoxon|mannwhitney)_p$')

# calculations requiring entity subsets
_BOOL_CALC_RE = re.compile('^(hamming|jaccard|russelrao)$')

# calculations requiring empirical CDF
_LP_CDF_EMPIRICAL_RE = re.compile('^L([0-9])_empirical$')
_LP_CDF_LINEAR_RE = re.compile('^L([0-9])_linear$')
_LP_CDF_PCHIP_RE = re.compile('^L([0-9])_pchip$')


def parse_metric_name(metric_name):
    """
    Given metric name, find the correct calculation type and necessary preprocessing requirements, such as estimating
    a CDF or a copula before calculating a difference metric

    :param metric_name: str
    :return: pd.Series
    """

    # state variables
    require_join = False
    require_subset = False
    require_cdf = False

    # overrides for point estimates
    if _MOMENT_RE.match(metric_name):
        calc_class = 'point'
        calc_type = 'moment'
        calc_order = int(_MOMENT_RE.match(metric_name).groups()[0])

    elif _NTILE_RE.match(metric_name):
        calc_class = 'point'
        calc_type = 'ntile'
        calc_order = float('.{}'.format(_NTILE_RE.match(metric_name).groups()[0]))

    elif _POINT_PDF_RE.match(metric_name):
        calc_class = 'point'
        calc_type = 'pdf_point'
        calc_order = int(_POINT_PDF_RE.match(metric_name).groups()[0])

    elif _POINT_CDF_RE.match(metric_name):
        calc_class = 'point'
        calc_type = 'cdf_point'
        calc_order = float(_POINT_CDF_RE.match(metric_name).groups()[0])

    # overrides for entity-joined metrics
    elif _LP_ENTITY_RE.match(metric_name):
        calc_class = 'entity'
        calc_type = 'entity_lp_corr'
        calc_order = int(_LP_ENTITY_RE.match(metric_name).groups()[0])
        require_join = True

    elif _RANK_CORR_RE.match(metric_name) or _RANK_CORR_P_RE.match(metric_name):
        calc_class = 'entity'
        calc_type = metric_name
        calc_order = 0
        require_join = True

    elif _BOOL_CALC_RE.match(metric_name):
        calc_class = 'entity'
        calc_type = metric_name
        calc_order = 0
        require_join = True
        require_subset = True

    # overrides for CDF metrics
    elif _LP_CDF_EMPIRICAL_RE.match(metric_name):
        calc_class = 'distribution'
        calc_type = 'empirical_cdf_diff'
        calc_order = int(_LP_CDF_EMPIRICAL_RE.match(metric_name).groups()[0])
        require_cdf = True

    elif _LP_CDF_LINEAR_RE.match(metric_name):
        calc_class = 'distribution'
        calc_type = 'linear_cdf_diff'
        calc_order = int(_LP_CDF_LINEAR_RE.match(metric_name).groups()[0])
        require_cdf = True

    elif _LP_CDF_PCHIP_RE.match(metric_name):
        calc_class = 'distribution'
        calc_type = 'pchip_cdf_diff'
        calc_order = int(_LP_CDF_PCHIP_RE.match(metric_name).groups()[0])
        require_cdf = True

    else:
        raise ValueError("Invalid metric name '{}'".format(metric_name))

    return pd.Series({
        'calc_class': calc_class,
        'calc_type': calc_type,
        'calc_order': calc_order,
        'require_join': require_join,
        'require_subset': require_subset,
        'require_cdf': require_cdf
    })


def _add_ranks(df):
    """
    Append absolute and percent ranks to existing sample dataframe

    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    df.loc[:, 'rank_abs'] = np.argsort(df['sample_value']) + 1
    df.loc[: 'rank_pct'] = df['rank_abs'] / float(len(df))
    return df


def apply_pointwise_calc(v1, v2, calc_type, calc_order=None):
    """
    Apply pointwise calculation to two unordered 1D vectors

    :param v1: np.array or pd.Series
    :param v2: np.array or pd.Series
    :param calc_type: calculation name
    :param calc_order: numeric parameter passed to different calculations, is nullable
    :return: float
    """
    if calc_type == 'moment':
        return np.mean(np.power(v2, calc_order)) - np.mean(np.power(v1, calc_order))
    elif calc_type == 'ntile':
        return stats.percentileofscore(v2, calc_order * 100) - stats.percentileofscore(v1, calc_order * 100)
    elif calc_type == 'point_pdf':
        return np.mean(np.where(v2 == calc_order, 1, 0)) - np.mean(np.where(v1 == calc_order, 1, 0))
    else:
        return np.mean(np.where(v2 <= calc_order, 1, 0)) - np.mean(np.where(v1 <= calc_order, 1, 0))


def apply_entity_calc(v1, v2, calc_type, calc_order=None):
    """
    Apply pairwise calculation to two entity-joined ordered 1D vectors

    :param v1: np.array or pd.Series
    :param v2: np.array or pd.Series
    :param calc_type: calculation name
    :param calc_order: numeric parameter passed to different calculations, is nullable
    :return: float
    """
    if calc_type == 'entity_lp_norm':
        return np.power(np.sum(np.power(np.abs(v1 - v2), calc_order)), 1. / calc_order)
    elif calc_type == 'dcorr':
        return spatial.distance.correlation(v1, v2)
    else:
        is_p = (calc_type[-2:] == '_p')
        if is_p:
            return getattr(stats, calc_type[:-2])(v1, v2)[1]
        else:
            return getattr(stats, calc_type)[0]


def apply_distribution_calc(d1, d2, calc_type, distibution_metadata, calc_order=None):
    """
    Apply pairwise calculation to two entity-joined ordered 1D vectors

    :param d1: BaseDistribution
    :param d2: BaseDistribution
    :param calc_type: calculation name
    :param calc_order: numeric parameter passed to different calculations, is nullable
    :return: float
    """

    if calc_type == 'empirical_cdf_diff':
        return empirical_cdf_distance(d1.support, d1.cdf_vals, d2.support, d2.cdf_vals, calc_order)

    else:
        # get interpolant functions
        if calc_type == 'linear_cdf_diff':
            f1 = d1.cdf_linear_interpolation()
            f2 = d2.cdf_linear_interpolation()

        else:
            f1 = d1.cdf_pchip_interpolation()
            f2 = d2.cdf_pchip_interpolation()

        # get integration range from merged cdf
        merged_support = cdf_merge(d1.support, d1.cdf_vals, d2.support, d2.cdf_vals)[2]

        return integrated_lp_cdf_distance(f1, f2, merged_support[0], merged_support[-1], calc_order)


def apply_metric_calculation(today_df, compare_df, metric_defs, distribution_metadata):
    """
    Apply metric calculations to two dataframes of sample values

    :param today_df: pd.DataFrame
    :param compare_df: pd.DataFrame
    :param metric_defs: pd.DataFrame
    :param distribution_metadata: NamedTuple
    :return: pd.DataFrame
    """
    requirements_df = metric_defs['metric_name'].apply(parse_metric_name, axis=1)
    collected_metric_dfs = []

    # build distributions (may not calculate CDF if only using preprocessing)
    d1 = distribution_factory(distribution_metadata)
    d2 = distribution_factory(distribution_metadata)

    # stage one: independent pointwise metrics
    point_metric_defs = requirements_df[requirements_df['calc_class'] == 'point']

    if not point_metric_defs.empty:

        # extract and preprocess samples
        v1 = d1._preprocess_values(today_df['sample_value'])
        v2 = d2._preprocess_values(compare_df['sample_value'])

        # apply all metrics
        point_metric_defs.loc[:, 'metric_value'] = point_metric_defs.apply(
            lambda row: apply_pointwise_calc(v1, v2, row['calc_type'], row['calc_order']),
            axis=1
        )
        collected_metric_dfs.append(point_metric_defs)

    # stage two: join metrics
    entity_join_metric_defs = requirements_df[requirements_df['calc_class'] == 'entity']

    if not entity_join_metric_defs.empty:

        # extract and preprocess samples
        joined_df = pd.merge(today_df, compare_df, how='inner', on='entity_id', suffixes=('_t', '_c'))
        v1 = d1._preprocess_values(joined_df['_t'])
        v2 = d2._preprocess_values(joined_df['_c'])

        # apply all metrics
        entity_join_metric_defs.loc[:, 'metric_value'] = point_metric_defs.apply(
            lambda row: apply_entity_calc(v1, v2, row['calc_type'], row['calc_order']),
            axis=1
        )

        collected_metric_dfs.append(entity_join_metric_defs)

    # stage three: distribution metrics
    distribution_metric_defs = requirements_df[requirements_df['calc_class'] == 'distribution']

    if not distribution_metric_defs.empty:

        # update distributions
        d1.reset()
        d2.reset()

        d1.update(today_df['sample_value'])
        d2.update(compare_df['sample_value'])

        # apply all metrics
        distribution_metric_defs.loc[:, 'metric_value'] = distribution_metric_defs.apply(
            lambda row: apply_distribution_calc(d1, d2, row['calc_type'], distribution_metadata, row['calc_order']),
            axis=1
        )

        collected_metric_dfs.append(distribution_metric_defs)

    return pd.concat(distribution_metric_defs)  # type: pd.DataFrame


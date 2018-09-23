import re
import numpy as np
import datetime as dt
import pandas as pd

import model_monitor.io.config_reader as cr
import model_monitor.calc.parser as cp
import model_monitor.calc.loss as cl


def daily_job(mm_config, target_date):
    """
    Daily ETL job for model_monitor

    :param mm_config: dict
    :param target_date: datetime
    :return:
    """
    # ----------------------------------------------------------------------------------------------------------------
    # Stage 1: process config file
    # ----------------------------------------------------------------------------------------------------------------

    # build feature and results extractor
    candidate_results_extractors, candidate_feature_extractors = \
        cr.candidate_extractors()
    results_extractor = \
        cr._results_extractor_factory(mm_config, candidate_results_extractors)  # type: cr.BaseResultsExtractor
    feature_extractor = \
        cr._feature_extractor_factory(mm_config, candidate_feature_extractors)  # type: cr.BaseFeatureExtractor

    # build random variable definitions
    rv_defs = cr.read_rv_defs(mm_config)
    unique_sources = pd.DataFrame(rv_defs.values())[['rv_type', 'is_test']].drop_duplicates()
    rv_evaluation_order = pd.DataFrame(rv_defs).T.sort_values(['rv_type', 'is_test']).index.values

    # built distribution
    distribution_metadata = cr.read_distribution_metadata(mm_config)

    # build metric definitions
    metric_defs = cr.tabulate_metric_defs(mm_config)

    # databasing settings
    is_initial_run = mm_config['runtime_settings']['initial_run']
    database_new_metadata = mm_config['runtime_settings']['database_new_metadata']

    # ----------------------------------------------------------------------------------------------------------------
    # Stage 2: extract shared results data
    # ----------------------------------------------------------------------------------------------------------------

    # select model configuration
    models = results_extractor.select_models()  # type: pd.DataFrame
    train_matrices = results_extractor.select_training_matrices()
    test_matrices = results_extractor.select_testing_matrices()

    # select training and testing matrices
    model_matrix_map = pd.merge(
        pd.merge(
            models[['model_id', 'model_group_id', 'train_matrix_id']],
            train_matrices[['train_matrix_id', 'train_matrix_uuid', 'train_as_of_date']],
            how='inner', on='train_matrix_id'
        ),
        test_matrices[['train_matrix_id', 'test_matrix_id', 'test_matrix_uuid', 'test_as_of_date']],
        how='inner', on='train_matrix_id'
    )

    # select target training and testing matrices
    mm_index_date = mm_config['runtime_settings']['mm_index_date']
    if mm_index_date == 'train':
        model_matrix_map_current = model_matrix_map[model_matrix_map['train_as_of_date'] == target_date]
    else:
        model_matrix_map_current = model_matrix_map[model_matrix_map['test_as_of_date'] == target_date]

    if model_matrix_map_current.empty:
        print("No target models specified. Exiting.")
        exit(0)

    predictions = pd.DataFrame()
    feature_importances = pd.DataFrame()

    # select predictions if used
    prediction_sources = unique_sources[unique_sources['rv_type'].isin(['prediction_raw', 'prediction_at_precision'])]
    if not prediction_sources.empty:
        predictions = results_extractor.select_predictions()

    # select feature importances if used
    feature_sources = unique_sources[unique_sources['rv_type'] == 'feature_importance']
    if not feature_sources.empty:
        feature_importances = results_extractor.select_feature_importances()

    # ----------------------------------------------------------------------------------------------------------------
    # Stage 3: process model and random variable pair
    # ----------------------------------------------------------------------------------------------------------------

    all_collected_metrics = []

    # for each model group
    for model_group_id, model_matrix_map_current_group in model_matrix_map_current.groupby('model_group_id'):

        # for each unique model_id / test_matrix_id pair
        for model_ix, model_row in model_matrix_map_current_group.iterrows():

            # calculate time difference between
            train_test_timedelta = model_row['test_as_of_date'] - model_row['train_as_of_date']

            # for each configured metric
            for metric_section_name, rv_def in rv_defs.items():
                print("Processing metric section '{}'...".format(metric_section_name))

                rv_distribution_metadata = distribution_metadata[metric_section_name]
                rv_metric_defs = metric_defs[metric_section_name]

                # extract target data
                if rv_def['rv_type'] in ['prediction_raw', 'prediction_at_precision']:
                    target_df = predictions[(predictions['model_id'] == model_row['model_id']) &
                                            (predictions['test_matrix_id'] == model_row['test_matrix_id'])]

                    # raw predictions
                    if rv_def['rv_name'] == 'score':
                        target_df = target_df[['entity_id', 'score']].rename({'score': 'sample_value'})
                    elif rv_def['rv_name'] == 'label':
                        target_df = target_df[['entity_id', 'label']].rename({'score': 'sample_value'})
                    else:
                        # estimated predictions
                        precision = float('.{}'.format(rv_defs['rv_name'].split('_')[-1]))
                        target_df['sample_value'] = \
                            cl.PredictionPreprocessor(target_df).predictions_at_precision(precision)

                        target_df = target_df[['entity_id', 'sample_value']]

                elif rv_def['rv_type'] == 'feature':
                    target_matrix_uuid = \
                        model_row['test_matrix_uuid'] if rv_def['is_test'] else model_row['train_matrix_uuid']

                    target_df = feature_extractor.load_feature_by_uuid(target_matrix_uuid, rv_def['rv_name'])
                    target_df.rename(columns={rv_def['rv_name']: 'sample_value'})

                else:
                    target_df = feature_importances[feature_importances['model_id'] == model_row['model_id']]
                    # note: renaming for parsing purposes only
                    target_df = target_df.rename({'feature': 'entity_id',
                                                  'feature_importance': 'sample_value'})

                # extract comparison

                for compare_interval, compare_interval_metric_defs in rv_metric_defs.groupby("compare_interval"):

                    # extract reference dates
                    reference_date = target_date - compare_interval

                    if mm_index_date == 'train':
                        reference_date_train = reference_date
                        reference_date_test = reference_date + train_test_timedelta
                    else:
                        reference_date_train = reference_date - train_test_timedelta
                        reference_date_test = reference_date

                    # select model
                    comparison_model = model_matrix_map[(model_matrix_map['model_group_id'] == model_group_id) &
                                                        (model_matrix_map['test_as_of_date'] == reference_date_test) &
                                                        (model_matrix_map['train_as_of_date'] == reference_date_train)]

                    # if no corresponding model with given time-dependence, continue
                    if comparison_model.empty:
                        continue

                    comparison_model_row = comparison_model.iloc[0]

                    # extract reference data
                    if rv_def['rv_type'] in ['prediction_raw', 'prediction_at_precision']:
                        reference_df = predictions[
                            (predictions['model_id'] == comparison_model_row['model_id']) &
                            (predictions['test_matrix_id'] == comparison_model_row['test_matrix_id'])
                            ]

                        # raw predictions
                        if rv_def['rv_name'] == 'score':
                            reference_df = reference_df[['entity_id', 'score']].rename({'score': 'sample_value'})
                        elif rv_def['rv_name'] == 'label':
                            reference_df = reference_df[['entity_id', 'label']].rename({'score': 'sample_value'})
                        else:
                            # estimated predictions
                            precision = float('.{}'.format(rv_defs['rv_name'].split('_')[-1]))
                            reference_df['sample_value'] = \
                                cl.PredictionPreprocessor(reference_df).predictions_at_precision(precision)

                            reference_df = reference_df[['entity_id', 'sample_value']]

                    elif rv_def['rv_type'] == 'feature':
                        target_matrix_uuid = comparison_model_row['test_matrix_uuid'] \
                            if rv_def['is_test'] else comparison_model_row['train_matrix_uuid']

                        reference_df = feature_extractor.load_feature_by_uuid(target_matrix_uuid, rv_def['rv_name'])
                        reference_df.rename(columns={rv_def['rv_name']: 'sample_value'})

                    else:
                        reference_df = feature_importances[
                            feature_importances['model_id'] == comparison_model_row['model_id']]
                        # note: renaming for parsing purposes only
                        reference_df = reference_df.rename({'feature': 'entity_id',
                                                            'feature_importance': 'sample_value'})

                    for (subset_name, subset_threshold), subset_metrics in \
                            metric_defs[metric_section_name].groupby(['subset_name', 'subset_threshold']):

                        # apply subset calculation if necessary
                        if subset_name != 'all_entities':
                            merged_df = pd.merge(target_df, reference_df,
                                                 how='inner', on='entity_id', suffixes=('_t', '_r'))

                            merged_df.sort_values('sample_values_t',
                                                  ascending=(subset_name == 'bottom_entities'),
                                                  inplace=True)

                            row_cutoff = int(len(merged_df) * subset_threshold) \
                                if subset_threshold < 1. else subset_threshold

                            subset_target_df = merged_df.loc[:row_cutoff, ['entity_id', 'sample_value_t']].rename(
                                columns={'sample_value_t': 'sample_value'}
                            )

                            subset_reference_df = merged_df.loc[:row_cutoff, ['entity_id', 'sample_value_t']].rename(
                                columns={'sample_value_t': 'sample_value'}
                            )

                        else:
                            subset_target_df = target_df.copy()
                            subset_reference_df = reference_df.copy()

                        # apply metrics
                        metric_calc_results = cp.apply_metric_calculation(
                            subset_target_df,
                            subset_reference_df,
                            subset_metrics,
                            distribution_metadata[metric_section_name]
                        )

                        # add auxiliary information
                        metric_calc_results['model_id'] = model_row['model_id']
                        metric_calc_results['metric_section_name'] = metric_section_name
                        metric_calc_results['test_matrix_id'] = model_row['test_matrix_id']

                        all_collected_metrics.append(metric_calc_results)

    all_collected_metrics = pd.concat(all_collected_metrics)  # type: pd.DataFrame

    # ----------------------------------------------------------------------------------------------------------------
    # Stage 4: database results
    # ----------------------------------------------------------------------------------------------------------------

    if database_new_metadata:
        raise NotImplementedError("Need to implement database upload layer")

    return all_collected_metrics

import unittest

import model_monitor.io.config_reader as cr


class TestConfigReader(unittest.TestCase):
    """
    Test class for configuration file parser
    """

    def test_candidate_extractors(self):
        results_extractors, feature_extractors = cr.candidate_extractors()

        for k, v in results_extractors.items():
            self.assertIsInstance(k, str)
            self.assertTrue(issubclass(v, cr.BaseResultsExtractor))

        for k, v in feature_extractors.items():
            self.assertIsInstance(k, str)
            self.assertTrue(issubclass(v, cr.BaseFeatureExtractor))

    def test_parse_triage_feature_name(self):
        valid_name = 'table_id_2y_number_of_events_sum'
        invalid_name = 'this_is_not_a_valid_feature_name'

        # valid name parsing
        valid_parse = cr._parse_default_triage_feature_name(valid_name)
        self.assertEqual(valid_parse['time_agg'], '2y')
        self.assertEqual(valid_parse['agg_func'], 'sum')

        # invalid name parsing
        invalid_parse = cr._parse_default_triage_feature_name(invalid_name)
        self.assertEqual(invalid_parse['time_agg'], '')
        self.assertEqual(invalid_parse['agg_func'], '')

    def test_parse_random_variable_name(self):
        score_config_name = 'score_metrics'
        prediction_at_precision_name = 'prediction_at_precision_10_metrics'
        feature_name = 'feature_table_id_2y_number_of_events_sum_metrics'

        # valid name parsing
        score_parse = cr._parse_random_variable_name(score_config_name)
        self.assertEqual(score_parse['rv_type'], 'prediction_raw')

        pred_parse = cr._parse_random_variable_name(prediction_at_precision_name)
        self.assertEqual(pred_parse['rv_type'], 'prediction_at_precision')

        feat_parse = cr._parse_random_variable_name(feature_name)
        self.assertEqual(feat_parse['rv_type'], 'feature')
        self.assertEqual(feat_parse['source_table'], 'table_id')

    def test_tabulate_metric_defs(self):
        test_config = {'score_metrics': {
            'global_metrics': {'compare_interval': ['1d', '10d']},
            'block_1_metrics': {'subset_name': ['all_entities'],
                                'metric_name': ['spearmanr', 'kendalltau']},
            'block_2_metrics': {'subset_name': ['top_entities'],
                                'metric_name': ['jaccard'],
                                'subset_threshold': [10, .1, .2]}
        }}

        metric_df = cr.tabulate_metric_defs(test_config)['score_metrics']

        self.assertEqual(len(metric_df), 10)
        self.assertEqual(len(metric_df['compare_interval'].unique()), 2)
        self.assertEqual(len(metric_df['subset_threshold'].unique()), 4)


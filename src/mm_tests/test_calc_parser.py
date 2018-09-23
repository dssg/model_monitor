import unittest
import numpy as np
import scipy.stats as stats

import model_monitor.calc.parser as mm_parser


class TestMetricCalculations(unittest.TestCase):
    """
    Test class for metric calculations
    """
    RNG_SEED = 12345

    @classmethod
    def setUpClass(cls):
        """
        Generate random samples
        """
        state = np.random.RandomState(seed=cls.RNG_SEED)
        cls.norm_sample1 = stats.norm.rvs(size=100, random_state=state)
        cls.norm_sample2 = stats.norm.rvs(size=100, random_state=state) + 1.5
        cls.norm_sample1_paired = cls.norm_sample1 * 2. + 2. * (stats.uniform.rvs(size=100, random_state=state) - .5)

        cls.beta_sample1 = stats.beta.rvs(a=1, b=2, size=100, random_state=state)
        cls.beta_sample2 = stats.beta.rvs(a=2, b=3, size=100, random_state=state) + 1.5

    def test_point_calculations(self):
        """
        Test individual point calculations
        """
        self.assertAlmostEqual(
            mm_parser.apply_pointwise_calc(self.norm_sample1, self.norm_sample2, 'moment', 1),
            1.4308026273865455
        )

        self.assertAlmostEqual(
            mm_parser.apply_pointwise_calc(self.norm_sample1, self.norm_sample2, 'central_moment', 2),
            -0.13818511388914345
        )

        self.assertAlmostEqual(
            mm_parser.apply_pointwise_calc(self.norm_sample1, self.norm_sample2, 'ntile', .50),
            1.1739410727342987
        )

        self.assertAlmostEqual(
            mm_parser.apply_pointwise_calc(self.norm_sample1, self.norm_sample2, 'point_cdf', 0.),
            -.35
        )

    def test_entity_calculations(self):
        """
        Test entity paired calculations
        """
        self.assertAlmostEqual(
            mm_parser.apply_entity_calc(self.norm_sample1,
                                        self.norm_sample1_paired,
                                        'spearmanr', 0),
            .9690009000900089
        )

        self.assertAlmostEqual(
            mm_parser.apply_entity_calc(self.norm_sample1,
                                        self.norm_sample1_paired,
                                        'cosine'),
            .03022202242080929
        )

        self.assertAlmostEqual(
            mm_parser.apply_entity_calc(self.norm_sample1,
                                        self.norm_sample1_paired,
                                        'dcorr'),
            .029859863434495537
        )

    def test_cdf_calculations(self):
        """
        Test CDF integral difference calculations
        """
        beta_dm = mm_parser.DistributionMetadata(
            is_discrete=False,
            tracking_mode='parametric',
            parametric_family='beta')

        beta_dist1 = mm_parser.distribution_factory(beta_dm)
        beta_dist2 = mm_parser.distribution_factory(beta_dm)

        beta_dist1.update(self.beta_sample1)
        beta_dist2.update(self.beta_sample2)

        self.assertAlmostEqual(
            mm_parser.apply_distribution_calc(beta_dist1, beta_dist2, 'lp_cdf', beta_dm, 1),
            1.5615213992799222
        )


class TestApplyMetricCalculations(unittest.TestCase):
    """
    Test class for applying metric calculations
    """

    def test_parse_point_metric_names(self):
        """
        Test name parsing for point metrics
        """
        self.assertEqual(mm_parser.parse_metric_name('moment_1')['calc_order'], 1)
        self.assertAlmostEqual(mm_parser.parse_metric_name('q_95')['calc_order'], .95)

    def test_parse_entity_metric_name(self):
        """
        Test name parsing for entity-joined metrics
        """
        # case-based p norms map names correctly
        self.assertEqual(mm_parser.parse_metric_name('euclidean')['calc_order'], 2)

        # calc order for tests returns statistic at index 0 and p-value at index 1
        self.assertEqual(mm_parser.parse_metric_name('spearmanr')['calc_order'], 0)
        self.assertEqual(mm_parser.parse_metric_name('spearmanr_p')['calc_order'], 1)

    def test_parse_cdf_metric_name(self):
        """
        Test name parsing for CDF metrics
        """
        # case based p norms for cdf maps
        self.assertEqual(mm_parser.parse_metric_name('cvm_cdf')['calc_order'], 2)
        self.assertEqual(mm_parser.parse_metric_name('L_2_cdf_inv')['calc_order'], 2)

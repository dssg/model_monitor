import unittest
import numpy as np
import scipy.stats as stats

import model_monitor.calc.distribution as mm_dist
import model_monitor.io.config_reader


class TestDistributionMetadata(unittest.TestCase):
    """
    Test class for distribution metadata parsing
    """

    def test_ctor_default_type_check(self):
        """
        Test default type checking
        """
        # valid default types
        model_monitor.io.config_reader.DistributionMetadata(is_discrete=True, default_type='int')

        # invalid default types
        with self.assertRaises(model_monitor.io.config_reader.BadMetadataError):
            model_monitor.io.config_reader.DistributionMetadata(is_discrete=True, default_type='invalid')

    def test_ctor_custom_quantile_parsing(self):
        """
        Test quantile parsing
        """
        valid_quantiles = [.01, .02, .03, .04]
        invalid_range_quantiles = [-1, .02, .3]
        nonnumeric_quantiles = [.01, .02, 'wrong']

        # valid quantiles
        model_monitor.io.config_reader.DistributionMetadata(is_discrete=False,
                                                            tracking_mode='quantile',
                                                            interpolation_mode='linear',
                                                            custom_quantiles=valid_quantiles)

        # invalid quantiles
        with self.assertRaises(model_monitor.io.config_reader.BadMetadataError):
            model_monitor.io.config_reader.DistributionMetadata(is_discrete=False,
                                                                tracking_mode='quantile',
                                                                interpolation_mode='linear',
                                                                custom_quantiles=invalid_range_quantiles)

        with self.assertRaises(model_monitor.io.config_reader.BadMetadataError):
            model_monitor.io.config_reader.DistributionMetadata(is_discrete=False,
                                                                tracking_mode='quantile',
                                                                interpolation_mode='linear',
                                                                custom_quantiles=nonnumeric_quantiles)

    def test_ctor_cluster_algorithm_ctor(self):
        """
        Test cluster algorithm constructor validity
        """
        valid_class = 'KMeans'
        invalid_class = 'not_valid'
        valid_args = {'init': 'k-means++'}
        invalid_args = {'not_valid': 'still_not_valid'}

        # valid cluster names and arguments should be parsed succesfully
        model_monitor.io.config_reader.DistributionMetadata(is_discrete=False,
                                                            tracking_mode='cluster',
                                                            clustering_algorithm=valid_class,
                                                            n_clusters=10)
        model_monitor.io.config_reader.DistributionMetadata(is_discrete=False,
                                                            tracking_mode='cluster',
                                                            clustering_algorithm=valid_class,
                                                            clustering_algorithm_kwargs=valid_args,
                                                            n_clusters=10)

        # invalid cluster names or arguments should raise an error
        with self.assertRaises(model_monitor.io.config_reader.BadMetadataError):
            model_monitor.io.config_reader.DistributionMetadata(is_discrete=False,
                                                                tracking_mode='cluster',
                                                                clustering_algorithm=invalid_class,
                                                                n_clusters=10)

        with self.assertRaises(model_monitor.io.config_reader.BadMetadataError):
            model_monitor.io.config_reader.DistributionMetadata(is_discrete=False,
                                                                tracking_mode='cluster',
                                                                clustering_algorithm=valid_class,
                                                                clustering_algorithm_kwargs=invalid_args,
                                                                n_clusters=10)

    def test_ctor_parametric_family(self):
        """
        Test parametric family constructor validity
        """

        # valid parametric families construct as expected
        model_monitor.io.config_reader.DistributionMetadata(is_discrete=False, tracking_mode='parametric', parametric_family='norm')

        # invalid parametric family should raise an error
        with self.assertRaises(model_monitor.io.config_reader.BadMetadataError):
            model_monitor.io.config_reader.DistributionMetadata(is_discrete=False, tracking_mode='parametric', parametric_family='wrong')

    def test_preprocessing_null_handling(self):
        """
        Test preprocessing null handling
        """
        # null handling
        null_dm = model_monitor.io.config_reader.DistributionMetadata(is_discrete=True, is_nullable=True)
        null_dd = mm_dist.DiscreteDistribution(null_dm)
        null_dd.update([1., 2., 3., np.NaN])
        self.assertAlmostEqual(null_dd.null_proportion(), .25)

        # unhandled nulls
        with self.assertWarns(mm_dist.NullableSampleWarning):
            nonnull_dm = model_monitor.io.config_reader.DistributionMetadata(is_discrete=True, is_nullable=False)
            nonnull_dd = mm_dist.DiscreteDistribution(nonnull_dm)
            nonnull_dd.update([1., 2., 3., np.NaN])

    def test_preprocessing_type_casting(self):
        """
        Test preprocessing type casting
        """
        # normal casting
        cast_dm = model_monitor.io.config_reader.DistributionMetadata(is_discrete=True, default_type='float')
        cast_dd = mm_dist.DiscreteDistribution(cast_dm)
        cast_dd.update([1, 2, 3])

    def test_preprocessing_support_handling(self):
        """
        Test preprocessing support handling
        """
        # remove values outside support
        remove_dm = model_monitor.io.config_reader.DistributionMetadata(is_discrete=True,
                                                                        support_maximum=2,
                                                                        remove_samples_out_of_support=True)

        with self.assertWarns(mm_dist.SampleOutsideSupportWarning):
            dd = mm_dist.DiscreteDistribution(remove_dm)
            dd.update([1, 2, 3, 4])
            self.assertEquals(dd.sample_size, 2)

        # keep values outside support
        keep_dm = model_monitor.io.config_reader.DistributionMetadata(is_discrete=True,
                                                                      support_maximum=2)

        with self.assertWarns(mm_dist.SampleOutsideSupportWarning):
            dd = mm_dist.DiscreteDistribution(keep_dm)
            dd.update([1, 2, 3, 4])
            self.assertEquals(dd.sample_size, 4)


class TestDiscreteDistribution(unittest.TestCase):
    """
    Test class for discrete distributions
    """

    @classmethod
    def setUpClass(cls):
        """
        Create offline and online discrete distributions
        """
        cls.offline_dist = mm_dist.DiscreteDistribution(model_monitor.io.config_reader.DistributionMetadata(
            is_discrete=True, is_online=False))

        cls.online_dist = mm_dist.DiscreteDistribution(model_monitor.io.config_reader.DistributionMetadata(
            is_discrete=True, is_online=True))

    def tearDown(self):
        """
        Reset existing distributions after each test
        """
        self.online_dist.reset()
        self.offline_dist.reset()

    def test_update_offline(self):
        """
        Test one shot updates of discrete distributions
        """
        # offline first update
        self.offline_dist.update([1, 1, 2, 2, 3, 3, 4, 4])
        np.testing.assert_almost_equal(self.offline_dist.cdf_vals,
                                       np.array([.25, .5, .75, 1.]))

        # offline second update
        with self.assertWarns(mm_dist.OfflineReinitializationWarning):
            self.offline_dist.update([1, 2])

    def test_update_online(self):
        """
        Test online discrete distribution estimation
        """
        # online first update
        self.online_dist.update([1, 2, ])
        np.testing.assert_almost_equal(self.online_dist.cdf_vals,
                                       np.array([.5, 1.]))

        # offline second update
        self.online_dist.update([1, 2, 3, 3, 4, 4])
        np.testing.assert_almost_equal(self.online_dist.cdf_vals,
                                       np.array([.25, .5, .75, 1.]))


class TestContinuousDistributionQuantile(unittest.TestCase):
    """
    Test class for quantile-tracked continuous distributions
    """

    RNG_SEED = 12345

    @classmethod
    def setUpClass(cls):
        """
        Create online and offline quantile-tracked distributions
        """
        state = np.random.RandomState(seed=cls.RNG_SEED)
        cls.sample1 = stats.norm.rvs(size=1000, random_state=state)
        cls.sample2 = stats.norm.rvs(size=1000, random_state=state)
        cls.cd_offline = mm_dist.ContinuousDistribution(model_monitor.io.config_reader.DistributionMetadata(
            is_discrete=False,
            tracking_mode='quantile',
            n_quantiles=5,
            n_upper_tail_quantiles=2,
            interpolation_mode='linear'
        ))

        cls.cd_online = mm_dist.ContinuousDistribution(model_monitor.io.config_reader.DistributionMetadata(
            is_discrete=False,
            tracking_mode='quantile',
            n_quantiles=5,
            n_upper_tail_quantiles=2,
            interpolation_mode='linear',
            is_online=True
        ))

    def tearDown(self):
        """
        Reset existing distributions after each test
        """
        self.cd_offline.reset()
        self.cd_online.reset()

    def test_quantile_args(self):
        """
        Test quantile arg parsing
        """
        np.testing.assert_almost_equal(self.cd_offline._quantiles,
                                       np.array([0., 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.]))

    def test_quantile_offline_calc(self):
        """
        Test quantile offline calculation (single shot calc)
        """
        self.cd_offline.update(self.sample1)
        # test CDF values
        np.testing.assert_almost_equal(
            self.cd_offline.support,
            np.array([-2.9493435, -0.86745457, -0.30174412, 0.23676523,
                      0.82787685, 1.29816155, 2.21242626, 3.92752804])
        )

        # test interpolations
        self.assertAlmostEqual(self.cd_offline._cdf_linear_interpolation()(0.0),
                               .5120664)

        self.assertAlmostEqual(self.cd_offline._cdf_pchip_interpolation()(0.0),
                               .51280366)

    def test_quantile_online_calc(self):
        """
        Test quantile online calculation using P2 algorithm
        """
        self.cd_online.update(self.sample1)
        self.cd_online.update(self.sample2)

        np.testing.assert_almost_equal(
            self.cd_online.support,
            np.array([-3.14765054, -0.87532419, -0.26926606, 0.24762581,
                      0.86820621, 1.31355589, 2.31653485, 3.23330354])
        )

        # test interpolations
        self.assertAlmostEqual(self.cd_online._cdf_linear_interpolation()(0.0),
                               .50418661058)

        self.assertAlmostEqual(self.cd_online._cdf_pchip_interpolation()(0.0),
                               .50463885933)


class TestContinuousDistributionHistogram(unittest.TestCase):
    """
    Test class for histogram-tracked continuous distributions
    """
    RNG_SEED = 23456

    @classmethod
    def setUpClass(cls):
        """
        Create online and offline histogram-tracked distributions
        """
        state = np.random.RandomState(seed=cls.RNG_SEED)
        cls.sample1 = stats.norm.rvs(size=1000, random_state=state)
        cls.sample2 = stats.norm.rvs(size=1000, random_state=state)
        cls.cd_offline = mm_dist.ContinuousDistribution(model_monitor.io.config_reader.DistributionMetadata(
            is_discrete=False,
            tracking_mode='histogram',
            histogram_min=-2,
            histogram_max=2,
            n_histogram_bins=8,
            interpolation_mode='linear'
        ))

        cls.cd_online = mm_dist.ContinuousDistribution(model_monitor.io.config_reader.DistributionMetadata(
            is_discrete=False,
            tracking_mode='histogram',
            histogram_min=-2,
            histogram_max=2,
            n_histogram_bins=8,
            interpolation_mode='linear',
            is_online=True
        ))

    def tearDown(self):
        """
        Reset existing distributions after each test
        """
        self.cd_offline.reset()
        self.cd_online.reset()

    def test_histogram_args(self):
        """
        Test histogram argument parsing
        """
        np.testing.assert_almost_equal(
            self.cd_offline._histogram_bins,
            np.array([-2., - 1.5, - 1., - 0.5, 0., 0.5, 1., 1.5, 2.])
        )

    def test_histogram_offline(self):
        """
        Test offline calc
        """
        self.cd_offline.update(self.sample1)

        np.testing.assert_almost_equal(
            self.cd_offline.cdf_vals,
            np.array([0.0, 0.03688093, 0.14646997, 0.28451001, 0.51211802,
                      0.69546891, 0.85353003, 0.95890411, 1.])
        )

    def test_histogram_online(self):
        """
        Test online calc
        """
        self.cd_online.update(self.sample1)
        self.cd_online.update(self.sample2)

        np.testing.assert_almost_equal(
            self.cd_online.cdf_vals,
            np.array([0., 0.04094488, 0.14330709, 0.28661417,
                      0.50498688, 0.69553806, 0.85669291, 0.95643045, 1.])
        )


class TestContinuousDistributionParametricEstimate(unittest.TestCase):
    """
    Test class for parametric distribution estimation
    """
    RNG_SEED = 34567

    @classmethod
    def setUpClass(cls):
        """
        Create random samples and distribution classes
        """
        state = np.random.RandomState(seed=cls.RNG_SEED)
        cls.norm_sample = stats.norm.rvs(size=1000, random_state=state)
        cls.beta_sample = stats.beta.rvs(size=1000, a=2, b=3, random_state=state)

        cls.cd_norm = mm_dist.ContinuousDistribution(model_monitor.io.config_reader.DistributionMetadata(
            is_discrete=False,
            tracking_mode='parametric',
            parametric_family='norm'
        ))

        cls.cd_beta = mm_dist.ContinuousDistribution(model_monitor.io.config_reader.DistributionMetadata(
            is_discrete=False,
            tracking_mode='parametric',
            parametric_family='beta'
        ))

    def tearDown(self):
        """
        Reset existing distributions after each test
        """
        self.cd_norm.reset()
        self.cd_beta.reset()

    def test_parametric_estimate_loc_scale(self):
        """
        Test distribution estimates that only depend on location-scale parameters
        """
        self.cd_norm.update(self.norm_sample)

        params = self.cd_norm._parametric_args
        self.assertAlmostEqual(params['loc'], -0.008346244905323466)
        self.assertAlmostEqual(params['scale'], 0.9466261033408847)

    def test_parametric_estimate_shape(self):
        """
        Test distribution estimates that depend on shape parameters
        """
        self.cd_beta.update(self.beta_sample)

        params = self.cd_beta._parametric_args
        self.assertAlmostEqual(params['a'], 2.0678013326644766)
        self.assertAlmostEqual(params['b'], 3.090479591572537)

    def test_parametric_cdf(self):
        """
        Test CDF construction and evaluation
        """
        self.cd_norm.update(self.norm_sample)
        self.assertAlmostEqual(self.cd_norm.cdf(0.0), 0.5035173621607916)


class TestContinuousDistributionCluster(unittest.TestCase):
    """
    Test class for cluster-tracked distributions
    """

    RNG_SEED = 45678
    ALGO_RNG_SEED = 56789

    @classmethod
    def setUpClass(cls):
        """
        Create mixture distributions with different CDF generation methods
        """
        state = np.random.RandomState(seed=cls.RNG_SEED)
        cls.norm_mix = np.concatenate([
            stats.norm.rvs(size=1000, random_state=state),
            stats.norm.rvs(size=1000, random_state=state) + 1.5
        ])

        cls.cd_mix_nonparam = mm_dist.ContinuousDistribution(model_monitor.io.config_reader.DistributionMetadata(
            is_discrete=False,
            tracking_mode='cluster',
            interpolation_mode='empirical',
            n_clusters=2,
            clustering_algorithm='KMeans'
        ))

        cls.cd_mix_param = mm_dist.ContinuousDistribution(model_monitor.io.config_reader.DistributionMetadata(
            is_discrete=False,
            tracking_mode='cluster',
            n_clusters=2,
            clustering_algorithm='KMeans',
            clustering_parametric_family='norm'
        ))

    def tearDown(self):
        """
        Reset existing distributions after each test
        """
        self.cd_mix_nonparam.reset()

    def test_nonparametric_cluster_fit(self):
        """
        Test nonparametric mixture distribution
        """
        self.cd_mix_nonparam.update(self.norm_mix,
                                    test_state=np.random.RandomState(seed=self.ALGO_RNG_SEED))

        np.testing.assert_almost_equal(
            self.cd_mix_nonparam.support,
            np.array([-0.2405309, 1.77324558])
        )

    def test_parametric_cluster_fit(self):
        """
        Test parametric mixture distribution
        """
        self.cd_mix_param.update(self.norm_mix,
                                 test_state=np.random.RandomState(seed=self.ALGO_RNG_SEED))

        self.assertAlmostEqual(self.cd_mix_param._cluster_parametric_args[0]['loc'],
                               1.7742694469032563)

        self.assertAlmostEqual(self.cd_mix_param.cdf(0),
                               0.3228527274992709)

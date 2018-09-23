import warnings
from functools import partial
from abc import abstractmethod, ABC, abstractproperty
import numpy as np
import scipy.stats as stats
import scipy.interpolate as interpolate
import sklearn.cluster as cluster

from model_monitor.io.config_reader import DistributionMetadata
from model_monitor.report.shared import (
    NullableSampleWarning,
    SampleOutsideSupportWarning,
    UnsafeCastingWarning,
    OnlineSupportChangeWarning,
    OfflineReinitializationWarning
)


def _can_cast(val, dtype):
    """
    Generic cast checking function

    :param val: object
    :param dtype: numpy dtype
    :return: bool
    """
    try:
        return np.can_cast(val, dtype, 'safe')
    except TypeError:
        return False


# vectorized version of _can_cast
can_cast = np.vectorize(_can_cast)


class BaseDistribution(ABC):
    """
    Base distribution to handle CDF estimation
    """

    def __init__(self, distribution_metadata):
        """
        Constructor

        :param distribution_metadata: DistributionMetadata
        """
        assert isinstance(distribution_metadata, DistributionMetadata)
        self.metadata = distribution_metadata
        self.sample_size = 0
        self.null_size = 0
        self.support = None
        self.pdf_vals = None
        self.cdf_vals = None

    def _preprocess_values(self, new_values):
        """
        Preprocess new values, updating exluded size

        :param new_values: np.array or pd.Series, raw samples
        :return: np.array
        """
        vals = np.array(new_values)

        # nullability
        mask = np.isnan(vals)

        if self.metadata.is_nullable:
            self.null_size += np.sum(mask)
        elif np.isnan(new_values).any():
            warnings.warn(
                "Nullable sample found but is_nullable=False",
                NullableSampleWarning
            )

        nvals = vals[~mask]

        # type casting
        if not self.metadata.default_type:
            tvals = nvals
        else:
            cast_mask = can_cast(nvals)

            # if all values safely cast
            if cast_mask.all():
                tvals = nvals.astype(self.metadata.default_type)

            else:
                # some values cannot be safely cast
                warnings.warn("Sample(s) fail safe casting",
                              UnsafeCastingWarning)
                # if can impute, then impute
                if self.metadata.default_value and self.metadata.use_default_value_on_unsafe_cast:
                    default_vals = np.where(
                        cast_mask,
                        np.repeat(self.metadata.default_value, len(nvals)).astype(self.metadata.default_type),
                        nvals)
                    tvals = default_vals.astype(self.metadata.default_type)

                # else drop values
                else:
                    self.null_size += np.sum(~cast_mask)
                    tvals = nvals[cast_mask]

        # support checking
        if self.metadata.support_maximum and tvals.max() > self.metadata.support_maximum:
            warnings.warn("Sample(s) outside support maximum",
                          SampleOutsideSupportWarning)
            if self.metadata.remove_samples_out_of_support:
                tvals = tvals[tvals <= self.metadata.support_maximum]
                self.null_size += np.sum(tvals > self.metadata.support_maximum)

        if self.metadata.support_minimum and tvals.min() < self.metadata.support_minimum:
            warnings.warn("Sample(s) outside support minimum",
                          SampleOutsideSupportWarning)
            if self.metadata.remove_samples_out_of_support:
                tvals = tvals[tvals >= self.metadata.support_minimum]
                self.null_size += np.sum(tvals < self.metadata.support_minimum)

        self.sample_size += len(tvals)
        return tvals

    def null_proportion(self):
        """
        Proportion of nulls in the sample

        :return: float
        """
        return self.null_size / float(self.sample_size + self.null_size) if self.sample_size > 0 else 0.

    @abstractmethod
    def update(self, new_values):
        """
        Abstract method for updating distribution with new sample

        :param new_values: np.array or pd.Series
        :return: None
        """
        pass

    @abstractproperty
    def cdf(self):
        """
        Abstract property for generating CDF

        :return: function
        """
        pass

    @abstractproperty
    def cdf_inv(self):
        """
        Abstract property for generating CDF

        :return: function
        """
        pass

    def _default_cdf_interpolation_mapping(self, inv=False):
        """
        If using default CDF interpolation, apply interpolation arguments

        :param inv: bool, if True, reverses support and cdf values
        :return: function
        """
        if self.metadata.interpolation_mode == 'empirical':
            return self._cdf_empirical_interpolation(inv=inv)
        elif self.metadata.interpolation_mode == 'nearest':
            return self._cdf_nearest_interpolation(inv=inv)
        elif self.metadata.interpolation_mode == 'linear':
            return self._cdf_linear_interpolation(inv=inv)
        else:
            return self._cdf_pchip_interpolation(inv=inv)

    def _cdf_empirical_interpolation(self, inv=False):
        """
        Default CDF empirical interpolation function

        :param inv: bool, if True, reverses support and cdf values
        :return: function
        """
        if inv:
            return interpolate.interp1d(self.cdf_vals, self.support, kind='previous')
        return interpolate.interp1d(self.support, self.cdf_vals, kind='next')

    def _cdf_nearest_interpolation(self, inv=False):
        """
        Default CDF empirical nearest neighbor interpolation

        :param inv: bool, if True, reverses support and cdf values
        :return: function
        """
        if inv:
            return interpolate.interp1d(self.cdf_vals, self.support, kind='nearest')
        return interpolate.interp1d(self.support, self.cdf_vals, kind='nearest')

    def _cdf_linear_interpolation(self, inv=False):
        """
        Default CDF linear interpolation function

        :param inv: bool, if True, reverses support and cdf values
        :return: function
        """
        if inv:
            return interpolate.interp1d(self.cdf_vals, self.support, kind='linear')
        return interpolate.interp1d(self.support, self.cdf_vals, kind='linear')

    def _cdf_pchip_interpolation(self, inv=False):
        """
        Default CDF cubic monotonic (piecewise cubic hermite interpolating polynomial) interpolation function

        :param inv: bool, if True, reverses support and cdf values
        :return: function
        """
        if inv:
            return interpolate.PchipInterpolator(self.cdf_vals, self.support)
        return interpolate.PchipInterpolator(self.support, self.cdf_vals)

    def reset(self):
        """
        Clear all data from distribution, keeping metadata in place

        :return: None
        """
        self.sample_size = 0
        self.null_size = 0
        self.support = None
        self.pdf_vals = None
        self.cdf_vals = None


class DiscreteDistribution(BaseDistribution):
    """
    CDF estimator class for discrete distributions

    This class dynamically tracks all unique samples (assumed to be significantly smaller than sample size), which
    allows for exact CDF form estimates.
    """

    def __init__(self, distribution_metadata):
        """
        Constructor

        :param distribution_metadata: DistributionMetadata
        """
        BaseDistribution.__init__(self, distribution_metadata)

    def update(self, new_values):
        """
        Update distribution with new values
        
        :param new_values: np.array or pd.Series
        :return: None
        """

        # preprocess values and update based on original sample size
        old_sample_size = self.sample_size
        insert_vals = self._preprocess_values(new_values)

        # return counts and sort by support
        unique_vals, unique_counts = np.unique(insert_vals, return_counts=True)
        val_sort_ix = np.argsort(unique_vals)
        unique_vals = unique_vals[val_sort_ix]
        unique_counts = unique_counts[val_sort_ix]

        # initialize support if not already present
        if not isinstance(self.support, np.ndarray):
            self.support = np.copy(unique_vals)
            self.pdf_vals = np.zeros(len(self.support))
            self.cdf_vals = np.zeros(len(self.support))

        else:
            # if support already exists, check for duplicate values:
            if self.metadata.is_online:
                # check if support varies with new sample
                new_support_values = ~np.in1d(unique_vals, self.support)  # type: np.ndarray
                if new_support_values.any():
                    if self.metadata.warn_on_online_support_change:
                        warnings.warn("New support values detected in sample",
                                      OnlineSupportChangeWarning)
                    # update support
                    self.support = np.concatenate([self.support, unique_vals[new_support_values]])
                    self.pdf_vals = np.concatenate([self.pdf_vals, np.repeat(0, len(unique_vals[new_support_values]))])

                    # reorder pdf for cdf calculation
                    new_support_order = np.argsort(self.support)
                    self.support = self.support[new_support_order]
                    self.pdf_vals = self.pdf_vals[new_support_order]

            else:
                warnings.warn("Attempted multiple initialization of distribution",
                              OfflineReinitializationWarning)

        # update pdf
        support_matches = np.in1d(self.support, unique_vals)
        match_indices = np.where(support_matches, np.cumsum(support_matches), 0)
        match_counts = unique_counts[match_indices - 1]
        self.pdf_vals *= old_sample_size / float(self.sample_size)
        self.pdf_vals += np.where(support_matches, match_counts / float(self.sample_size), 0.)

        # update cdf
        self.cdf_vals = np.cumsum(self.pdf_vals)

    @property
    def cdf(self):
        """
        Empirical CDF property

        :return: function
        """
        return self._cdf_empirical_interpolation()

    @property
    def cdf_inv(self):
        """
        Empirical CDF inverse property

        :return: function
        """
        return self._cdf_empirical_interpolation(inv=True)


class ContinuousDistribution(BaseDistribution):
    """
    CDF estimator class for continuous distributions
    
    This class requires a quantization method, specified in the associated distribution metadata
    """

    def __init__(self, distribution_metadata):
        """
        Constructor

        :param distribution_metadata: DistributionMetadata
        """
        BaseDistribution.__init__(self, distribution_metadata)

        if not self.metadata.is_online:
            self._offline_initialized = False

        # ------------------------------------------------------------------------------------------------------------
        # quantile estimates
        # ------------------------------------------------------------------------------------------------------------

        if self.metadata.tracking_mode == 'quantile':

            if self.metadata.custom_quantiles:
                self._quantiles = np.array(self.metadata.custom_quantiles)

            else:
                # initialize target quantiles
                self._quantiles = np.linspace(0., 1., self.metadata.n_quantiles + 1)

                # add tail quantiles if configured
                if self.metadata.n_lower_tail_quantiles > 0:
                    new_quantiles = np.power(10., -np.array(range(1, self.metadata.n_upper_tail_quantiles + 1)))
                    print('before tail update')
                    print(self._quantiles)
                    self._quantiles = np.union1d(self._quantiles, new_quantiles)

                if self.metadata.n_upper_tail_quantiles > 0:
                    new_quantiles = 1. - np.power(10., -np.array(range(1, self.metadata.n_upper_tail_quantiles + 1)))
                    self._quantiles = np.union1d(self._quantiles, new_quantiles)

            self._quantile_values = np.zeros(len(self._quantiles))

            # initialize P2 algorithm components if online
            if self.metadata.is_online:
                self.__quantile_tracks = np.array([[0., q / 2, q, (1 + q) / 2, 1.] for q in self._quantiles])
                self.__quantile_orders = np.array([[1., 1 + 2 * q, 1 + 4 * q, 3 + 2 * q, 5.] for q in self._quantiles])
                self.__quantile_pos = np.array([range(1, 6) for _ in self._quantiles])
                self.__quantile_heights = np.zeros((len(self._quantiles), 5)) * np.NaN

                self.__initial_q_placeholder = np.zeros(5) * np.NaN
                self.__p2_initialized = False

        # ------------------------------------------------------------------------------------------------------------
        # histogram estimates
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'histogram':

            effective_min = \
                self.metadata.histogram_min if self.metadata.histogram_min else self.metadata.support_minimum
            effective_max = \
                self.metadata.histogram_max if self.metadata.histogram_max else self.metadata.support_maximum

            if self.metadata.custom_histogram_bins:
                self._histogram_bins = np.array(self.metadata.custom_histogram_bins)
            else:
                self._histogram_bins = np.linspace(effective_min,
                                                   effective_max,
                                                   num=self.metadata.n_histogram_bins + 1)

            self._hist_values = np.zeros(len(self._histogram_bins))

        # ------------------------------------------------------------------------------------------------------------
        # parametric estimates
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'parametric':

            self._parametric_args = None
            self._parametric_cdf = None
            self._parametric_cdf_inv = None

        # ------------------------------------------------------------------------------------------------------------
        # cluster estimates
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'cluster':

            self._cluster_model = None
            self._cluster_parametric_args = None
            self._cluster_parametric_cdf = None

        else:
            raise ValueError("Must track quantile, histogram, clusters, or parametric estimate of distribution")

    @staticmethod
    def mle_fit_argmap(samples, parametric_family):
        """
        Map scipy.stats.fit parameters to true labels

        Note that scipy.stats does NOT handle the argument mapping, and arguments vary based on loc and scale families

        :param samples: np.array
        :param parametric_family: str, family name
        :return: dict
        """
        # fit distribution
        dist = getattr(stats, parametric_family)
        dist_params = dist.fit(samples)
        dist_param_names = dist.shapes

        # if params only characterized by location or scale
        if not dist_param_names:
            # if parameterized by location and scale
            if len(dist_params) == 2:
                parametric_args = {'loc': dist_params[0],
                                         'scale': dist_params[1]}
            else:
                parametric_args = {'loc': dist_params[0]}

        # else if there are family-specific parameters
        else:
            dist_param_names = dist_param_names.split(', ')

            parametric_args = dict(zip(dist_param_names, dist_params))

            # if missing loc parameter
            if len(dist_params) - len(dist_param_names) == 1:
                parametric_args.update({'loc': dist_params[-1]})

            # if missing loc and scale parameter
            elif len(dist_params) - len(dist_param_names) == 2:
                parametric_args.update({'loc': dist_params[-2],
                                              'scale': dist_params[-1]})

        # apply corrected mapping to distribution to generate cdf
        return parametric_args

    def update(self, new_values, test_state=None):
        """
        Add new values

        :param new_values: np.array or pd.Series
        :param test_state: np.random.RandomState, used for testing reproducibility only
        """

        if not self.metadata.is_online and self._offline_initialized:
            warnings.warn("Failed attempt at multiple initialization for offline distribution",
                          OfflineReinitializationWarning)
            pass

        # preprocess values for insert
        vals = self._preprocess_values(new_values)

        # ------------------------------------------------------------------------------------------------------------
        # quantile estimates
        # ------------------------------------------------------------------------------------------------------------

        if self.metadata.tracking_mode == 'quantile':

            # online estimate: P2 algorithm - dynamic tracking of quantiles with unknonwn support
            if self.metadata.is_online:

                # if not initialized yet
                if not self.__p2_initialized:
                    # append values until 5 in place
                    remaining_values = np.sum(np.isnan(self.__initial_q_placeholder))

                    # if enough to fully initialize
                    if remaining_values <= len(vals):

                        # initialize quantiles
                        self.__initial_q_placeholder[-remaining_values:] = vals[:remaining_values]
                        self.__initial_q_placeholder.sort()
                        self.__quantile_heights = np.array(
                            [self.__initial_q_placeholder for _ in range(len(self._quantiles))])
                        self.__p2_initialized = True

                        # remaining values will be added with P2 algorithm
                        input_vals = vals[remaining_values:]

                    # if not enough to fully initialize
                    else:
                        # insert values and wait for next iteration
                        self.__initial_q_placeholder[5 - remaining_values: 5 - remaining_values + len(vals)] = vals
                        input_vals = None
                else:
                    input_vals = vals

                if self.__p2_initialized:
                    for input_val in input_vals:
                        for ix, quantile in enumerate(self._quantiles):
                            # find index of new position
                            new_ix = np.searchsorted(self.__quantile_heights[ix, :], input_val, side='left')

                            # if new global minimum
                            if new_ix == 0 and input_val < self.__quantile_heights[ix, 0]:
                                self.__quantile_heights[:, 0] = input_val
                                # min case: increment all but first
                                increment_tail = 4

                            # if new global maximum
                            elif new_ix == 5:
                                self.__quantile_heights[:, 4] = input_val
                                # max case: increment only last
                                increment_tail = 1
                            else:
                                increment_tail = 5 - new_ix

                            # update estimated quantile positions
                            self.__quantile_pos[ix, -increment_tail:] += 1

                            # update estimated quantile orders
                            self.__quantile_orders[ix, :] += self.__quantile_tracks[ix, :]

                            # adjust quantiles with polynomial expansion
                            for qix in np.arange(1, 4):
                                n = self.__quantile_pos[ix, qix]
                                q = self.__quantile_heights[ix, qix]
                                d = self.__quantile_orders[ix, qix] - n

                                if (d >= 1 and self.__quantile_pos[ix, qix + 1] - n > 1) or \
                                        (d <= -1 and self.__quantile_pos[ix, qix - 1] - n < -1):

                                    # P2 algorithm iteration step
                                    d = np.sign(d)
                                    qp_ = self.__quantile_heights[ix, qix + 1]
                                    qm_ = self.__quantile_heights[ix, qix - 1]
                                    np_ = self.__quantile_pos[ix, qix + 1]
                                    nm_ = self.__quantile_pos[ix, qix - 1]

                                    # calculate polynomial interpolation coefficients
                                    outer = d / (np_ - nm_)
                                    inner_left = (n - nm_ + d) * (qp_ - q) / (np_ - n)
                                    inner_right = (np_ - n - d) * (q - qm_) / (n - nm_)
                                    qn = q + outer * (inner_left + inner_right)

                                    # determine approximate order polynomial based on edge coefficients
                                    if qm_ < qn < qp_:
                                        self.__quantile_heights[ix, qix] = qn
                                    else:
                                        self.__quantile_heights[ix, qix] = \
                                            q + d * (self.__quantile_heights[ix, qix + int(d)] - q) / \
                                            (self.__quantile_pos[ix, qix + int(d)] - n)

                                    self.__quantile_pos[ix, qix] = n + int(d)

                # if successfully initialized, update mapped CDF values
                if self.__p2_initialized:
                    self.support = self.__quantile_heights[:, 2]
                    self.cdf_vals = self._quantiles

            else:
                self._quantile_values = np.array([np.percentile(vals, q * 100) for q in self._quantiles])

                # update mapped CDF values
                self.support = self._quantile_values
                self.cdf_vals = self._quantiles

        # ------------------------------------------------------------------------------------------------------------
        # histogram estimate
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'histogram':
            # update CDF online or offline, depending on existing support
            if self.metadata.is_online:
                self._hist_values += np.concatenate([np.array([0.]),
                                                     np.histogram(vals, bins=self._histogram_bins)[0]])

            else:
                self._hist_values = np.concatenate([np.array([0.]),
                                                    np.histogram(vals, bins=self._histogram_bins)[0]])

            # update mapped CDF values
            self.support = self._histogram_bins
            self.cdf_vals = np.cumsum(self._hist_values).astype(float)
            self.cdf_vals /= self.cdf_vals[-1]

        # ------------------------------------------------------------------------------------------------------------
        # parametric estimate
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'parametric':
            # only allow one initialization
            if not self._offline_initialized:
                # fit distribution
                self._parametric_args = self.mle_fit_argmap(vals, self.metadata.parametric_family)

                # apply corrected mapping to distribution to generate cdf
                dist = getattr(stats, self.metadata.parametric_family)
                self._parametric_cdf = partial(dist.cdf, **self._parametric_args)
                self._parametric_cdf_inv = partial(dist.ppf, **self._parametric_args)

            else:
                warnings.warn("Attempted multiple initialization of distribution",
                              OfflineReinitializationWarning)

        # ------------------------------------------------------------------------------------------------------------
        # cluster KDE estimate
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'cluster':

            # clustering: discretize sample into nearest cluster values
            sorted_vals = np.sort(vals)

            # update RNG seed (used for testing reproducibility)
            if test_state:
                self.metadata.clustering_algorithm_kwargs.update({'random_state': test_state})

            self._cluster_model = \
                getattr(cluster, self.metadata.clustering_algorithm)(
                    **self.metadata.clustering_algorithm_kwargs
                )

            estimated_labels = self._cluster_model.fit_predict(sorted_vals.reshape(-1, 1))
            cluster_centers = self._cluster_model.cluster_centers_.flatten()
            estimated_clusters = cluster_centers[estimated_labels]

            # cluster parametric fit
            if self.metadata.clustering_parametric_family:

                # fit parametric estimate for each cluster
                n = float(len(sorted_vals))
                dist = getattr(stats, self.metadata.clustering_parametric_family)
                _component_args = []
                _component_cdfs = []
                _component_weights = []

                for cluster_ix in range(self.metadata.n_clusters):
                    cluster_samples = sorted_vals[estimated_labels == cluster_ix]
                    cluster_params = self.mle_fit_argmap(cluster_samples,
                                                         self.metadata.clustering_parametric_family)
                    cluster_cdf = partial(dist.cdf, **cluster_params)
                    cluster_weight = float(len(cluster_samples)) / n

                    _component_args.append(cluster_params)
                    _component_cdfs.append(cluster_cdf)
                    _component_weights.append(cluster_weight)

                # combine each constituent fit
                def _combined_cluster_cdf(x):
                    cdf_vals = np.array([f(x) for f in _component_cdfs])
                    cdf_weights = np.array(_component_weights)
                    return np.sum(cdf_vals * cdf_weights)

                self._cluster_parametric_args = _component_args
                self._cluster_parametric_cdf = _combined_cluster_cdf

            # cluster nonparametric fit
            else:
                # treat cluster estimates as discrete quantization
                unique_vals, unique_counts = np.unique(estimated_clusters, return_counts=True)
                self.support = unique_vals
                self.cdf_vals = unique_counts.astype(float) / float(np.sum(unique_counts))

    @property
    def cdf(self):
        """
        CDF property

        :return: function
        """
        # check for parametric estimates first
        if self.metadata.tracking_mode == 'parametric':
            return self._parametric_cdf
        elif self.metadata.tracking_mode == 'cluster' and self.metadata.clustering_parametric_family:
            return self._cluster_parametric_cdf

        # else fall back on interpolant solution
        else:
            return self._default_cdf_interpolation_mapping()

    @property
    def cdf_inv(self):
        """
        CDF inverse property

        :return: function
        """
        # check for parametric estimates first
        if self.metadata.tracking_mode == 'parametric':
            return self._parametric_cdf_inv
        elif self.metadata.tracking_mode == 'cluster' and self.metadata.clustering_parametric_family:
            raise NotImplementedError("Inverse for parameterized clusters not implemented")

        # else fall back on interpolant solution
        else:
            return self._default_cdf_interpolation_mapping(inv=True)


def distribution_factory(distribution_metadata):
    """
    Create distribution class from metadata

    :param distribution_metadata: DistributionMetadata
    :return: BaseDistribution
    """
    if distribution_metadata.is_discrete:
        return DiscreteDistribution(distribution_metadata)
    else:
        return ContinuousDistribution(distribution_metadata)

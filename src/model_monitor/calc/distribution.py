import warnings
import numpy as np
import scipy.stats as stats
import scipy.interpolate as interpolate
import sklearn.cluster as cluster

from model_monitor.report.shared import (
    NullableSampleWarning,
    SampleOutsideSupportWarning,
    UnsafeCastingWarning,
    OnlineSupportChangeWarning,
    OfflineReinitializationWarning
)

SUPPORTED_CLUSTER_MODELS = ['KMeans', 'MiniBatchKMeans', 'AgglomerativeClustering', 'Birch', 'SpectralClustering']


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


class BaseDistribution(object):
    """
    Base distribution to handle CDF estimation
    """

    def __init__(self, distribution_metadata):
        """
        Constructor

        :param distribution_metadata: NamedTuple
        """
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
        self.sample_size += np.sum(~mask)

        if self.metadata.is_nullable:
            self.null_size += np.sum(mask)
        elif np.isnan(new_values).any():
            warnings.warn(
                "Feature '{}' has null values, but is marked as not nullable".format(self.metadata.feature_name),
                NullableSampleWarning
            )

        nvals = vals[~mask]

        # type casting
        if not self.metadata.safe_cast:
            tvals = nvals.astype(self.metadata.default_type)
        else:
            cast_mask = can_cast(nvals)

            # if all values safely cast
            if cast_mask.all():
                tvals = nvals.astype(self.metadata.default_type)

            else:
                # some values cannot be safely cast
                warnings.warn("Feature '{}' has sample(s) that fail safe casting",
                              UnsafeCastingWarning)
                default_vals = np.where(
                    cast_mask,
                    np.repeat(self.metadata.default_value, len(nvals)).astype(self.metadata.default_type),
                    nvals)
                tvals = default_vals.astype(self.metadata.default_type)

        # support checking
        if self.metadata.support_maximum and tvals.max() > self.metadata.support_maximum:
            warnings.warn("Feature '{}' has sample(s) outside support maximum",
                          SampleOutsideSupportWarning)
            if self.metadata.remove_support_violators:
                tvals = tvals[tvals <= self.metadata.support_maximum]
                self.null_size += np.sum(tvals > self.metadata.support_maximum)

        if self.metadata.support_minimum and tvals.min() < self.metadata.support_minimum:
            warnings.warn("Feature '{}' has sample(s) outside support minimum",
                          SampleOutsideSupportWarning)
            if self.metadata.remove_support_violators:
                tvals = tvals[tvals >= self.metadata.support_minimum]
                self.null_size += np.sum(tvals < self.metadata.support_minimum)

        return tvals

    def null_proportion(self):
        """
        Proportion of nulls in the sample

        :return: float
        """
        return self.null_size / float(self.sample_size + self.null_size) if self.sample_size > 0 else 0.

    def cdf_linear_interpolation(self):
        """
        CDF linear interpolation function

        :return: function
        """
        return interpolate.interp1d(self.support, self.cdf_vals, kind='linear')

    def cdf_pchip_interpolation(self):
        """
        CDF cubic monotonic (piecewise cubic hermite interpolating polynomial) interpolation function

        :return: function
        """
        return interpolate.PchipInterpolator(self.support, self.cdf_vals)


class DiscreteDistribution(BaseDistribution):
    """
    CDF estimator class for discrete distributions

    This class dynamically tracks all unique samples (assumed to be significantly smaller than sample size), which
    allows for exact CDF form estimates.
    """

    def __init__(self, distribution_metadata):
        """
        Constructor

        :param distribution_metadata: NamedTuple
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
                    self.support = np.concatenate([self.support, new_support_values])
                    self.pdf_vals = np.concatenate([self.pdf_vals, np.repeat(0, len(new_support_values))])

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


class ContinuousDistribution(BaseDistribution):
    """
    CDF estimator class for continuous distributions
    
    This class requires a quantization method, specified in the associated distribution metadata
    """
    def __init__(self, distribution_metadata):
        """
        Constructor

        :param distribution_metadata: NamedTuple
        """
        BaseDistribution.__init__(self, distribution_metadata)

        if not self.metadata.is_online:
            self._offline_initialized = False

        # ------------------------------------------------------------------------------------------------------------
        # quantile estimates
        # ------------------------------------------------------------------------------------------------------------

        if self.metadata.tracking_mode == 'quantiles':

            # initialize target quantiles
            self._quantiles = np.linspace(0., 1., self.metadata.n_quantiles + 1)[1:-1]

            # add tail quantiles if configured
            if self.metadata.n_lower_tail_quantiles > 0:
                new_quantiles = np.power(10., -np.array(range(1, self.metadata.n_upper_tail_quantiles + 1)))
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
            
            if self.metadata.custom_histogram_bins:
                self._histogram_bins = np.array(self.metadata.custom_histogram_bins.split(",")).astype(float)
            else:
                self._histogram_bins = np.linspace(self.metadata.histogram_min,
                                                   self.metadata.histogram_max,
                                                   self.metadata.n_histogram_bins)
            if self.metadata.count_outside_range:
                self._histogram_bins = np.insert(self._histogram_bins, 0, -np.Inf)
                self._histogram_bins = np.append(self._histogram_bins, np.Inf)

            self._hist_values = np.zeros(len(self._histogram_bins))

        # ------------------------------------------------------------------------------------------------------------
        # parametric estimates
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'parametric':
            
            self._parametric_estimates = None

        # ------------------------------------------------------------------------------------------------------------
        # cluster KDE estimates
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'cluster_kde':
            
            self._cluster_model = None
            self._cluster_centers = []
            self._cluster_parameters = []

        else:
            raise ValueError("Must track quantiles, histogram, clusters, or parametric estimate of distribution")

    def update(self, new_values):
        """
        Constructor

        :param distribution_metadata: NamedTuple
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

        if self.metadata.tracking_mode == 'quantiles':

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
                self.cdf_vals = self._quantile_values

        # ------------------------------------------------------------------------------------------------------------
        # histogram estimate
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'quantiles':
            if self.metadata.is_online:
                self._hist_values += np.histogram(vals, bins=self._histogram_bins)[0]

            else:
                self._hist_values = np.histogram(vals, bins=self._histogram_bins)[0]

            # update mapped CDF values
            self.support = self._histogram_bins
            self.cdf_vals = np.cumsum(self._hist_values)

        # ------------------------------------------------------------------------------------------------------------
        # parametric estimate
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'parametric':
            if not self._offline_initialized:
                self._parametric_estimates = getattr(stats, self.metadata.parametric_family).fit(vals)
            else:
                pass

        # ------------------------------------------------------------------------------------------------------------
        # cluster KDE estimate
        # ------------------------------------------------------------------------------------------------------------

        elif self.metadata.tracking_mode == 'cluster_kde':

            # clustering: discretize sample into nearest cluster values
            sorted_vals = np.sort(vals)

            self._cluster_model = \
                getattr(cluster, self.metadata.cluster_algorithm)(
                    n_clusters=self.metadata.n_clusters,
                    **self.metadata.cluster_algorithm_kwargs
                )

            estimated_labels = self._cluster_model.fit_predict(sorted_vals)
            self._cluster_centers = self._cluster_model.cluster_centers

            # if nonparametric interpolation, reconstruct point estimates
            if self.metadata.cluster_interpolation_type == 'nonparametric':

                relative_freqs = np.cumsum(np.unique(estimated_labels, return_counts=True)[1])
                relative_freqs /= float(relative_freqs[-1])

                self.support = self._cluster_centers
                self.cdf_vals = relative_freqs

            # else, fit MLE kernel to each
            else:
                parametric_family = self.metadata.cluster_algorithm_kwargs['parametric_family']
                parameter_index = self.metadata.cluster_algorithm_kwargs['parameter_index']

                for ix, cluster_center in enumerate(self._cluster_centers):
                    self._cluster_centers[ix] = cluster_center
                    self._cluster_parameters = getattr(stats, parametric_family).fit(
                        sorted_vals[np.where(estimated_labels == ix, 1, 0)]
                    )[parameter_index]

    def cdf_linear_interpolation(self):
        """
        CDF linear interpolation function

        :return: function
        """
        if self.metadata.tracking_mode in ['quantiles', 'histogram'] or (
                self.metadata.tracking_mode == 'cluster_kde' and
                self.metadata.cluster_interpolation_type == 'nonparametric'):

            return BaseDistribution.cdf_linear_interpolation(self)

        else:
            raise ValueError("Cannot interpolate parametric estimate")

    def cdf_pchip_interpolation(self):
        """
        CDF cubic monotonic (piecewise cubic hermite interpolating polynomial) interpolation function

        :return: function
        """
        if self.metadata.tracking_mode in ['quantiles', 'histogram'] or (
                self.metadata.tracking_mode == 'cluster_kde' and
                self.metadata.cluster_interpolation_type == 'nonparametric'):

            return BaseDistribution.cdf_pchip_interpolation(self)

        else:
            raise ValueError("Cannot interpolate parametric estimate")


def distribution_factory(distribution_metadata):
    """
    Create distribution class from metadata

    :param distribution_metadata: NamedTuple
    :return: BaseDistribution
    """
    if distribution_metadata.is_discrete:
        return DiscreteDistribution(distribution_metadata)
    else:
        return ContinuousDistribution(distribution_metadata)

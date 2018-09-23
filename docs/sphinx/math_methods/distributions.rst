.. _distributions:

Distributions
======================

CDF estimation in ``model_monitor`` is highly configurable. Distribution quantization settings are stored in the
``DistributionMetadata`` object, and distribution quantization results are stored in an instance of ``BaseDistribution``.


Metadata
----------------------------

Each distribution can be quantized using a number of different methods, with settings stored in the ``DistributionMetadata``
class. Here, weill only list the possible options available: if you have questions about the mathematical details or
suggested usage, please consult the :download:`white paper <../../white_paper/mm_white_paper.pdf>`

Only two main arguments are required for each distribution: ``is_discrete`` and ``tracking_mode``. The latter only applies
to continuous distributions, for which it is computationally infeasible to quantize large samples using standard empirical methods.

The following arguments handle sample preprocessing settings. Note that all of these are optional:

- ``is_nullable``: if non-nullable, null values are removed
- ``default_type``: default ``numpy`` type to cast samples
- ``default_value``: default value for imputation purposes
- ``use_default_value_on_unsafe_cast``: if cast fails, use imputed value defined above
- ``support_minimum``: minimum sample value
- ``support_maximum``: maximum sample value
- ``remove_samples_out_of_support``: if True, removes values that violate support bounds
- ``is_online``: if True, distribution supports multiple sample updates
- ``warn_on_online_support_change``: if online and support changes after update, raise warning
- ``interpolation_mode``: if using CDF point estimate interpolation, specify method

The following arguments handle ``tracking_mode == 'quantile'``, where quantiles are estimated and interpolated.

- ``n_quantiles``: number of evenly spaced quantiles
- ``n_lower_tail_quantiles``: number of lower tail quantiles, in powers of ten
- ``n_upper_tail_quantiles``: number of upper tail quantiles, in powers of ten
- ``custom_quantiles``: list of custom quantile values (all in [0, 1])


The following arguments handle ``tracking_mode == 'histogram'``, where histograms are estimated and interpolated.

- ``histogram_min``: minimum histogram bin value
- ``histogram_max``: maximum histogram bin value
- ``n_histogram_bins``: number of evenly spaced histogram bins
- ``custom_histogram_bins``: list of custom histogram bins

The following arguments handle ``tracking_mode == 'cluster'``, where clustering is used to quantize the distribution.

- ``n_clusters``: number of clusters to quantize
- ``clustering_algorithm``: clustering algorithm name (as specified by ``sklearn.cluster``)
- ``clustering_algorithm_kwargs``: arguments to pass to the algorithm class constructor
- ``clustering_parametric_family``: if specified, fits parametric estimate to each cluster, else does interpolation

The following arguments handle ``tracking_mode == 'parametric'``, where parameters are directly estimated from a sample

- ``parametric_family``: name of parametric family (as specified by ``scipy.stats``)


Distributions
-------------------------

Each distribution object inherits from ``BaseDistribution`` and has two main exposed attributes: ``update(s)``, a method
which accepts and processes a new sample of data points, and ``cdf``, a property that returns the calculated CDF callable.
Since the behavior is extremely different for discrete distributions and continuous distributions, the ``distribution_factory``
function is used to automatically instantiate distribution instances from a given ``DistributionMetadata`` instance.


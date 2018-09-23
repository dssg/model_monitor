.. _output-schema:

Output Schema
==================================

The following schema holds output data from ``model_monitor``. This allows ``model_monitor`` to reuse data, such as
estimates of quantized distribution functions, repeatedly without having to recalculate them. Additionally, users can
directly query this data to view historical performance on metrics over time.

The schema has three tables for ``model_monitor`` configuration...

- ``distribution_metadata``: defines how distributions are quantized
- ``metric_defs``: defines how differences in distributions are calculated
- ``random_variables``: defines which random variables are tracked over time


... five tables for distribution quantization ...

- ``moments``: moment estimates
- ``quantiles``: quantile estimates
- ``histograms``: histogram estimates
- ``cluster_kdes``: clustering estimates
- ``parametric_estimates``: parametric estimates from known distributions

... and one table for storing metric results (``metrics``).

``distribution_metadata``:

.. table::

    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |              name              |   dtype    |                                  description                                  |nullable|constraints|
    +================================+============+===============================================================================+========+===========+
    |distribution_metadata_id        |Integer     |distribution metadata unique ID                                                |False   |PK         |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |is_discrete                     |Boolean     |if True, assumes support is quantized and tracks all unique samples            |False   |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |is_nullable                     |Boolean     |if non-nullable, null values are removed                                       |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |default_type                    |Text        |default ``numpy`` type to cast samples                                         |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |default_value                   |Float       |default value for imputation purposes                                          |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |use_default_value_on_unsafe_cast|Boolean     |if cast fails, use imputed value defined above                                 |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |support_minimum                 |Float       |minimum sample value                                                           |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |support_maximum                 |Float       |maximum support value                                                          |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |remove_samples_out_of_support   |Boolean     |if True, removes values that violate support bounds                            |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |is_online                       |Boolean     |if True, distribution supports multiple sample updates                         |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |warn_on_online_support_change   |Boolean     |if online and support changes after update, raise warning                      |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |interpolation_mode              |Text        |if using CDF point estimate interpolation, specify method                      |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |tracking_mode                   |Text        |quantization method for continuous distributions                               |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |n_quantiles                     |Integer     |number of evenly spaced quantiles                                              |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |n_lower_tail_quantiles          |Integer     |number of lower tail quantiles, in powers of ten                               |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |n_upper_tail_quantiles          |Integer     |number of upper tail quantiles, in powers of ten                               |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |custom_quantiles                |ARRAY(Float)|list of custom quantile values (all in [0, 1])                                 |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |histogram_min                   |Float       |minimum histogram bin value                                                    |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |histogram_max                   |Float       |maximum histogram bin value                                                    |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |n_histogram_bins                |Integer     |number of evenly spaced histogram bins                                         |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |custom_histogram_bins           |Text        |list of custom histogram bins                                                  |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |n_clusters                      |Integer     |number of clusters to quantize                                                 |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |clustering_algorithm            |Text        |clustering algorithm name (as specified by ``sklearn.cluster``)                |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |clustering_algorithm_kwargs     |JSONB       |arguments to pass to the algorithm class constructor                           |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |clustering_parametric_family    |Text        |if specified, fits parametric estimate to each cluster, else does interpolation|True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+
    |parametric_family               |Text        |name of parametric family (as specified by ``scipy.stats``)                    |True    |           |
    +--------------------------------+------------+-------------------------------------------------------------------------------+--------+-----------+

``random_variables``:

.. table::

    +------------------------+-------+--------------------------------------------------+--------+-----------+
    |          name          | dtype |                   description                    |nullable|constraints|
    +========================+=======+==================================================+========+===========+
    |rv_id                   |Integer|random variable ID                                |False   |PK         |
    +------------------------+-------+--------------------------------------------------+--------+-----------+
    |rv_name                 |Text   |random variable name                              |False   |unique     |
    +------------------------+-------+--------------------------------------------------+--------+-----------+
    |rv_type                 |Text   |random variable type                              |False   |           |
    +------------------------+-------+--------------------------------------------------+--------+-----------+
    |distribution_metadata_id|Integer|distribution_metadata_id                          |True    |           |
    +------------------------+-------+--------------------------------------------------+--------+-----------+
    |source_table            |Text   |feature source table                              |True    |           |
    +------------------------+-------+--------------------------------------------------+--------+-----------+
    |is_test                 |Boolean|if True, use test-associated data                 |True    |           |
    +------------------------+-------+--------------------------------------------------+--------+-----------+
    |latent_variable_name    |Integer|observed feature (independent of calculation type)|False   |           |
    +------------------------+-------+--------------------------------------------------+--------+-----------+
    |agg_func                |Text   |aggregation function used to generate feature     |False   |           |
    +------------------------+-------+--------------------------------------------------+--------+-----------+
    |time_agg                |Text   |time interval (in postgresql interval format)     |False   |           |
    +------------------------+-------+--------------------------------------------------+--------+-----------+

``quantiles``:

.. table::

    +--------------+-------+---------------------------------------+--------+-----------+
    |     name     | dtype |              description              |nullable|constraints|
    +==============+=======+=======================================+========+===========+
    |rv_id         |Integer|random variable ID                     |False   |PK, FK     |
    +--------------+-------+---------------------------------------+--------+-----------+
    |model_id      |Integer|model ID                               |False   |PK, ~FK    |
    +--------------+-------+---------------------------------------+--------+-----------+
    |quantile      |Float  |quantile, defined from [0, 1] inclusive|False   |PK         |
    +--------------+-------+---------------------------------------+--------+-----------+
    |quantile_value|Float  |quantile value                         |True    |           |
    +--------------+-------+---------------------------------------+--------+-----------+

``histograms``:

.. table::

    +---------------+-------+--------------------------------+--------+-----------+
    |     name      | dtype |          description           |nullable|constraints|
    +===============+=======+================================+========+===========+
    |rv_id          |Integer|random variable ID              |False   |PK, FK     |
    +---------------+-------+--------------------------------+--------+-----------+
    |model_id       |Integer|model ID                        |False   |PK, ~FK    |
    +---------------+-------+--------------------------------+--------+-----------+
    |bin_min        |Float  |bin minimum                     |False   |PK         |
    +---------------+-------+--------------------------------+--------+-----------+
    |bin_max        |Float  |bin maximum                     |False   |PK         |
    +---------------+-------+--------------------------------+--------+-----------+
    |value_count    |Integer|count of values in given bin    |True    |           |
    +---------------+-------+--------------------------------+--------+-----------+
    |value_frequency|Float  |frequency of values in given bin|True    |           |
    +---------------+-------+--------------------------------+--------+-----------+

``cluster_nonparametric_weights``:

.. table::

    +--------------+-------+------------------+--------+-----------+
    |     name     | dtype |   description    |nullable|constraints|
    +==============+=======+==================+========+===========+
    |rv_id         |Integer|random variable ID|False   |PK, FK     |
    +--------------+-------+------------------+--------+-----------+
    |model_id      |Integer|model ID          |False   |PK, ~FK    |
    +--------------+-------+------------------+--------+-----------+
    |cluster_center|Float  |cluster center    |False   |PK         |
    +--------------+-------+------------------+--------+-----------+
    |cluster_weight|Float  |cluster weight    |False   |           |
    +--------------+-------+------------------+--------+-----------+

``cluster_parameters``:

.. table::

    +-----------------------+-------+------------------------------------------------------+--------+-----------+
    |         name          | dtype |                     description                      |nullable|constraints|
    +=======================+=======+======================================================+========+===========+
    |rv_id                  |Integer|random variable ID                                    |False   |PK, FK     |
    +-----------------------+-------+------------------------------------------------------+--------+-----------+
    |model_id               |Integer|model ID                                              |False   |PK, ~FK    |
    +-----------------------+-------+------------------------------------------------------+--------+-----------+
    |cluster_center         |Float  |cluster center                                        |False   |PK         |
    +-----------------------+-------+------------------------------------------------------+--------+-----------+
    |cluster_parameter_index|Integer|cluster parameter index in ``scipy.stats`` constructor|False   |PK         |
    +-----------------------+-------+------------------------------------------------------+--------+-----------+
    |cluster_parameter_name |Text   |cluster parameter name                                |False   |           |
    +-----------------------+-------+------------------------------------------------------+--------+-----------+
    |cluster_parameter_value|Float  |cluster parameter value                               |False   |           |
    +-----------------------+-------+------------------------------------------------------+--------+-----------+
    |cluster_weight         |Float  |cluster weight                                        |False   |           |
    +-----------------------+-------+------------------------------------------------------+--------+-----------+

``parametric_estimates``:

.. table::

    +---------------+-------+----------------------------------------------+--------+-----------+
    |     name      | dtype |                 description                  |nullable|constraints|
    +===============+=======+==============================================+========+===========+
    |rv_id          |Integer|random variable ID                            |False   |PK, FK     |
    +---------------+-------+----------------------------------------------+--------+-----------+
    |model_id       |Integer|model ID                                      |False   |PK, ~FK    |
    +---------------+-------+----------------------------------------------+--------+-----------+
    |parameter_index|Integer|parameter index in ``scipy.stats`` constructor|False   |PK         |
    +---------------+-------+----------------------------------------------+--------+-----------+
    |parameter_name |Text   |parameter name                                |False   |           |
    +---------------+-------+----------------------------------------------+--------+-----------+
    |parameter_value|Float  |parameter value                               |False   |           |
    +---------------+-------+----------------------------------------------+--------+-----------+

``metric_defs``:

.. table::

    +----------------+--------+-----------------------------------+--------+-----------+
    |      name      | dtype  |            description            |nullable|constraints|
    +================+========+===================================+========+===========+
    |metric_id       |Integer |metric definition ID               |False   |PK         |
    +----------------+--------+-----------------------------------+--------+-----------+
    |metric_calc_name|Text    |metric calc name                   |False   |           |
    +----------------+--------+-----------------------------------+--------+-----------+
    |compare_interval|interval|comparison interval                |False   |           |
    +----------------+--------+-----------------------------------+--------+-----------+
    |subset_name     |Text    |subset name, as specified in config|False   |           |
    +----------------+--------+-----------------------------------+--------+-----------+
    |subset_threshold|Float   |subset filter argument             |True    |           |
    +----------------+--------+-----------------------------------+--------+-----------+

``metrics``:

.. table::

    +------------+-------+--------------------+--------+-----------+
    |    name    | dtype |    description     |nullable|constraints|
    +============+=======+====================+========+===========+
    |model_id    |Integer|model ID            |False   |PK, ~FK    |
    +------------+-------+--------------------+--------+-----------+
    |rv_id       |Integer|random variable ID  |False   |PK, FK     |
    +------------+-------+--------------------+--------+-----------+
    |metric_id   |Integer|metric definition ID|False   |PK, FK     |
    +------------+-------+--------------------+--------+-----------+
    |metric_value|Float  |metric value        |False   |           |
    +------------+-------+--------------------+--------+-----------+

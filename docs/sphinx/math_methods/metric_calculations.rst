.. _metric-calculations:


Metric calculations
===========================

Metric calculations in ``model_monitor`` are categorized by their calculation requirements. Some can be computed
directly from preprocessed sample values, and some require significant preprocessing before usage. Each

Note that here we only list the available calculations themselves. If you are interested in definitions, usages, or
mathematical considerations, please consult the :download:`white paper <../../white_paper/mm_white_paper.pdf>`


Independent point calculations
---------------------------------

These calculations operate on unordered samples from two random variables:

- ``moment_<p>``: the :math:`p` th moment
- ``central_moment_<p>``: the :math:`p` th central moment
- ``standardized_moment_<p>``: the :math:`p` th standardized moment
- ``q_<ntile>``: the :math:`ntile`:th quantile (ex: ``ntile='001'`` implies :math:`q = .001`)
- ``point_pdf_<v>``: the PDF estimate :math:`\mathbb{P}(X = v)` (only available for discrete distributions)
- ``point_cdf_<v>``: the CDF estimate :math:`\mathbb{P}(X \leq v)`


Ordered point comparisons
------------------------------

These calculations operate on ordered samples, where samples at the same index correspond to the same entity.

- ``l_<p>_entity``: the :math:`\ell_p` vector norm. Alternate definitions that pre-specify ``<p>``:

  - ``manhattan``: :math:`p = 1`
  - ``euclidean``: :math:`p = 2`
  - ``chebyshev``: :math:`p = \infty`

- ``cosine``: the geometric cosine distance
- ``dcorr``: distance correlation (AKA the cosine distance of the centered vectors)
- ``spearman``: Spearman rank correlation
- ``spearman_p``: p-value for the Spearman rank correlation test
- ``kendalltau``: Kendall's tau
- ``kendalltau_p``: p-value for the Kendall tau test
- ``wilcoxon``: Wilcoxon signed-rank test statistic
- ``wilcoxon``: p-value for the Wilcoxon signed-rank test statistic



Subset comparisons
------------------------

These calculations operate on a subset of entities of ordered samples. These metrics are not meaningful when looking at
the set of all entities.

- ``jaccard``: Jaccard similarity
- ``hamming``: Hamming distance
- ``russellrao``: Russell-rao distance


CDF comparisons
--------------------

These calculations require CDF estimates before they can be fully estimated.

- ``L_<p>_cdf``: the :math:`L_p` norm of the CDF difference. Alternate definitions that pre-specify ``<p>``:

  - ``ks_cdf``: Kolmogorov-Smirnov test statistic, :math:`p = \infty`
  - ``cvm_cdf``: Cramer-von-mises test statistic :math:`p = 2`, plus weighting function

- ``L_<p>_cdf_inv``:  the :math:`L_p` norm of the inverse CDF difference. Alternate definitions that pre-specify ``<p>``:

  - ``earthmover_cdf``: :math:`p=1`
  - ``gen_chi2_cdf``: :math:`p=2`



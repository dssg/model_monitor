import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize


# -------------------------------------------------------------------------------------------------------------------
# CDF differences
# -------------------------------------------------------------------------------------------------------------------


def cdf_merge(x1_support, x1_cdf, x2_support, x2_cdf):
    """
    Calculated merged empirical CDF from two PDFs with non-equal support

    Args:
        x1_support: first distribution support values
        x1_cdf: first distribution support PDF weights
        x2_support: second distribution support values
        x2_cdf: second distribution support PDF weights

    Returns:
        cdf1, cdf2, combined support, bin widths
    """
    # NB: below code was modified from scipy internal approximating function scipy.stats._cdf_distance
    # https://github.com/scipy/scipy/blob/14142ff70d84a6ce74044a27919c850e893648f7/scipy/stats/stats.py#L5422

    u_sorter = np.argsort(x1_support)
    v_sorter = np.argsort(x2_support)

    all_values = np.concatenate((x1_support, x2_support))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of both distributions.
    u_cdf_indices = x1_support[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = x2_support[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights.
    u_sorted_cumweights = np.concatenate(([0], np.cumsum(x1_cdf[u_sorter])))
    u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    v_sorted_cumweights = np.concatenate(([0], np.cumsum(x2_cdf[v_sorter])))
    v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    return u_cdf, v_cdf, all_values, deltas


def empirical_cdf_distance(x1_support, x1_cdf, x2_support, x2_cdf, p=1):
    """
    :math:`L^p` norm of joint CDF differences, for :math:`p \\in [0, \\infty)`

    Specific choices of $p$ yield specific distribution distances:
    * :math:`p=1`: Wasserstein metric (equivalent to "Earth-mover's distance")
    * :math:`p=2`: Energy distance (used in Cramer-von-Mises test)
    * :math:`p=\\infty`: Total variation distance (used in Kolmogorov-Smirnov test)

    Args:
        x1_support: first distribution support values
        x1_cdf: first distribution support PDF weights
        x2_support: second distribution support values
        x2_cdf: second distribution support PDF weights
        p: float, if p=np.Inf then total variation

    Returns:
        float, distance
    """
    u_cdf, v_cdf, all_values, deltas = cdf_merge(x1_support, x1_cdf, x2_support, x2_cdf)

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    if p == np.Inf:
        return np.max(np.abs(u_cdf - v_cdf))
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p),
                                       deltas)), 1 / p)


def integrated_lp_cdf_distance(f1, f2, xmin, xmax, p=1):
    """
    Numerical integral of Lp cdf distance between two CDFs

    :param f1: cdf function 1
    :param f2: cdf function 2
    :param xmin: integration start
    :param xmax: integration end
    :param p: Lp order, can be infinite
    :return: distance
    """

    if p == np.Inf:
        return -1 * optimize.fmin(lambda x: -np.abs(f1(x) - f2(x)), np.average([xmin, xmax]))

    else:
        if p == 1:
            def ifunc(x):
                return np.abs(f1(x) - f2(x))
        elif p == 2:
            def ifunc(x):
                return np.square(f1(x) - f2(x))
        else:
            def ifunc(x):
                return np.power(f1(x) - f2(x), p)

        return integrate.quad(ifunc, xmin, xmax)[0]

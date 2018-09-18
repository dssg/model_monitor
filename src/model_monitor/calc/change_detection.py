import numpy as np
import scipy.linalg as linalg
import scipy.special as spec
from cvxopt import matrix, spmatrix, solvers
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN


def lp_trendfilter_solve(time_series, lam=.5, p=0):
    """
    Perform the l^p trendfilter optimization problem, minimize (1/2) * ||x-s||_2^2 + lam * sum(y)
    subject to  -y <= D_p*x <= y

    :param time_series: np.array, ordered
    :param lam: float, regularization parameter
    :param p: int, trendfilter order (0 is constant, AKA variational denoising)
    :return: np.array
    """

    # normalize series and convert to cvxopt matrix
    ts_min = float(time_series.min())
    ts_max = float(time_series.max())

    if ts_max - ts_min == 0:
        # if time series is constant, return series directly
        return time_series

    ts_normed = (time_series - ts_min) / (ts_max - ts_min)
    ts = matrix(ts_normed)

    # construct difference matrix
    n = ts.size[0]
    m = n - p - 1

    binom_components = np.array([spec.binom(p + 1, k) for k in range(p + 2)])
    sign_array = np.arange(p + 2) % 2 * -2 + 1

    first_row = np.pad(binom_components * sign_array, (0, n - p - 2), 'constant')
    first_col = np.zeros(n - p - 1)
    first_col[0] = first_row[0]
    d = matrix(linalg.toeplitz(first_col, first_row))

    # recast optimization as quadratic programming problem
    pmat = d * d.T
    q = -d * ts
    g = spmatrix([], [], [], (2 * m, m))
    g[:m, :m] = spmatrix(1.0, range(m), range(m))
    g[m:, :m] = -spmatrix(1.0, range(m), range(m))
    h = matrix(lam, (2 * m, 1), tc='d')

    # solve quadratic problem and extract constants
    res = solvers.qp(pmat, q, g, h)
    normed_estimates = ts - d.T * res['x']

    # undo normalization
    return np.array((ts_max - ts_min) * normed_estimates + ts_min).flatten()


def lp_trendfilter_breakpoints(lp_filter, p=0, epsilon=10 ** -3):
    """
    Identify breakpoints in output of pth order trendfilter

    :param lp_filter: np.array, trendfilter prediction
    :param p: int, trendfilter order
    :param epsilon: changepoint detection threshold
    :return: np.array, breakpoint
    """

    diff = np.diff(lp_filter, n=p + 1)
    breakpoint_idx = np.where(np.abs(diff) > epsilon, np.arange(p + 1, len(diff) + p + 1), 0)
    return breakpoint_idx[np.nonzero(breakpoint_idx)[0]]


def piecewise_local_outlier_factor(time_series,
                                   breakpoints,
                                   contamination=.01,
                                   **lof_kwargs):
    """
    For each trendfilter component, apply local outlier analysis

    :param time_series: np.array, unfiltered time series
    :param breakpoints: np.array, breakpoints
    :param contamination: float, prior estimate of contamination proportion
    :return: np.array, outlier estimates
    """
    current_start_ix = 0
    outlier_estimates = np.zeros(len(time_series))

    for piece_arr in np.split(time_series, breakpoints):
        segment_len = len(piece_arr)

        lof = LocalOutlierFactor(n_neighbors=int(np.sqrt(segment_len)),
                                 contamination=contamination,
                                 **lof_kwargs)

        # TODO: return LOF scores (not preds) because sklearn only returns predictions, not scores

        vals = lof.fit_predict(piece_arr)
        outlier_estimates[current_start_ix:current_start_ix + segment_len] = vals

        # update start index
        current_start_ix += segment_len

    return outlier_estimates


def piecewise_dbscan_outliers(time_series,
                              breakpoints,
                              min_points,
                              epsilon,
                              **dbscan_kwargs):
    """
    For each trendfilter component, apply local outlier analysis

    :param time_series: np.array, unfiltered time series
    :param breakpoints: np.array, breakpoints
    :param min_points: int, DBSCAN min points for fully connected region
    :param epsilon: float, DBSCAN threshold for connected points
    :return: np.array, outlier estimates
    """
    current_start_ix = 0
    outlier_estimates = np.zeros(len(time_series))

    for piece_arr in np.split(time_series, breakpoints):
        segment_len = len(piece_arr)

        dbscan = DBSCAN(eps=epsilon,
                        min_samples=min_points,
                        **dbscan_kwargs)

        vals = dbscan.fit_predict(piece_arr)
        outlier_estimates[current_start_ix:current_start_ix + segment_len] = np.where(vals == -1, 1, 0)

        # update start index
        current_start_ix += segment_len

    return outlier_estimates

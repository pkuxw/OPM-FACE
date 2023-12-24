# -*- coding: utf-8 -*-
# Authors: Wei Xu <wxu@stu.pku.edu.cn>
#          Bingjiang Lyu <bingjiang.lyu@gmail.com>
# License: Simplified BSD

__all__ = ['Cluster_Perm_Test', 'find_latency']

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import label
from scipy.stats import t
from skimage.morphology import remove_small_holes


def find_max_mass(data: np.ndarray, thresh: int | float | np.ndarray) -> tuple:
    bwmatrix = np.zeros(data.shape)

    if thresh > 0:
        bwmatrix[data > thresh] = 1
    else:
        bwmatrix[data < thresh] = 1

    clusters_matrix, maxclu = label(bwmatrix)  # type: ignore

    maxclu = clusters_matrix.max().astype(int)

    if maxclu != 0:
        clust_mass = np.zeros(maxclu)
        for i in range(maxclu):
            clust_mass[i] = data[clusters_matrix.squeeze() == i + 1].sum()
    else:
        clust_mass = np.array([0])

    max_clust_mass = np.max(clust_mass)

    return max_clust_mass, clust_mass, clusters_matrix


def ttest(X: np.ndarray) -> np.ndarray:
    """Perform one-sample t-test.

    A modified version of scipy.stats.ttest_1samp avoiding
    a (relatively) time-consuming p-value calculation.

    Parameters
    ----------
    X : np.ndarray
        Array to return t-values for.

    Returns
    -------
    t : np.ndarray
        T-values, potentially adjusted using the hat method.
    """

    var = np.var(X, axis=0, ddof=1)
    return np.mean(X, axis=0) / np.sqrt(var / X.shape[0])


def Cluster_Perm_Test(
    x: np.ndarray,
    y: np.ndarray | int | float = 0,
    cft_p: float = 0.05,
    clu_p: float = 0.05,
    n_shuf: int = 2000,
    n_jobs: int = -1,
    n_thres: None | int = None,
) -> tuple:
    """Perform cluster-based permutation test for temporal data.

    Parameters
    ----------
    x : np.ndarray
        Data used in permutation test. For 1-D temporal data,
        x should be in shape (n_subj, n_timepoint). For 2-D
        temporal generalization data, x should be in shape
        (n_subj, n_timepoint_1, n_timepoint_2). Data with
        other dimensions are not supported. Also, currently
        we do not support spatiotemporal data.

    y : np.ndarray | int | float (default: 0)
        Data used in permutation test. If numeric, perform
        one-sample permutation test. If np.ndarray, perform
        permutation test for related data (equivalent to
        x - y one-sample permutation test in this case).
        Default to 0.

    cft_p : float (default: 0.05)
        The so-called “cluster forming threshold” in the form
        of a test statistic. It is noteworthy that this is not
        an alpha level or p-value. Data values more extreme than
        this threshold will be used to form clusters. Default to
        0.05, which is a common practice in previous literature.

    clu_p : float (default: 0.05)
        This is cluster corrected p-value. Cluster correction
        takes advantage of the fact that the timepoints in a
        typical time series are not completely independent.
        Instead of testing each timepoint individually,
        clusters of timepoints are tested for significance.

    n_shuf : int (default: 2000)
        The number of permutations to compute. Default to 2000.

    n_jobs : int (default: -1)
        The number of jobs to run in parallel. If -1, it is set
        to the number of CPU cores. Requires the joblib package.
        Note that permutation test for temporal generalization
        is quite time-consuming. Hence, it is recommended to set
        n_jobs to -1 for temporal generalization data. Howver,
        this argument will be automatically set to 1 when data
        is 1-D temporal data with shape (n_subj, n_timepoint).
        Also, remember to set n_jobs to 1 if permutation test
        is running on the server.

    n_thres : None | int (default: None)
        Warning! This parameter is set only for visualization
        purpose and will be automatically ignored for 1-D data.
        For 2-D generalization data, we simply remove all holes
        in significance masks whose area is smaller that n_thres.
        This is useful to reduce noise on significance mask.
        Also note that, n_thres only masks significance mask,
        while significance p-value map is unaffected.

    Returns
    -------
    ps_dict: dict
        A dict of tuple of ndarray, with the value containing
        the indices of locations that together form the given
        cluster along the given dimension. The key is the index
        of clusters.

    ps_map: np.ndarray
        Significance p-value map. Here we present the p-value
        for each cluster, but in a mask style. Will be useful
        for visualization. Note: Values larger that clu_p will
        be set to `np.nan`.

    ps_mask: np.ndarray
        Significance mask. Returns a list of boolean arrays,
        each with the same shape as the input data. Just by
        masking ps_map. Note: Values that are not significant
        in terms of clu_p will be set to `0`.

    Notes
    -----
    The goal of cluster-based permutation test is to find timepoints
    in a time series that differ between two conditions over subjects
    without performing single independent tests for each time-point.

    References
    ----------
    [1] Maris, E., & Oostenveld, R. (2007). Nonparametric statistical
    testing of EEG-and MEG-data. Journal of neuroscience methods, 164(1), 177-190.

    [2] Sassenhagen, J., & Draschkow, D. (2019). Cluster‐based permutation tests
    of MEG/EEG data do not establish significance of effect latency or location.
    Psychophysiology, 56(6), e13335.

    [3] For a clear explanation, please refer to this website:
    https://benediktehinger.de/blog/science/statistics-cluster-permutation-test
    """

    data = x - y
    n_subj = data.shape[0]
    thres = t.ppf(1 - cft_p, n_subj - 1)
    if x.ndim == 2:
        n_jobs = 1

    max_clust_mass_perm = np.full(n_shuf, np.nan)

    np.random.default_rng()

    def para_shuffle(data=data, n_subj=n_subj, thres=thres):
        data_p = data.copy()
        to_flip = np.random.rand(n_subj) > 0.5
        data_p[to_flip is True, :] *= -1

        tstats = ttest(data_p)
        max_clust_mass_perm_one, _, _ = find_max_mass(abs(tstats), thres)
        return max_clust_mass_perm_one

    worker = Parallel(n_jobs=n_jobs)
    max_clust_mass_perm = worker(delayed(para_shuffle)() for _ in range(n_shuf))

    tstats = ttest(data)
    _, clust_mass, ccc = find_max_mass(abs(tstats), thres)

    clust_p = np.zeros(clust_mass.shape[0])
    for i in range(clust_mass.shape[0]):
        clust_p[i] = (max_clust_mass_perm > clust_mass[i]).mean()

    ps_dict = {}
    ps_map = np.full(x.shape[1:], np.nan)

    cnt = 1
    for i in range(clust_mass.shape[0]):
        if clust_p[i] < clu_p:
            idx = np.where(ccc == i + 1)
            ps_dict[cnt] = idx
            ps_map[*idx] = clust_p[i]
            cnt += 1

    ps_mask = ~np.isnan(ps_map)

    if x.ndim > 2 and n_thres is not None:
        remove_small_holes(ps_mask, n_thres, in_place=True)

    return ps_dict, ps_map, ps_mask


def find_latency(
    result: np.ndarray,
    ps_mask: np.ndarray,
    custom_onset: int = 200,
    max_scheme: str | int = 'in_mask',
) -> tuple:
    # TODO: Docstring
    ps = ps_mask.copy()
    ps[:custom_onset] = 0
    idx = np.where(ps != 0)

    try:
        onset = idx[0][0]
        offset = idx[0][-1]
        duras = offset - onset
        res = result.mean(axis=0).copy()

        res[:onset] = np.nan
        if max_scheme == 'in_mask':
            res[offset:] = np.nan
        elif isinstance(max_scheme, int):
            res[max_scheme:] = np.nan

        peak = np.nanargmax(np.abs(res))

    except:
        onset, peak, duras = np.nan, np.nan, np.nan

    return onset, peak, duras


# import joblib as jl

# # a = jl.load('data-ctg.jl').ctg_mat[:, 0, :, :]
# a = jl.load('data-svm.jl').rsa_value_smoothed[:, :, 0]
# ps_dict, ps_map, ps_mask = Cluster_Perm_Test(a, 0, n_shuf=1000)

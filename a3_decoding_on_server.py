# -*- coding: utf-8 -*-
# Authors: Wei Xu <wxu@stu.pku.edu.cn>
# License: Simplified BSD
#
# This script is used to deploy time-consuming
# decoding on the server, including:
# (1) Train SVM classifier for multivariate SVM decoding
# (2) Train SVM classifier for temporal generalization analysis
# (3) Train SVM classifier for cross-section test-retest analysis

import platform
import warnings
from pathlib import Path
from typing import List

import fire
import joblib as jl
import numpy as np
import py7zr
from mne_rsa import rsa
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT = Path("/lustre/grp/gjhlab/xuw/data/")

SUBJ_LIST = [
    "S01",
    "S02",
    "S03",
    "S04",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
    "S18",
    "S19",
    "S20",
    "S21",
    "S22",
    "S23",
]

SUBJ_LIST_15 = [
    "S01",
    "S03",
    "S04",
    "S06",
    "S07",
    "S09",
    "S10",
    "S11",
    "S13",
    "S14",
    "S17",
    "S19",
    "S20",
    "S21",
    "S23",
]


# SVM decoding
def svm(
    XY: List[np.ndarray], tp: int = 0, kfold: int = 5, rep_n: int = 1000
) -> np.ndarray:
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    X, Y = XY  # X:(2560, n_chn, n_tp), Y:(2560?, )
    rdm_at_tp = np.zeros((64, 64))

    for s1 in range(64):
        for s2 in range(s1 + 1):
            s1X, s2X = X[Y == s1 + 1, :, tp], X[Y == s2 + 1, :, tp]
            s1Y, s2Y = np.tile(1, s1X.shape[0]), np.tile(0, s2X.shape[0])
            _acc = np.zeros(rep_n)
            n_train = int(np.min([s1Y.shape[0], s2Y.shape[0]]) * kfold / 10)
            for rep in range(rep_n):
                sd1 = (s1 + 3) * (s2 + 5) * (rep + 7) + sum(Y[[1, 2, 3, 4, 5]])  # type: ignore
                sd2 = (s1 + 5) * (s2 + 7) * (rep + 11) + sum(Y[[6, 7, 8, 9, 10]])
                sd3 = (s1 + 7) * (s2 + 11) * (rep + 13) + sum(Y[[11, 12, 13, 14, 15]])

                s1Xtr, s1Xte, s1Ytr, s1Yte = train_test_split(
                    s1X, s1Y, train_size=n_train, random_state=sd1
                )
                s2Xtr, s2Xte, s2Ytr, s2Yte = train_test_split(
                    s2X, s2Y, train_size=n_train, random_state=sd2
                )

                _Xtr, _Xte = np.vstack((s1Xtr, s2Xtr)), np.vstack((s1Xte, s2Xte))
                _Ytr, _Yte = np.hstack((s1Ytr, s2Ytr)), np.hstack((s1Yte, s2Yte))

                clf = LinearSVC(class_weight="balanced", random_state=sd3)
                clf.fit(_Xtr, _Ytr)
                pred = clf.predict(_Xte)

                _acc[rep] = np.sum(np.equal(pred, _Yte)) / _Yte.shape[0] * 100

            rdm_at_tp[s1, s2] = np.mean(_acc)
            rdm_at_tp[s2, s1] = np.mean(_acc)

    return rdm_at_tp


# Temporal generalization
def ctg(XY: List[np.ndarray], t1: int = 0) -> np.ndarray:
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    X, Y = XY  # X:(2560?, n_chn, n_tp), Y:(2560?, )
    X = X[:, :, ::2]
    n_sample = int(X.shape[0] / 64)
    tg = np.zeros((64, 64, 501, n_sample))
    for s1 in range(64):
        for s2 in range(s1 + 1):
            s1Xtr, s2Xtr = X[Y == s1 + 1, :, t1], X[Y == s2 + 1, :, t1]  # (40? chn)
            for rep in range(n_sample):
                _Xtr = np.vstack((np.delete(s1Xtr, rep, 0), np.delete(s2Xtr, rep, 0)))
                clf = LinearSVC(
                    class_weight="balanced",
                    random_state=(s1 + 1999) * (s2 + 10) * (rep + 28),
                )
                clf.fit(_Xtr, np.repeat([1, 0], n_sample - 1))
                for t2 in range(501):
                    s1Xte, s2Xte = X[Y == s1 + 1, :, t2], X[Y == s2 + 1, :, t2]
                    pred = clf.predict(np.vstack((s1Xte[rep, :], s2Xte[rep, :])))
                    tg[s1, s2, t2, rep] = np.equal(pred, np.array([1, 0])).mean()

    tg = tg.mean(axis=3)
    return tg[:, :, None, :]  # (64, 64, 1, 501)


# Cross-session test-retest
def trt(
    train_XY: List[np.ndarray],
    test_XY: List[np.ndarray],
    tp: int = 0,
    kfold: int = 5,
    rep_n: int = 1000,
) -> np.ndarray:
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    X_tr, Y_tr = train_XY  # X:(2560, n_chn, n_tp), Y:(2560?, )
    X_te, Y_te = test_XY  # X:(2560, n_chn, n_tp), Y:(2560?, )
    rdm_at_tp = np.zeros((64, 64))

    for s1 in range(64):
        for s2 in range(s1 + 1):
            s1X, s2X = X_tr[Y_tr == s1 + 1, :, tp], X_tr[Y_tr == s2 + 1, :, tp]
            s1X_2, s2X_2 = X_te[Y_te == s1 + 1, :, tp], X_te[Y_te == s2 + 1, :, tp]

            s1Y, s2Y = np.tile(1, s1X.shape[0]), np.tile(0, s2X.shape[0])
            s1Y_2, s2Y_2 = np.tile(1, s1X_2.shape[0]), np.tile(0, s2X_2.shape[0])

            _acc = np.zeros(rep_n)
            n_train = int(np.min([s1Y.shape[0], s2Y.shape[0]]) * kfold / 10)
            for rep in range(rep_n):
                sd1 = (
                    (s1 + 3) * (s2 + 5) * (rep + 7)
                    + sum(Y_tr[[1, 2, 3, 4, 5]])  # type: ignore
                    + sum(Y_te[[16, 17, 18, 19, 20]])
                )
                sd2 = (
                    (s1 + 5) * (s2 + 7) * (rep + 11)
                    + sum(Y_tr[[6, 7, 8, 9, 10]])
                    + sum(Y_te[[21, 22, 23, 24, 25]])
                )
                sd3 = (
                    (s1 + 7) * (s2 + 11) * (rep + 13)
                    + sum(Y_tr[[11, 12, 13, 14, 15]])
                    + sum(Y_te[[26, 27, 28, 29, 30]])  # type: ignore
                )

                s1Xtr, _, s1Ytr, _ = train_test_split(
                    s1X, s1Y, train_size=n_train, random_state=sd1
                )
                _, s1Xte, _, s1Yte = train_test_split(
                    s1X_2, s1Y_2, train_size=n_train, random_state=sd1
                )

                s2Xtr, _, s2Ytr, _ = train_test_split(
                    s2X, s2Y, train_size=n_train, random_state=sd2
                )
                _, s2Xte, _, s2Yte = train_test_split(
                    s2X_2, s2Y_2, train_size=n_train, random_state=sd2
                )

                _Xtr, _Xte = np.vstack((s1Xtr, s2Xtr)), np.vstack((s1Xte, s2Xte))
                _Ytr, _Yte = np.hstack((s1Ytr, s2Ytr)), np.hstack((s1Yte, s2Yte))

                clf = LinearSVC(class_weight="balanced", random_state=sd3)
                clf.fit(_Xtr, _Ytr)
                pred = clf.predict(_Xte)

                _acc[rep] = np.sum(np.equal(pred, _Yte)) / _Yte.shape[0] * 100

            rdm_at_tp[s1, s2] = np.mean(_acc)
            rdm_at_tp[s2, s1] = np.mean(_acc)

    return rdm_at_tp


def single(task: str, tp: int = 0) -> None:
    if task == "svm":
        for subj in SUBJ_LIST:
            res_dir = ROOT / "svm" / subj
            res_dir.mkdir(exist_ok=True)
            fname_res_7z = res_dir.parent / f"{subj}.7z"
            if not fname_res_7z.exists():
                fname_res = res_dir / f"tp{tp:04d}-kf5.jl"
                if fname_res.exists():
                    continue
                d1 = jl.load(ROOT / "raw" / "dec" / f"{subj}-A.jl")
                if (ROOT / "raw" / "dec" / f"{subj}-B.jl").exists():
                    d2 = jl.load(ROOT / "raw" / "dec" / f"{subj}-B.jl")
                    X = np.vstack((d1[0], d2[0]))
                    Y = np.hstack((d1[1], d2[1]))
                else:
                    X = d1[0]
                    Y = d1[1]
                res = svm([X, Y], tp)
                jl.dump(res, fname_res)

                if len(list(res_dir.glob("*.jl"))) != 1001:  # complete
                    continue

                with py7zr.SevenZipFile(fname_res_7z, "w") as z:
                    z.writeall(res_dir)

    elif task == "ctg":
        for subj in SUBJ_LIST:
            res_dir = ROOT / "ctg" / subj
            res_dir.mkdir(exist_ok=True)
            fname_res = res_dir / f"tp{tp:04d}.jl"
            if not fname_res.exists():
                d1 = jl.load(ROOT / "raw" / "dec" / f"{subj}-A.jl")
                if (ROOT / "raw" / "dec" / f"{subj}-B.jl").exists():
                    d2 = jl.load(ROOT / "raw" / "dec" / f"{subj}-B.jl")
                    X = np.vstack((d1[0], d2[0]))
                    Y = np.hstack((d1[1], d2[1]))
                else:
                    X = d1[0]
                    Y = d1[1]
                res = ctg([X, Y], tp)
                jl.dump(res, fname_res)

                _ctgmat = res  # (64, 64, 1, 501)

                mb = jl.load(ROOT / "low" / "mRDMs.jl")
                dsm_model = [mb.rdm_race, mb.rdm_gender, mb.rdm_age, mb.rdm_emotion]
                _res = np.zeros((5, 1, 501))
                dsm_data = []
                for ii in range(501):
                    _tmp = _ctgmat[:, :, 0, ii]
                    _tmp = np.tril(_tmp) + np.tril(_tmp).T  # type: ignore
                    np.fill_diagonal(_tmp, 0)
                    _ctgmat[:, :, 0, ii] = _tmp
                    dsm_data.append(_tmp)

                rsa_val_origin = rsa(
                    dsm_data, dsm_model, metric="partial-spearman", n_jobs=1
                )
                _res[1:, 0, :] = rsa_val_origin.swapaxes(0, 1)  # type: ignore  # (4, 501)

                for ii in range(64):
                    for jj in range(501):
                        _ctgmat[ii, ii, 0, jj] = np.nan

                _res[0, 0, :] = np.nanmean(_ctgmat, axis=(0, 1, 2))
                fname_bin = fname_res.parent / f"res-tp{tp:04d}.jl"
                jl.dump(_res, fname_bin)

            if len(list(res_dir.rglob("*.jl"))) == 501:  # completed
                fname_res = ROOT / "ctg" / f"{subj}.jl"
                if not fname_res.exists():
                    _all_res = np.zeros((5, 501, 501))
                    for tp in range(501):
                        _tmp = jl.load(res_dir / f"res-tp{tp:04d}.jl")
                        _all_res[:, tp, :] = _tmp[:, 0, :]
                    jl.dump(_all_res, fname_res)

    elif task == "retest":
        for subj in SUBJ_LIST_15:
            for train_sess in ["A", "B"]:
                for test_sess in ["A", "B"]:
                    res_dir = ROOT / "rel" / f"{train_sess} to {test_sess}" / subj
                    res_dir.mkdir(exist_ok=True, parents=True)
                    fname_res_7z = (
                        ROOT / "rel" / f"{train_sess} to {test_sess}" / f"{subj}.7z"
                    )
                    if not fname_res_7z.exists():
                        fname_res = res_dir / f"tp{tp:04d}-kf5.jl"
                        if fname_res.exists():
                            continue

                        train_ = jl.load(
                            ROOT / "raw" / "dec" / f"{subj}-{train_sess}.jl"
                        )
                        test_ = jl.load(ROOT / "raw" / "dec" / f"{subj}-{test_sess}.jl")
                        res = trt(train_, test_, tp)
                        jl.dump(res, fname_res)

                        if len(list(res_dir.glob("*.jl"))) != 1001:  # complete
                            continue

                        with py7zr.SevenZipFile(fname_res_7z, "w") as z:
                            z.writeall(res_dir)


if __name__ == "__main__":
    if platform.system() == "Linux":
        fire.Fire({"single": single})

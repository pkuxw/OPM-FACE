# -*- coding: utf-8 -*-
# Authors: Wei Xu <wxu@stu.pku.edu.cn>
# License: Simplified BSD
#
# This script is used to apply statistical test for:
# (1) Multivariate SVM decoding (svm)
# (2) Temporal generalization analysis (ctg)
# (3) Cross-section test-retest analysis (rel)
# (4) Sub-analysis within each face dimension (sub)

import datetime
import random
import string
import warnings
from itertools import product
from pathlib import Path

import bottleneck as bn
import cv2
import joblib as jl
import numpy as np
import pandas as pd
import pingouin as pg
from box import Box
from mne.stats import bootstrap_confidence_interval as bootci
from pose import pose_estimation  # type: ignore
from scipy.signal import convolve2d
from scipy.stats import rankdata
from statfunc import Cluster_Perm_Test, find_latency  # type: ignore
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

warnings.filterwarnings("ignore")

OVERWRITE = False

ROOT = Path("../data/")

SUBJ = [
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

SUBJ_15 = [
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

fname_low_rdm = ROOT / "low" / "mrdm.jl"
fname_low_lev = ROOT / "low" / "low.jl"

fname_svm = ROOT / "svm" / "data-svm.jl"
fname_ctg = ROOT / "ctg" / "data-ctg.jl"
fname_sub = ROOT / "sub" / "data-*.jl"
fname_rel = ROOT / "rel" / "rel-res.jl"

fname_svm_stat = ROOT / "svm" / "stat-svm.jl"
fname_ctg_stat = ROOT / "ctg" / "stat-ctg.jl"
fname_sub_stat = ROOT / "sub" / "stat-*.jl"
fname_rel_iccr = ROOT / "rel" / "icc.jl"
fname_rel_stat = ROOT / "rel" / "ps-diff.jl"

fname_svm_stat2 = ROOT / "svm" / "stat2-svm.jl"
fname_sub_stat2 = ROOT / "sub" / "stat2-sub.jl"

warnings.filterwarnings("ignore")

_DIM = ["image", "race", "gender", "age", "emotion"]

if OVERWRITE or not fname_svm_stat.exists():
    rbox = jl.load(fname_svm)
    nRDM = rbox.rsa_data  # (n_subj, 1001, 64, 64)
    for [i, j, k] in product(range(nRDM.shape[0]), range(1001), range(64)):
        nRDM[i, j, k, k] = np.nan

    rsa_data = rbox.rsa_value_smoothed  # (n_subj, 1001, 4)

    res = Box()
    for i, dim in enumerate(_DIM):
        if i == 0:
            _tc, level = np.nanmean(nRDM, axis=(2, 3)), 50  # (n_subj, 1001)
            _tc = bn.move.move_mean(_tc, 30, min_count=1, axis=1)
        else:
            _tc, level = rsa_data[:, :, i - 1], 0

        _, _, ps = Cluster_Perm_Test(_tc, level)

        onset, peak, dura = find_latency(_tc, ps, max_scheme=420)

        tmp_dict = Box()
        tmp_dict["Sig"] = ps
        tmp_dict["Onset"] = onset
        tmp_dict["Peak"] = peak
        tmp_dict["Duration"] = dura
        tmp_dict["Mean"] = _tc.mean(axis=0)
        tmp_dict["Mean+SE"] = _tc.mean(axis=0) + _tc.std(axis=0) / np.sqrt(_tc.shape[0])
        tmp_dict["Mean-SE"] = _tc.mean(axis=0) - _tc.std(axis=0) / np.sqrt(_tc.shape[0])
        tmp_dict["+0.95CI"] = bootci(_tc, ci=0.95, n_bootstraps=1000)[1, :]
        tmp_dict["-0.95CI"] = bootci(_tc, ci=0.95, n_bootstraps=1000)[0, :]

        res[dim] = tmp_dict

    jl.dump(res, fname_svm_stat)

if OVERWRITE or not Path(str(fname_sub_stat).replace("*", "race")).exists():
    for dims in ["age", "emotion", "gender", "race"]:
        fname_tmp = Path(str(fname_sub_stat).replace("*", dims))
        if not fname_tmp.exists():
            rbox = jl.load(Path(str(fname_sub).replace("*", dims)))
            res = Box()
            match dims:
                case "race":
                    other_dim = ["gender", "age", "emotion"]
                    __type = ["native", "exotic"]
                case "gender":
                    other_dim = ["race", "age", "emotion"]
                    __type = ["male", "female"]
                case "age":
                    other_dim = ["race", "gender", "emotion"]
                    __type = ["young", "elderly"]
                case "emotion":
                    other_dim = ["race", "gender", "age"]
                    __type = ["neutral", "joyful"]
                case _:
                    raise NotImplementedError

            for type in __type:
                res[type] = Box()
                rsa_data = rbox[f"{type}_rsa_val"]
                for i, dim in enumerate(other_dim):
                    _tc = rsa_data[:, :, i]
                    _, _, ps = Cluster_Perm_Test(_tc, 0)
                    onset, peak, dura = find_latency(_tc, ps, max_scheme=420)
                    tmp_dict = Box()
                    tmp_dict["Sig"] = ps
                    tmp_dict["Onset"] = onset
                    tmp_dict["Peak"] = peak
                    tmp_dict["Duration"] = dura
                    tmp_dict["Mean"] = _tc.mean(axis=0)
                    tmp_dict["Mean+SE"] = _tc.mean(axis=0) + _tc.std(axis=0) / np.sqrt(
                        _tc.shape[0]
                    )
                    tmp_dict["Mean-SE"] = _tc.mean(axis=0) - _tc.std(axis=0) / np.sqrt(
                        _tc.shape[0]
                    )
                    tmp_dict["+0.95CI"] = bootci(_tc, ci=0.95, n_bootstraps=1000)[1, :]
                    tmp_dict["-0.95CI"] = bootci(_tc, ci=0.95, n_bootstraps=1000)[0, :]
                    res[type][dim] = tmp_dict

            res["difference"] = Box()
            for i, dimdiff in enumerate([rr + "_diff" for rr in other_dim]):
                _tc = (
                    rbox[__type[0] + "_rsa_val"] - rbox[__type[1] + "_rsa_val"]
                )  # [n_subj, 1001, 3]
                _tc = _tc[:, :, i]
                _tc_1 = rbox[__type[0] + "_rsa_val"][:, :, i]
                _tc_2 = rbox[__type[1] + "_rsa_val"][:, :, i]

                sig_1 = res[__type[0]][dimdiff.rstrip("_diff")].Sig
                sig_2 = res[__type[1]][dimdiff.rstrip("_diff")].Sig
                all_sig = sig_1 + sig_2
                all_sig[:200] = 0
                t1 = np.where(all_sig != 0)[0][0]
                t2 = np.where(all_sig != 0)[0][-1]

                _tc_1[:, :t1] = 0
                _tc_1[:, t2:] = 0
                _tc_2[:, :t1] = 0
                _tc_2[:, t2:] = 0

                _, _, ps = Cluster_Perm_Test(_tc_1, _tc_2)

                onset, peak, dura = find_latency(_tc_1 - _tc_2, ps)

                tmp_dict = Box()
                tmp_dict["Sig"] = ps
                tmp_dict["Onset"] = onset
                tmp_dict["Peak"] = peak
                tmp_dict["Duration"] = dura
                tmp_dict["Mean"] = _tc.mean(axis=0)
                tmp_dict["Mean+SE"] = _tc.mean(axis=0) + _tc.std(axis=0) / np.sqrt(
                    _tc.shape[0]
                )
                tmp_dict["Mean-SE"] = _tc.mean(axis=0) - _tc.std(axis=0) / np.sqrt(
                    _tc.shape[0]
                )
                tmp_dict["+0.95CI"] = bootci(_tc, ci=0.95, n_bootstraps=1000)[1, :]
                tmp_dict["-0.95CI"] = bootci(_tc, ci=0.95, n_bootstraps=1000)[0, :]
                res["difference"][dimdiff] = tmp_dict

            jl.dump(res, fname_tmp)

if not fname_rel_iccr.exists():
    res = jl.load(fname_rel)
    coh = np.zeros((1001))
    coh_up = np.zeros((1001))
    coh_lo = np.zeros((1001))
    with tqdm(total=1001) as p:
        for tp in range(1001):
            for subj in range(15):
                _tmp1 = np.zeros((4, 16 * 16))
                for i, cond in enumerate(["AA", "AB", "BB", "BA"]):
                    rd = res[cond].data[subj, tp, :, :].flatten()
                    for j in range(16 * 16):
                        _tmp1[i, j] = np.nanmean(rd[j * 4 : j * 4 + 4])
                    _tmp1[i, :] = rankdata(
                        _tmp1[i, :], method="average", nan_policy="omit"
                    )
                data = {
                    "Pair": np.tile(range(16 * 16), 4),
                    "Cond": np.repeat(range(4), 16 * 16),
                    "Scores": _tmp1.flatten(),
                }
                df = pd.DataFrame(data)
                _res = pg.intraclass_corr(
                    data=df,
                    targets="Pair",
                    raters="Cond",
                    ratings="Scores",
                    nan_policy="omit",
                )
                coh[tp] = coh[tp] + _res["ICC"][5]
                coh_lo[tp] = coh_lo[tp] + _res["CI95%"][5][0]
                coh_up[tp] = coh_up[tp] + _res["CI95%"][5][1]
            p.update(1)
    jl.dump([coh / 15, coh_lo / 15, coh_up / 15], fname_rel_iccr)

if OVERWRITE or not fname_rel_stat.exists():
    res = jl.load(fname_rel)
    psig = {
        "Image": dict(),
        "Age": dict(),
        "Emotion": dict(),
        "Gender": dict(),
        "Race": dict(),
    }
    abl = ["AA", "BB", "AB", "BA"]
    idx = ["Image", "Race", "Gender", "Age", "Emotion"]

    for tt in ["Image", "Age", "Emotion", "Gender", "Race"]:
        AA_all = res["AA"].dec[:, :, idx.index(tt)]
        AB_all = res["AB"].dec[:, :, idx.index(tt)]
        BB_all = res["BB"].dec[:, :, idx.index(tt)]
        BA_all = res["BA"].dec[:, :, idx.index(tt)]
        for cond in abl:
            level = 50 if tt == "Image" else 0
            _, _, psig[tt][cond] = Cluster_Perm_Test(eval(f"{cond}_all"), level)
            for cond2 in abl:
                if abl.index(cond2) > abl.index(cond):
                    _, _, ps = Cluster_Perm_Test(
                        eval(f"{cond}_all"), eval(f"{cond2}_all")
                    )
                    psig[tt][
                        f"Count: {cond}-{cond2}"
                    ] = f"{ps.sum()} different timepoints..."
                    psig[tt][f"{cond}-{cond2}"] = ps

    jl.dump(psig, fname_rel_stat)

if not fname_ctg_stat.exists():
    res = np.zeros((5, 501, 501))
    cbox = jl.load(fname_ctg)
    cbox = cbox.ctg_mat
    for i in range(5):
        level = 0.5 if i == 0 else 0
        kernel = np.ones((10, 10)) / 100
        m = np.zeros(cbox[:, i, :, :].shape)
        for sub in range(m.shape[0]):
            m[sub, :, :] = convolve2d(cbox[sub, i, :, :], kernel, mode="same")

        print(f"Dimension {i} begins at {datetime.datetime.today()}...")
        _, _, res[i, :, :] = Cluster_Perm_Test(m, level)
        print(f"Dimension {i} finished at {datetime.datetime.today()}...")

    jl.dump(res, fname_ctg_stat)

if not fname_low_lev.exists():
    _luminance = np.zeros(64)
    _contrast = np.zeros(64)
    _face_size = np.zeros(64)
    _face_loc_x = np.zeros(64)
    _face_loc_y = np.zeros(64)

    for t in range(64):
        fname_pic = f"./pose/pic/{t+1}.png"
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )  # type: ignore
        img = cv2.imread(fname_pic)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y, w, h = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )[0]
        center_x = x + w // 2
        center_y = y + h // 2
        pic_size = gray.shape[0]
        _luminance[t] = gray.mean()
        _contrast[t] = np.sqrt(
            np.sum((gray - gray.mean()) ** 2) / (pic_size * pic_size)
        )
        _face_size[t] = w / pic_size
        _face_loc_x[t] = (center_x - pic_size / 2) / pic_size * 2
        _face_loc_y[t] = (center_y - pic_size / 2) / pic_size * 2

    rm = np.array([[1] * 32, [0] * 32] * 1).flatten()
    gm = np.array([[1] * 16, [0] * 16] * 2).flatten()
    am = np.array([[1] * 8, [0] * 8] * 4).flatten()
    em = np.array([[1] * 4, [0] * 4] * 8).flatten()

    mrdm = jl.load(fname_low_rdm)
    smat = mrdm.rdm_lpips  # (64, 64)

    a = pose_estimation.batch_pose_detect("./pose/pic_nobg")  # type: ignore

    _MET = ["luminance", "contrast", "face_size", "face_loc_x", "face_loc_y", "pose"]
    res = Box()
    for d, dim in zip(["r", "g", "a", "e"], _DIM[1:]):
        exec(
            f"res.intra_loss_{dim}=[smat[i,j] for i in range(64) for j in range(i) if {d}m[i]=={d}m[j]]",
            locals(),
        )
        exec(
            f"res.inter_loss_{dim}=[smat[i,j] for i in range(64) for j in range(i) if {d}m[i]!={d}m[j]]",
            locals(),
        )
        exec(f"res.pose_{dim} = [a[{d}m==1, 1], a[{d}m==0, 1]]", locals())
        for met in _MET[:5]:
            exec(f"res.{met}_{dim} = [_{met}[{d}m==1], _{met}[{d}m==0]]", locals())

    jl.dump(res, fname_low_lev)


def GenRandomString(slen=15):
    return "".join(random.sample(string.ascii_letters + string.digits, slen))


def _boot_svm() -> None:
    TMPDIR = ROOT / "svm" / "bootstrap"

    fname_boot = TMPDIR / f"tmp-{GenRandomString()}.jl"

    rbox = jl.load(ROOT / "svm" / "data-svm.jl")
    nRDM = rbox.rsa_data  # (n_subj, 1001, 64, 64)

    for i in range(nRDM.shape[0]):
        for j in range(1001):
            for k in range(64):
                nRDM[i, j, k, k] = np.nan

    rsa_data = rbox.rsa_value_smoothed  # (n_subj, 1001, 4)

    if not fname_boot.exists():
        res = Box()
        res.onsets = np.zeros(5)
        res.peaks = np.zeros(5)
        res.duras = np.zeros(5)
        n_subj = nRDM.shape[0]

        idx = np.random.choice(n_subj, size=n_subj, replace=True)

        for i in range(5):
            if i == 0:
                tc, level = np.nanmean(nRDM, axis=(2, 3)), 50  # (n_subj, 1001)
            else:
                tc, level = rsa_data[:, :, i - 1], 0

            _, _, ps = Cluster_Perm_Test(tc[idx, :], level)

            res.onsets[i], res.peaks[i], res.duras[i] = find_latency(tc[idx, :], ps)

        jl.dump(res, fname_boot)


if not fname_svm_stat2.exists():
    TMPDIR = ROOT / "svm" / "bootstrap"
    TMPDIR.mkdir(exist_ok=True)

    n_boot = 2000

    with tqdm_joblib(desc="Progress", total=n_boot) as progress_bar:
        jl.Parallel(n_jobs=-1)(jl.delayed(_boot_svm)() for _ in range(n_boot))

    if len(list(TMPDIR.glob("*.jl"))) == n_boot:
        res = Box()
        res.onsets = np.zeros((5, n_boot))
        res.peaks = np.zeros((5, n_boot))
        res.duras = np.zeros((5, n_boot))
        res.label = ["image", "race", "gender", "age", "emotion"]

        for i, tmpf in enumerate(TMPDIR.glob("*.jl")):
            _b = jl.load(tmpf)
            res.onsets[:, i] = _b.onsets
            res.peaks[:, i] = _b.peaks
            res.duras[:, i] = _b.duras

        jl.dump(res, fname_svm_stat2)


def _boot_sub() -> None:
    TMPDIR = ROOT / "sub" / "bootstrap"

    fname_boot = TMPDIR / f"tmp-{GenRandomString()}.jl"

    if not fname_boot.exists():
        res = Box(default_box=True)

        for dim in ["race", "gender", "age", "emotion"]:
            dat = jl.load(ROOT / "sub" / f"data-{dim}.jl")
            kk = list(dat.keys())
            kk.remove("rsa_dimension")

            n_subj = 21
            idx = np.random.choice(n_subj, size=n_subj, replace=True)

            for cla in kk:
                o = np.zeros(3)
                p = np.zeros(3)
                d = np.zeros(3)

                for i in range(3):
                    tc, level = dat[cla][idx, :, i], 0
                    _, _, ps = Cluster_Perm_Test(tc, level)
                    o[i], p[i], d[i] = find_latency(tc, ps)

                res[dim][cla].onsets = o
                res[dim][cla].peaks = p
                res[dim][cla].duras = d

        jl.dump(res, fname_boot)


if not fname_sub_stat2.exists():
    TMPDIR = ROOT / "sub" / "bootstrap"
    TMPDIR.mkdir(exist_ok=True)

    n_boot = 2000

    with tqdm_joblib(desc="Progress", total=n_boot) as progress_bar:
        jl.Parallel(n_jobs=-1)(jl.delayed(_boot_sub)() for _ in range(n_boot))

    if len(list(TMPDIR.glob("*.jl"))) == n_boot:
        res = Box(default_box=True)
        for dim in ["race", "gender", "age", "emotion"]:
            res[dim].onsets = np.zeros((3, 2, n_boot))
            res[dim].peaks = np.zeros((3, 2, n_boot))
            res[dim].duras = np.zeros((3, 2, n_boot))

            for i, tmpf in enumerate(TMPDIR.glob("*.jl")):
                _b = jl.load(tmpf)
                kk = list(_b[dim].keys())
                for k in range(2):
                    res[dim].onsets[:, k, i] = _b[dim][kk[k]].onsets
                    res[dim].peaks[:, k, i] = _b[dim][kk[k]].peaks
                    res[dim].duras[:, k, i] = _b[dim][kk[k]].duras

        jl.dump(res, fname_sub_stat2)

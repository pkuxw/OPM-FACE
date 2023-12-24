# -*- coding: utf-8 -*-
# Authors: Wei Xu <wxu@stu.pku.edu.cn>
# License: Simplified BSD
#
# This script is used to apply RSA to:
# (1) Multivariate SVM decoding (svm)
# (2) Temporal generalization analysis (ctg)
# (3) Cross-section test-retest analysis (rel)
# (4) Sub-analysis within each face dimension (sub)

import warnings
from itertools import product
from pathlib import Path
from typing import Tuple

import bottleneck as bn
import cv2
import joblib as jl
import lpips
import numpy as np
import torch
from box import Box
from mne_rsa import rsa
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings('ignore')

ROOT = Path('../data/')

# fmt: off
SUBJ = ['S01', 'S02', 'S03', 'S04', 'S06', 'S07', 'S08',
        'S09', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16',
        'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23']

SUBJ_15 = ['S01', 'S03', 'S04', 'S06', 'S07',
           'S09', 'S10', 'S11', 'S13', 'S14',
           'S17', 'S19', 'S20', 'S21', 'S23']
# fmt: on


def GenRDM(picROOT: Path = Path('./pose/pic/')) -> Box:
    def _read(fn_img: Path) -> torch.Tensor:
        img = cv2.imread(str(fn_img))
        img = transforms.ToTensor()(img)
        mi, mx = torch.min(img), torch.max(img)
        img = (img - mi) / (mx - mi)
        image = torch.zeros(1, 3, 500, 500)
        image[0, :, :, :] = img * 2 - 1
        return image

    _rdm = np.zeros((64, 64), dtype=np.float32)

    loss_fn_alex = lpips.LPIPS(net='alex')

    with tqdm(total=2016) as pb:
        pb.set_description('Processing:')
        for [ii, jj] in product(*(range(64),) * 2):
            if ii <= jj:
                continue
            img1 = _read(picROOT / f'{ii+1}.png')
            img2 = _read(picROOT / f'{jj+1}.png')
            _rdm[ii, jj] = loss_fn_alex(img1, img2)
            _rdm[jj, ii] = _rdm[ii, jj]
            pb.update(1)

    RDM = Box()
    RDM.rdm_lpips = _rdm
    RDM.rdm_race = 1.0 - np.array((([1] * 32 + [0] * 32) * 32 + ([0] * 32 + [1] * 32) * 32) * 1).reshape((64, 64))
    RDM.rdm_gender = 1.0 - np.array((([1] * 16 + [0] * 16) * 32 + ([0] * 16 + [1] * 16) * 32) * 2).reshape((64, 64))
    RDM.rdm_age = 1.0 - np.array((([1] * 8 + [0] * 8) * 32 + ([0] * 8 + [1] * 8) * 32) * 4).reshape((64, 64))
    RDM.rdm_emotion = 1.0 - np.array((([1] * 4 + [0] * 4) * 32 + ([0] * 4 + [1] * 4) * 32) * 8).reshape((64, 64))

    return RDM


if __name__ == '__main__':
    n_subj = len(SUBJ)

    fname_rdm = ROOT / 'low' / 'mrdm.jl'  # model RDMs
    fname_svm = ROOT / 'svm' / 'data-svm.jl'  # SVM decoding
    fname_ctg = ROOT / 'ctg' / 'data-ctg.jl'  # temporal generalization
    fname_rel = ROOT / 'rel' / 'rel-res.jl'  # test-retest

    if not fname_rdm.exists():
        mb = GenRDM()
        jl.dump(mb, fname_rdm)
    else:
        mb = jl.load(fname_rdm)

    if not fname_svm.exists():
        dsm_data = []
        for i, dirs in enumerate(SUBJ):
            for tp in range(1001):
                _tmp = jl.load(ROOT / 'svm' / dirs / f'tp{tp:04d}-kf5.jl')
                _tmp = np.tril(_tmp) + np.tril(_tmp).T
                np.fill_diagonal(_tmp, 0)
                dsm_data.append(_tmp)

        dsm_model = [mb.rdm_race, mb.rdm_gender, mb.rdm_age, mb.rdm_emotion]
        rsa_val = rsa(dsm_data, dsm_model, metric='partial-spearman', n_jobs=-1, verbose=True)
        rsa_val_origin = rsa_val.reshape([n_subj, 1001, len(dsm_model)])  # type: ignore # [21, 1001, 4]
        rsa_val_smooth = bn.move.move_mean(rsa_val_origin, 30, min_count=1, axis=1)

        res = Box()
        res.rsa_value_original = rsa_val_origin
        res.rsa_value_smoothed = rsa_val_smooth
        res.rsa_dimension = ['race', 'gender', 'age', 'emotion']
        res.rsa_model = dsm_model
        res.rsa_data = np.array(dsm_data).reshape(n_subj, 1001, 64, 64)
        res.subject_names = SUBJ

        jl.dump(res, fname_svm)

    def swap_2(rdm: np.ndarray, dim: str = 'race', idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        match dim:
            case 'race':
                njm = ([1] * 32 + [2] * 32) * 1
            case 'gender':
                njm = ([1] * 16 + [2] * 16) * 2
            case 'age':
                njm = ([1] * 8 + [2] * 8) * 4
            case 'emotion':
                njm = ([1] * 4 + [2] * 4) * 8
            case _:
                raise NotImplementedError

        li = []
        for i in range(64):
            for j in range(64):
                if njm[i] == njm[j] == (idx + 1):
                    li.append(rdm[i, j])
        return np.array(li).reshape((32, 32))  # type: ignore

    for dim in ['race', 'gender', 'age', 'emotion']:
        fname_data = ROOT / 'sub' / f'data-{dim}.jl'
        fname_data.parent.mkdir(parents=True, exist_ok=True)
        if not fname_data.exists():
            rbox = jl.load(fname_svm)
            data = rbox.rsa_data  # (n_subj, 1001, 64, 64)
            data = bn.move.move_mean(data, 30, min_count=1, axis=1)

            res = Box()
            match dim:
                case 'race':
                    res.rsa_dimension = ['gender', 'age', 'emotion']
                    model = [mb.rdm_gender, mb.rdm_age, mb.rdm_emotion]
                    __type = ['native', 'exotic']
                case 'gender':
                    res.rsa_dimension = ['race', 'age', 'emotion']
                    model = [mb.rdm_race, mb.rdm_age, mb.rdm_emotion]
                    __type = ['male', 'female']
                case 'age':
                    res.rsa_dimension = ['race', 'gender', 'emotion']
                    model = [mb.rdm_race, mb.rdm_gender, mb.rdm_emotion]
                    __type = ['young', 'elderly']
                case 'emotion':
                    res.rsa_dimension = ['race', 'gender', 'age']
                    model = [mb.rdm_race, mb.rdm_gender, mb.rdm_age]
                    __type = ['neutral', 'joyful']
                case _:
                    raise NotImplementedError

            for type_id, type in enumerate(__type):
                _d = np.zeros((data.shape[0], data.shape[1], 32, 32))  # (21, 1001, 32, 32)
                for aa in range(_d.shape[0]):
                    for bb in range(_d.shape[1]):
                        _d[aa, bb, :, :] = swap_2(data[aa, bb, :, :], dim, type_id)
                _model = [swap_2(i, dim, type_id) for i in model]

                _data = [_d[i, j, :, :] for i in range(n_subj) for j in range(1001)]
                rsa_val = rsa(_data, _model, metric='partial-spearman', n_jobs=-1, verbose=True)
                _val = rsa_val.reshape([n_subj, 1001, len(model)])  # type: ignore # [21, 1001, 3]
                _val = bn.move.move_mean(_val, 30, min_count=1, axis=1)
                res[f'{type}_rsa_val'] = _val

            jl.dump(res, fname_data)

    if not fname_ctg.exists():
        _mat = np.zeros((n_subj, 5, 501, 501))
        for t in range(501):
            for i, f in enumerate(SUBJ):
                _mat[i, :, t, :] = jl.load(ROOT / 'ctg' / f / f'res-tp{t:04d}.jl')[:, 0, :]

        res = Box()
        res.ctg_mat = _mat
        res.subject_index = list(range(n_subj))
        res.rsa_dimension = ['image', 'race', 'gender', 'age', 'emotion']
        jl.dump(res, fname_ctg)

    if not fname_rel.exists():
        mb = jl.load(fname_rdm)
        ROOT_REL = fname_rel.parent
        dat = Box()

        for part, part2 in zip(['AA', 'BB', 'AB', 'BA'], ['A to A', 'B to B', 'A to B', 'B to A']):
            dat[part] = np.zeros((15, 1001, 64, 64))
            for idx, subj in enumerate(SUBJ_15):
                for tp in range(1001):
                    dat[part][idx][tp] = jl.load(ROOT_REL / part2 / subj / f'tp{tp:04d}.jl')

        res = Box()
        for cond in ['AA', 'BB', 'AB', 'BA']:
            for i in range(15):
                for j in range(1001):
                    for k in range(64):
                        dat[cond][i, j, k, k] = np.nan
            res[cond] = Box()
            res[cond].data = dat[cond]
            res[cond].dec = np.zeros((15, 1001, 5))
            res[cond].dec[:, :, 0] = np.nanmean(dat[cond], axis=(2, 3))
            for i in range(15):
                for j in range(1001):
                    for k in range(64):
                        dat[cond][i, j, k, k] = 0

        for cond in ['AA', 'BB', 'AB', 'BA']:
            dsm_data = []
            for i in range(15):
                for tp in range(1001):
                    _tmp = dat[cond][i, tp, :, :]
                    _tmp = np.tril(_tmp) + np.tril(_tmp).T
                    np.fill_diagonal(_tmp, 0)
                    dsm_data.append(_tmp)

            dsm_model = [mb.rdm_race, mb.rdm_gender, mb.rdm_age, mb.rdm_emotion]
            rsa_val = rsa(dsm_data, dsm_model, metric='partial-spearman', n_jobs=-1, verbose=True)
            rsa_val_origin = rsa_val.reshape([15, 1001, 4])  # type: ignore
            rsa_val_smooth = bn.move.move_mean(rsa_val_origin, 30, min_count=1, axis=1)
            res[cond].dec[:, :, 1:] = rsa_val_smooth
            res[cond].dimension = ['image', 'race', 'gender', 'age', 'emotion']

        jl.dump(res, fname_rel)

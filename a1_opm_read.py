# -*- coding: utf-8 -*-
# Authors: Wei Xu <wxu@stu.pku.edu.cn>
# License: Simplified BSD
#
# This script is used to read OPM-MEG data.
# Original OPM-MEG data is in binary format,
# which is directly exported from in-house
# OPM-MEG software. In this code, we read
# out the binary data and convert it into
# .mat format, facilitating subsequent data
# analysis. No data preprocessing is made in
# this script.

import warnings
from pathlib import Path
from typing import Tuple

import joblib as jl
import numpy as np
from box import Box
from rawutil import unpack
from scipy.io import savemat

warnings.filterwarnings("ignore")

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


def OPM_drg(file_path: Path) -> Box:
    with open(file_path, "rb") as f:
        data = Box()
        num_pkgs = unpack(">I", f.read(4))[0]
        if "data" in file_path.name:
            data.pulse_cnt = np.zeros(num_pkgs * 10)
            data.field = np.zeros(num_pkgs * 10)
            data.is_missing = False
            for i in range(num_pkgs):
                pulse_cnt = unpack(">I", f.read(4))[0]
                _ = unpack(">29b", f.read(29))
                field_array = unpack(">10f", f.read(40))
                field_array = np.array(field_array) * 1e6
                data.pulse_cnt[i * 10 : (i + 1) * 10] = np.arange(0, 20, 2) + pulse_cnt
                data.field[i * 10 : (i + 1) * 10] = field_array
                if i == 0:
                    min_cnt = pulse_cnt
                if i == num_pkgs - 1:
                    tmp = np.floor((pulse_cnt - min_cnt) / 20) + 1  # type: ignore
                    data.is_missing = tmp != num_pkgs
        elif "Trigger" in file_path.name:
            data.pulse_cnt = np.zeros(num_pkgs)
            data.trigger = np.zeros(num_pkgs)
            for i in range(num_pkgs):
                pulse_cnt = unpack(">I", f.read(4))[0]
                trigger = unpack(">B", f.read(1))[0]
                data.pulse_cnt[i] = pulse_cnt
                data.trigger[i] = trigger

    return data


def OPM_chn(folder_name: Path, sess: int) -> Tuple[np.ndarray, int]:
    channels = list(range(1, 1 + 64))
    min_cnt = np.zeros(len(channels))
    max_cnt = np.zeros(len(channels))

    file_list = [folder_name / f"data{chn}_{sess}.drg" for chn in channels]
    data_all = jl.Parallel(n_jobs=8)(jl.delayed(OPM_drg)(_f) for _f in file_list)

    for i, data in enumerate(data_all):
        min_cnt[i] = np.floor(data.pulse_cnt[0] / 2)  # type: ignore
        max_cnt[i] = np.floor(data.pulse_cnt[-1] / 2)  # type: ignore

    min_median = np.median(min_cnt)
    max_median = np.median(max_cnt)

    min_outlier = np.zeros(len(channels), dtype=bool)
    max_outlier = np.zeros(len(channels), dtype=bool)

    min_outlier = abs(min_cnt - min_median) > 200
    max_outlier = abs(max_cnt - max_median) > 200

    min_cnt[min_outlier] = 1e20
    max_cnt[max_outlier] = 0

    bad_channel = np.logical_or(min_outlier, max_outlier)

    cnt_start, cnt_end = int(min(min_cnt)), int(max(max_cnt))
    len_data = cnt_end - cnt_start + 1
    field_raw = np.zeros((len_data, len(channels)))

    for i, data in enumerate(data_all):
        if not bad_channel[i]:
            if data.is_missing:  # type: ignore
                warnings.warn(f"Channel {channels[i]} has data missing!!!", UserWarning)
                field_tmp = 0
                cnt = 0
                for pulse_cnt, field in zip(data.pulse_cnt, data.field):  # type: ignore
                    cnt += 1
                    pulse_cnt = pulse_cnt // 2
                    if cnt == pulse_cnt - min_cnt[i] + 1:
                        field_raw[cnt - 1, i] = field
                    else:
                        while cnt != pulse_cnt - min_cnt[i] + 1:
                            field_raw[cnt - 1, i] = field_tmp
                            cnt += 1
                        field_raw[cnt - 1, i] = field
                    field_tmp = field
            else:
                a = np.zeros((int(min_cnt[i]) - cnt_start, 1)).flatten()
                b = np.zeros((cnt_end - int(max_cnt[i]), 1)).flatten()
                field_raw[:, i] = np.concatenate((a, data.field, b), axis=0)  # type: ignore

    return field_raw, cnt_start


def OPM_evt(folder_name: Path, sess: int, min_cnt: int) -> Box:
    data = OPM_drg(folder_name / f"Trigger_{sess}.drg")
    data.pulse_cnt = data.pulse_cnt // 2 - min_cnt
    i = np.where(data.trigger > 0)[0]
    data.trigger = data.trigger[i].astype(int)
    data.pulse_cnt = data.pulse_cnt[i].astype(int)

    return data


def OPM_proc(ROOT: Path, subj: str, sess: str) -> None:
    raw_path = ROOT / "drg" / subj / sess
    channels = list(range(1, 1 + 64))
    n_sess = int(len(list(raw_path.glob("data*"))) / len(channels))

    for ii in range(1, n_sess + 1):
        fname_out = ROOT / "raw" / subj / sess / f"run{ii}.mat"

        if fname_out.exists():
            continue

        raw, min_cnt = OPM_chn(raw_path, ii)
        raw = np.append(raw, np.zeros((raw.shape[0], 1)), axis=1)
        fname_trigger = raw_path / f"Trigger_{ii}.drg"

        if fname_trigger.exists():
            tri = OPM_evt(raw_path, ii, min_cnt)
            if "resting" in str(raw_path) or "emptyroom" in str(raw_path):
                raw[5999, -1] = 1
            else:
                raw[tri.pulse_cnt - 1, -1] = tri.trigger

        savemat(fname_out, {"raw": raw, "channels": channels}, do_compression=True)

    print(f"{subj} {sess} is done...")


if __name__ == "__main__":
    for subj in SUBJ:
        for proc in ["A", "B", "emptyroom", "resting-1", "resting-2"]:
            OPM_proc(ROOT, subj, proc)

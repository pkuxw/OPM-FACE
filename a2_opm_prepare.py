# -*- coding: utf-8 -*-
# Authors: Wei Xu <wxu@stu.pku.edu.cn>
# License: Simplified BSD
#
# This script is used to complete some
# basic preprocessing steps for OPM-MEG
# data, including:
# (1) Wrapping the *.mat file into MNE-compatiple *.fif file
# (2) Notch filter the data at 50/100/150/200/250 Hz
# (3) Notch filter the data at 44 Hz
# (4) Bandpass filter the data at 1-40 Hz
# (5) Epoch the data and apply post hoc projector latency correction
# (6) ICA
# (7) Export data to a friendly format for SVM decoding

import os
import shutil
import warnings
from pathlib import Path

import joblib as jl
import mne
import numpy as np
import openpyxl
from box import Box
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
from scipy.io import loadmat

warnings.filterwarnings('ignore')

ROOT = Path('../data/')

# fmt: off
SUBJ = ['S01', 'S02', 'S03', 'S04', 'S06', 'S07', 'S08',
        'S09', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16',
        'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23']
# fmt: on


def xlsx2dict(xlsx_path):
    dict_list = []
    workbook = openpyxl.load_workbook(xlsx_path)
    sheet = workbook.active
    headers = [cell.value for cell in sheet[1]]  # type: ignore
    for row in sheet.iter_rows(min_row=2, values_only=True):  # type: ignore
        row_dict = {headers[i]: value for i, value in enumerate(row)}
        dict_list.append(Box(row_dict))
    return dict_list


def proc_data(fn_data, fn_chn=ROOT / 'raw' / 'cdef.xlsx'):
    fname_bad = str(fn_data).replace('.mat', '.jl').replace('raw', 'bad')
    fname_epo = str(fn_data).replace('.mat', '.fif').replace('raw', 'epo')

    if not Path(fname_epo).exists():
        ch_types = ['mag'] * 64 + ['stim']
        ch_names = [f'MEG{n:02}' for n in range(1, 65)] + ['STIM']

        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=1000)  # type: ignore

        data = Box(loadmat(fn_data))
        data.raw[:, :64] = data.raw[:, :64] / 1e15

        raw = mne.io.RawArray(data.raw.T, info, verbose=False)

        df_list = xlsx2dict(fn_chn)

        for ii in range(65):
            str1 = df_list[ii].loc.replace(' ', ',')
            df_list[ii].loc = np.array(eval(str1))

            if df_list[ii].kind == 1:
                df_list[ii].kind = mne.io.constants.FIFF.FIFFV_MEG_CH  # type: ignore
            elif df_list[ii].kind == 3:
                df_list[ii].kind = mne.io.constants.FIFF.FIFFV_STIM_CH  # type: ignore

            if df_list[ii].coil_type == 3024:
                df_list[ii].coil_type = mne.io.constants.FIFF.FIFFV_COIL_VV_MAG_T3  # type: ignore
            elif df_list[ii].coil_type == 0:
                df_list[ii].coil_type = mne.io.constants.FIFF.FIFFV_COIL_NONE  # type: ignore

            if df_list[ii].unit == 112:
                df_list[ii].unit = mne.io.constants.FIFF.FIFFB_CONTINUOUS_DATA  # type: ignore
            elif df_list[ii].unit == 107:
                df_list[ii].unit = mne.io.constants.FIFF.FIFFB_ISOTRAK  # type: ignore

        raw.info._unlocked = 1
        raw.info['chs'] = [i.to_dict() for i in df_list]
        raw.info['bads'] = jl.load(fname_bad)

        raw.notch_filter(freqs=np.arange(50, 251, 50), n_jobs=-1, verbose=False)
        raw.notch_filter(freqs=44, n_jobs=-1, verbose=False)
        raw.filter(l_freq=1.0, h_freq=40.0, n_jobs=-1, verbose=False)

        events = mne.find_events(raw, stim_channel='STIM')
        events = mne.pick_events(events, include=1)
        assert events.shape == (256, 3), f'Incomplete trigger in {fn_data} !!!'
        epochs = mne.Epochs(
            raw,
            events,
            tmin=-0.2 + 33 / 1000,
            tmax=0.8 + 33 / 1000,
            baseline=(-0.2 + 33 / 1000, 0 + 33 / 1000),
            preload=True,
            detrend=1,
        )

        ica_dir = ROOT / 'raw' / 'tmp_ica'
        ica_dir.mkdir(exist_ok=True)

        n_good = 64 - len(epochs.info['bads'])
        ica = ICA(n_components=n_good, max_iter='auto', random_state=666)
        ica.fit(epochs)
        ica.save(ica_dir / 'ica.fif', overwrite=True, verbose=False)

        for t in range(n_good):
            fig = ica.plot_properties(epochs, picks=[t], show=False, verbose=False)  # noqa
            plt.savefig(ica_dir / f'{t}.png')
            plt.close()

        os.startfile(ica_dir)

        _c = input('Please input components to be removed:')
        ica.exclude = eval(f'[{_c}]')
        ica.apply(epochs)
        epochs.apply_baseline((-0.2 + 33 / 1000, 0 + 33 / 1000))
        epochs.save(fname_epo, overwrite=True, verbose=False)

        shutil.rmtree(ica_dir)


if __name__ == '__main__':
    for sub in (ROOT / 'raw').glob('S*'):
        for ii in sub.rglob('*.mat'):
            proc_data(ii)

        finished = len(list(sub.rglob('*.fif'))) == len(list(sub.rglob('*.mat')))
        if finished:
            all_bads = []
            for jj in sub.rglob('*.jl'):
                all_bads += jl.load(jj)
            all_bads = list(set(all_bads))
            for sess in ['A', 'B']:
                sess_dir = sub / sess
                fname_res = ROOT / 'dec' / f'{sub.name}-{sess}.jl'
                if sess_dir.exists():
                    fname_res.parent.mkdir(exist_ok=True, parents=True)
                    if not fname_res.exists():
                        res = []
                        n_run = len(list(sess_dir.glob('*.fif')))
                        for ii in range(n_run):
                            tmp = mne.read_epochs(sess_dir / f'epo-run{1+ii}.fif')
                            tmp.info['bads'] = all_bads
                            res.append(tmp)
                        XX = mne.concatenate_epochs(res).get_data('data') * 1e14  # (960, 29, 90)
                        behv_list = []
                        sess_dir = ROOT / 'beh' / sub.name / sess
                        for ii in range(len(res)):
                            fn_exp = sess_dir / f'run{1+ii:02}.mat'
                            data = loadmat(fn_exp)
                            Y = data['TABLE'][data['TABLE'][:, 1] == 0, 2]
                            behv_list.append(Y)

                        YY = np.concatenate(behv_list).astype(int)
                        jl.dump([XX, YY], fname_res)

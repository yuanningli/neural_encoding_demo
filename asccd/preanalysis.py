import os

asccd_data_path = os.path.join(os.path.dirname(__file__), 'data', 'asccd')
subject_data_path = os.path.join(os.path.dirname(__file__), 'subjects')
preanalysis_results_path = os.path.join(os.path.dirname(__file__), 'preanalysis_results')

import numpy as np
import scipy.io as sio
from scipy.stats import zscore
from scipy.signal import resample

from cl_iotools import HTK
from . import util
from . import timit
import h5py

def get_subject_block_path(subject, block):
    subject_block = subject + "_B" + str(block)
    return os.path.join(subject_data_path, subject, subject_block)


@util.save(preanalysis_results_path, "hg", "hg")
def preanalyze_data_HS(subject, block, hz, chans=np.arange(128), car=False):
    """Load high-gamma htks and downsample. Takes parameters subject, block, and hz
    """
    subject_block = subject + "_B" + str(block) + "_70_150.mat"
    print("Reading .mat")
    #hg_htk = HTK.readHTKs(os.path.join(subject_data_path, subject, subject_block, "HilbAA_70to150_8band_noCAR"), chans)['data']
    hg_htk = sio.loadmat(os.path.join(subject_data_path, subject, subject_block))['bands']
    assert hz in [100, 25]
    resampling_factor = 400/hz
    print("Resampling")
    n = hg_htk.shape[1]
    y = np.floor(np.log2(n))
    nextpow2 = np.power(2, y+1)
    hg_htk = np.pad(hg_htk , ((0,0), (0, int(nextpow2-n))), mode='constant')
    hg = resample(hg_htk, np.int(hg_htk.shape[1]/resampling_factor), axis=1)
    hg = hg[:, :np.int(n/resampling_factor)]

    hg = nan_zscore_hg(hg)

    if car is True:
        car = np.expand_dims(np.nanmean(hg, axis=0), axis=0)
        hg_car = hg - np.repeat(car, 256, axis=0)
        hg_car = nan_zscore_hg(hg_car)
        hg = hg_car

    return hg


@util.save(preanalysis_results_path, "hg", "hg")
def preanalyze_data(subject, block, hz, chans=np.arange(128), car=False):
    """Load high-gamma htks and downsample. Takes parameters subject, block, and hz
    """
    subject_block = subject + "_B" + str(block)
    print("Reading HTKs")
    #hg_htk = HTK.readHTKs(os.path.join(subject_data_path, subject, subject_block, "HilbAA_70to150_8band_noCAR"), chans)['data']
    hg_htk = HTK.readHTKs(os.path.join(subject_data_path, subject, subject_block, "HilbAA_70to150_8band"), chans)[
        'data']
    assert hz in [100, 25]
    resampling_factor = 400/hz
    print("Resampling")
    n = hg_htk.shape[1]
    y = np.floor(np.log2(n))
    nextpow2 = np.power(2, y+1)
    hg_htk = np.pad(hg_htk , ((0,0), (0, int(nextpow2-n))), mode='constant')
    hg = resample(hg_htk, np.int(hg_htk.shape[1]/resampling_factor), axis=1)
    hg = hg[:, :np.int(n/resampling_factor)]

    hg = nan_zscore_hg(hg)

    if car is True:
        car = np.expand_dims(np.nanmean(hg, axis=0), axis=0)
        hg_car = hg - np.repeat(car, 256, axis=0)
        hg_car = nan_zscore_hg(hg_car)
        hg = hg_car

    return hg

def load_hg(subject, block, hz=100):
    subject_block = subject + "_B" + str(block)
    hg_path = os.path.join(preanalysis_results_path, subject_block + "_" + str(hz) + "hz_hg.mat")
    return sio.loadmat(hg_path)['hg']

def nan_zscore_hg(hg):
    hg_to_return = []
    for hg_chan in hg:
        z = np.copy(hg_chan)
        z[~np.isnan(z)] = zscore(z[~np.isnan(z)])
        hg_to_return.append(z)
    return np.array(hg_to_return)    

def get_bad_channels(subject, block):
    subject_block = subject + "_B" + str(block)
    bcs_path = os.path.join(subject_data_path, subject, subject_block, 'Artifacts', 'badChannels.txt')
    with open(bcs_path) as f:
        bad_channels = f.read().strip().split()
    return [int(b) for b in bad_channels]

def get_anin_HS(subject, block, anin_chan=2):
    subject_block = subject + "_B" + str(block) + "_anin.hdf5"
    anin_path = os.path.join(subject_data_path, subject, subject_block)
    anin_htk = h5py.File(anin_path, 'r')
    anin_signal = np.expand_dims(anin_htk['anin_data'][anin_chan-1], axis=0)
    anin_fs = int(np.array(anin_htk['anin_fs']))
    return anin_signal, anin_fs

def get_anin(subject, block, anin_chan=2):
    subject_block = subject + "_B" + str(block)
    anin_path = os.path.join(subject_data_path, subject, subject_block, "Analog", "ANIN" + str(anin_chan) + ".htk")
    anin_htk = HTK.readHTK(anin_path)
    anin_signal = anin_htk['data']
    anin_fs = anin_htk['sampling_rate']
    return anin_signal, anin_fs

def load_times(subject_number, block):
    subject_block = util.get_subject_block(subject_number, block)
    times_path = os.path.join(preanalysis_results_path, subject_block + "_times.mat")
    return sio.loadmat(times_path)['times']



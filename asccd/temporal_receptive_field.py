"""Code for fitting temporal receptive models using Ridge (L2 regularized) regression.

Given two ndarrays, stim and resp, there are two ways to fit trfs in terms of the cross validation strategy.
Both CV strategies split the data into three mutually exclusive sets, training, validation (ridge), and test.
The regression weights are fit to the training data. The best ridge parameter is found by testing on the ridge
set. The final model performance (correlation between actual and predicted response) is calculated from the
test set. 

The two strategies are:

1. Simple KFold: run_cv_temporal_ridge_regression_model
    The total samples of the data are split into a (K-1)/K train set, 1/2K ridge set, and 1/2K test set
    K times.

2. User-defined: run_cv_temporal_ridge_regression_model_fold
    Use this function when you want to specify your own train, ridge, and test sets (e.g. I use this to
    make sure training sets have TIMIT sentences with low-to-high pitch variability so that I don't end up
    with a training set that is only low pitch variability or only high pitch variability)
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
from scipy.signal import resample
from . import timit
from . import util

def get_alphas(start=2, stop=7, num=10):
    """Returns alphas from num^start to num^stop in log space.
    """
    return np.logspace(start, stop, num)

def get_delays(delay_seconds=0.4, fs=100):
    """Returns 1d array of delays for a given window (in s). Default sampling frequency (fs) is 100Hz.
    """
    return np.arange(np.floor(delay_seconds * fs), dtype=int)

def get_dstim_with_different_delays(stim_list, delays_list, add_edges=True):
    dstims = []
    dstim_lens = []
    for stim, delays in zip(stim_list, delays_list):
        dstim = get_dstim(stim, delays, add_edges=add_edges)
        dstims.append(dstim)
        dstim_lens.append(dstim.shape[1])
    return np.concatenate(dstims, axis=1), dstim_lens

def get_dstim(stim, delays=get_delays(), add_edges=True):
    """Returns stimulus features with given delays.

    Args:
        stim: (n_samples, n_features)
        delays: list of delays to use, values in delays have units of indices for stim.
        add_edges: adds 3 additional delays to both sides of the delays list to account for edge effects in temporal
            receptive fields.

    Returns:
        dstim (ndarray): (n_samples, n_features x n_delays (including edge delays if added))
    """
    n_samples, n_features = stim.shape
    if add_edges:
        step = delays[1] - delays[0]
        delays_beg = [delays[0]-3*step, delays[0]-2*step, delays[0]-step]
        delays_end = [delays[-1]+step, delays[-1]+2*step, delays[-1]+3*step]
        delays = np.concatenate([delays_beg, delays, delays_end])
    dstim = []
    for i, d in enumerate(delays):
        dstim_slice = np.zeros((n_samples, n_features))
        if d<0:
            dstim_slice[:d, :] = stim[-d:, :]
        elif d>0:
            dstim_slice[d:, :] = stim[:-d, :]
        else:
            dstim_slice = stim.copy()
        dstim.append(dstim_slice)

    dstim = np.hstack(dstim)
    return dstim

def run_cv_temporal_ridge_regression_model_with_dstim(dstim, resp, alphas=get_alphas(), n_folds=5):
    """Given stim and resp, fit temporal receptive fields using ridge regression and KFold cross validation.

    Args:
        stim: (n_samples, n_features)
        resp: (n_samples, n_chans)
        delays: (n_delays)
        alphas: (n_alphas)
        n_folds (int): number of folds to use for KFold cross validation. The 1/K fraction of data usually used
            for the test set is split in half for the ridge parameter validation set and the test set.

    Returns:
        (tuple)
            * **test_corr_folds** (*ndarray*): Correlation between predicted and actual responses on
                test set using wts computed for alpha with best performance on validation set. 
                Shape of test_corr_folds is (n_folds, n_chans)
            * **wts_folds** (*ndarray*): Computed regression weights. Shape of wts_folds is 
                (n_folds, n_features, n_chans)
    """
    n_features = dstim.shape[1]
    n_chans = resp.shape[1]

    test_corr_folds = np.zeros((n_folds, n_chans))
    wts_folds = np.zeros((n_folds, n_features, n_chans))
    best_alphas = np.zeros((n_folds, n_chans))

    kf = model_selection.KFold(n_splits=n_folds)

    for i, (train, test) in enumerate(kf.split(dstim)):
        print('Running fold ' + str(i) + ".", end=" ")

        train_stim = dstim[train, :]
        train_resp = resp[train, :]

        #Use half of the test set returned by KFold for validation and half for test.
        ridge_stim = dstim[test[:round(len(test)/2)], :]
        ridge_resp = resp[test[:round(len(test)/2)], :]
        test_stim = dstim[test[round(len(test)/2):], :]
        test_resp = resp[test[round(len(test)/2):], :]

        wts_alphas, ridge_corrs = run_ridge_regression(train_stim, train_resp, ridge_stim, ridge_resp, alphas)
        best_alphas[i, :] = ridge_corrs.argmax(0) #returns array with length nchans. 
        best_alphas = best_alphas.astype(np.int)

        #For each chan, see which alpha did the best on the validation and choose the wts for that alpha
        best_wts = [wts_alphas[best_alphas[i, chan], :, chan] for chan in range(n_chans)]
        test_pred = [np.dot(test_stim, best_wts[chan]) for chan in range(n_chans)]
        test_corr = np.array([np.corrcoef(test_pred[chan], test_resp[:, chan])[0,1] for chan in range(resp.shape[1])])
        test_corr[np.isnan(test_corr)] = 0

        test_corr_folds[i, :] = test_corr
        wts_folds[i, :, :] = np.array(best_wts).T

    return test_corr_folds, wts_folds, best_alphas

def run_cv_temporal_ridge_regression_model(stim, resp, delays=get_delays(), alphas=get_alphas(), 
                                           n_folds=5, add_edges=True, pred=False):
    """Given stim and resp, fit temporal receptive fields using ridge regression and KFold cross validation.

    Args:
        stim: (n_samples, n_features)
        resp: (n_samples, n_chans)
        delays: (n_delays)
        alphas: (n_alphas)
        n_folds (int): number of folds to use for KFold cross validation. The 1/K fraction of data usually used
            for the test set is split in half for the ridge parameter validation set and the test set.

    Returns:
        (tuple)
            * **test_corr_folds** (*ndarray*): Correlation between predicted and actual responses on
                test set using wts computed for alpha with best performance on validation set. 
                Shape of test_corr_folds is (n_folds, n_chans)
            * **wts_folds** (*ndarray*): Computed regression weights. Shape of wts_folds is 
                (n_folds, n_features, n_chans)
    """
    if delays.size > 0:
        dstim = get_dstim(stim, delays, add_edges=add_edges)
    else:
        dstim = stim.copy()

    n_features = dstim.shape[1]
    n_chans = resp.shape[1]

    test_corr_folds = np.zeros((n_folds, n_chans))
    wts_folds = np.zeros((n_folds, n_features, n_chans))
    best_alphas = np.zeros((n_folds, n_chans))

    kf = model_selection.KFold(n_splits=n_folds)
    
    if pred:
        pred_all = np.zeros(resp.shape)

    for i, (train, test) in enumerate(kf.split(dstim)):
        print('Running fold ' + str(i) + ".", end=" ")

        train_stim = dstim[train, :]
        train_resp = resp[train, :]

        #Use half of the test set returned by KFold for validation and half for test.
        ridge_stim = dstim[test[:round(len(test)/2)], :]
        ridge_resp = resp[test[:round(len(test)/2)], :]
        test_stim = dstim[test[round(len(test)/2):], :]
        test_resp = resp[test[round(len(test)/2):], :]

        wts_alphas, ridge_corrs = run_ridge_regression(train_stim, train_resp, ridge_stim, ridge_resp, alphas)
        best_alphas[i, :] = ridge_corrs.argmax(0) #returns array with length nchans. 
        best_alphas = best_alphas.astype(np.int)

        #For each chan, see which alpha did the best on the validation and choose the wts for that alpha
        best_wts = [wts_alphas[best_alphas[i, chan], :, chan] for chan in range(n_chans)]
        test_pred = [np.dot(test_stim, best_wts[chan]) for chan in range(n_chans)]
        test_corr = np.array([np.corrcoef(test_pred[chan], test_resp[:, chan])[0,1] for chan in range(resp.shape[1])])
        test_corr[np.isnan(test_corr)] = 0

        test_corr_folds[i, :] = test_corr
        wts_folds[i, :, :] = np.array(best_wts).T
        
        if pred:
            pred_all[test, :] = np.array([np.dot(dstim[test, :], best_wts[chan]) for chan in range(n_chans)]).T
    if pred:
        return test_corr_folds, wts_folds, best_alphas, pred_all
    else:
        return test_corr_folds, wts_folds, best_alphas

def run_cv_temporal_ridge_regression_model_fold(stims, resps, delays=get_delays(), alphas=get_alphas()):
    """Fit trf models with user-given split of data into training, validation, and test.

    Args:
        stims (list): list of stim data split into training, validation, and test
            i.e. [train_stim, ridge_stim, test_stim] where train_stim is (n_training_samples x n_features)
        resps (list): list of resp data split into training, validation, and test.
            The number of samples in each set should match that for the stims.

    Returns:
        (tuple)
            * **test_corr** (*ndarray*): Correlation between predicted and actual responses on
                test set using wts computed for alpha with best performance on validation set. 
                Shape of test_corr is (n_chans)
            * **wts** (*ndarray*): Computed regression weights using best alpha from testing on 
                validation set. Shape of wts is (n_chans, n_features)

    """
    if delays.size > 0:
        dstims = [get_dstim(stim, delays) for stim in stims]
    else:
        dstims = [stim.copy() for stim in stims]
    n_chans = resps[0].shape[1]

    wts_alphas, ridge_corrs_alphas = run_ridge_regression(dstims[0], resps[0], dstims[1], resps[1], alphas)
    best_alphas = ridge_corrs_alphas.argmax(0) #returns array with length nchans.
    best_wts = [wts_alphas[best_alphas[chan], :, chan] for chan in range(n_chans)]
    test_pred = [np.dot(dstims[2], best_wts[chan]) for chan in range(n_chans)]
    test_corr = np.array([np.corrcoef(test_pred[chan], resps[2][:, chan])[0,1] for chan in range(n_chans)])
    test_corr[np.isnan(test_corr)] = 0

    wts = np.array(best_wts)
    return test_corr, wts

def run_ridge_regression(train_stim, train_resp, ridge_stim, ridge_resp, alphas):
    """Runs ridge (L2 regularized) regression for ridge parameters in alphas and returns wts fit
    on training data and correlation between actual and predicted on validation data for each alpha.

    Args:
        train_stim: (n_training_samples x n_features)
        train_resp: (n_training_samples x n_chans)
        ridge_stim: (n_validation_samples x n_features)
        ridge_resp: (n_validation_samples x n_chans)
        alphas: 1d array with ridge parameters to use

    Returns:
        (tuple):
            * **wts** (*ndarray*): Computed regression weights. Shape of wts is 
                (n_alphas, n_features, n_chans)
            * **ridge_corrs** (*ndarray*): Correlation between predicted and actual responses on
                ridge validation set. Shape of ridge_corrs is (n_alphas, n_chans)

    For multiple regression with stim X and resp y and wts B:

    1. XB = y
    2. X'XB = X'y
    3. B = (X'X)^-1 X'y

    Add L2 (Ridge) regularization:

    4. B = (X'X + aI)^-1 X'y

    Because covariance X'X is a real symmetric matrix, we can decompose it to QLQ', where
    Q is an orthogonal matrix with the eigenvectors and L is a diagonal matrix with the eigenvalues
    of X'X. Furthermore, (QLQ')^-1 = QL^-1Q'

    5. B = (QLQ' + aI)^-1 X'y
    6. B = Q (L + aI)^-1 Q'X'y

    Variables in code below:

    * `covmat` is X'X
    * `l` contains the diagonal entries of L
    * `Q` is Q
    * `Usr` is Q'X'y
    * `D_inv` is (L + aI)^-1

    The wts (B) can be calculated by the matrix multiplication of [Q, D_inv, Usr]
    """
    n_features = train_stim.shape[1] #stim shape is time x features
    n_chans = train_resp.shape[1] #resp shape is time x channels
    n_alphas = alphas.shape[0]

    wts = np.zeros((n_alphas, n_features, n_chans))
    ridge_corrs = np.zeros((n_alphas, n_chans))

    dtype = np.single
    covmat = np.array(np.dot(train_stim.astype(dtype).T, train_stim.astype(dtype)))
    l, Q = np.linalg.eigh(covmat)
    Usr = np.dot(Q.T, np.dot(train_stim.T, train_resp))

    for alpha_i, alpha in enumerate(alphas):
        D_inv = np.diag(1/(l+alpha)).astype(dtype)
        wt = np.array(np.dot(np.dot(Q, D_inv), Usr).astype(dtype))
        pred = np.dot(ridge_stim, wt)
        ridge_corr = np.zeros((n_chans))
        for i in range(ridge_resp.shape[1]):
            ridge_corr[i] = np.corrcoef(ridge_resp[:, i], pred[:, i])[0, 1]
        ridge_corr[np.isnan(ridge_corr)] = 0

        ridge_corrs[alpha_i, :] = ridge_corr
        wts[alpha_i, :, :] = wt

    return wts, ridge_corrs

def get_all_pred(wts, dstim):
    all_pred = np.array([np.dot(dstim, wts[chan]) for chan in range(wts.shape[0])])
    return all_pred

def reshape_wts_to_2d(wts, delays_used=get_delays(), delay_edges_added=True):
    """Expand the 1d array of wts to the 2d shape of n_delays x n_features.

    Args:
        wts: (n_chans, n_features x n_delays)

    Returns:
        wts_2d: (n_chans, n_delays, n_features)
    """
    n_chans = wts.shape[0]
    n_delays = len(delays_used) + 6 if delay_edges_added else len(delays_used)
    n_features = np.int(wts.shape[1]/n_delays)
    print(n_features)
    if delay_edges_added:
        wts_2d = wts.reshape(n_chans, n_delays, n_features)[:, 3:-3, :]
    else:
        wts_2d = wts.reshape(n_chans, n_delays, n_features)
    return wts_2d

__all__ = ['get_alphas', 'get_delays', 'run_cv_temporal_ridge_regression_model_fold', 'get_dstim', 
           'run_cv_temporal_ridge_regression_model', 'get_all_pred', 'run_ridge_regression']

def get_trf_stim_resp(all_hgs, all_times, all_names, shuffle_i=None):
    stim_resp = [[], [], [], [], [], [], []]

    for hg, times, names in zip(all_hgs, all_times, all_names):
        stim_resp_single = get_trf_stim_resp_single_block(hg, times, names, shuffle_i=shuffle_i)
        for i, s in enumerate(stim_resp_single):
            stim_resp[i].extend(s)
    
    return stim_resp

def get_trf_stim_resp_single_block(hg, times, names, shuffle_i=None):
    annotations = timit.get_timit_annotations()

    abs_bin_edges, rel_bin_edges, change_bin_edges = get_bin_edges_abs_rel_change(annotations)

    Ys = []
    all_spec_features = []
    all_intensity_features = []
    all_binary_pitch_features = []
    all_abs_pitch_features = []
    all_rel_pitch_features = []
    all_pitch_change_features = []
    
    for onset_time, name in zip(times[0], names):
        if onset_time < 0:
            continue

        duration = len(annotations.loc[name])
        onset_index = util.time_to_index(onset_time)
        Y = hg[:, int(onset_index): int(onset_index + duration)]
        Ys.append(Y.T)
        
        spec_features = timit.get_mel_spectrogram(name, time_bin=10, n_mels=30)[:, :duration]
        all_spec_features.append(spec_features.T)

        intensity_features = np.asarray(annotations.loc[name]['zscore_intensity'])[:, np.newaxis]
        intensity_features[np.isnan(intensity_features)] = 0
        all_intensity_features.append(intensity_features)

        abs_pitch = annotations.loc[name]['abs_pitch'].values
        rel_pitch = annotations.loc[name]['rel_pitch_global'].values
        pitch_change = annotations.loc[name]['abs_pitch_change'].values
        
        stim_abs_pitch = get_pitch_matrix(abs_pitch, abs_bin_edges)
        stim_rel_pitch = get_pitch_matrix(rel_pitch, rel_bin_edges)
        stim_pitch_change = get_pitch_matrix(pitch_change, change_bin_edges)

        binary_pitch = np.any(stim_abs_pitch, axis=1).astype(np.int)
        all_binary_pitch_features.append(binary_pitch[:, np.newaxis])

        all_abs_pitch_features.append(stim_abs_pitch)
        all_rel_pitch_features.append(stim_rel_pitch)
        all_pitch_change_features.append(stim_pitch_change)

    return all_spec_features, all_intensity_features, all_binary_pitch_features, all_abs_pitch_features, all_rel_pitch_features, all_pitch_change_features, Ys

def get_bin_edges_percent_range(a, bins=10, percent=95):
    assert percent > 1 
    assert percent < 100
    tail_percentage = (100 - percent)/2
    a = a[~np.isnan(a)]
    a_range = np.percentile(a, [tail_percentage, 100-tail_percentage])
    counts, bin_edges = np.histogram(a, bins=bins, range=a_range)
    return bin_edges

def get_bin_edges_abs_rel_change(corpus_pitch=None, bins=10, percent=95):
    """Returns abs_bin_edges, rel_bin_edges, and change_bin_edges"""
    if corpus_pitch is None:
        corpus_pitch = timit.get_timit_annotations()
        abs_bin_edges = get_bin_edges_percent_range(corpus_pitch['abs_pitch_global'], bins=bins, percent=percent)
    else:
        abs_bin_edges = get_bin_edges_percent_range(corpus_pitch['abs_pitch'], bins=bins, percent=percent)
    rel_bin_edges = get_bin_edges_percent_range(corpus_pitch['rel_pitch_global'], bins=bins, percent=percent)
    change_bin_edges = get_bin_edges_percent_range(corpus_pitch['abs_pitch_change'], bins=bins, percent=percent)
    return abs_bin_edges, rel_bin_edges, change_bin_edges

def get_pitch_matrix(pitch, bin_edges, keep_extremes=True):
    if keep_extremes:
        pitch[pitch < bin_edges[0]] = bin_edges[0] + 0.0001
        pitch[pitch > bin_edges[-1]] = bin_edges[-1] - 0.0001
    bin_indexes = np.digitize(pitch, bin_edges) - 1
    stim_pitch = np.zeros((len(pitch), len(bin_edges)))
    for i, b in enumerate(bin_indexes):
        if b < len(bin_edges):
            stim_pitch[i, b] = 1
    return stim_pitch[:, :-1]

def concatenate_trf_stim_resp(stim_resp, exclude=None):
    all_spec_features, all_intensity_features, all_binary_pitch_features, all_abs_pitch_features, all_rel_pitch_features, all_pitch_change_features, Ys = stim_resp

    Y = np.concatenate(Ys)
    fs_spec = np.concatenate(all_spec_features)
    fs_int = np.concatenate(all_intensity_features)
    fs_bin = np.concatenate(all_binary_pitch_features)
    fs_abs = np.concatenate(all_abs_pitch_features)
    fs_rel = np.concatenate(all_rel_pitch_features)
    fs_change = np.concatenate(all_pitch_change_features)

    if exclude is None:
        features = np.concatenate([fs_spec, fs_int, fs_bin, fs_abs, fs_rel, fs_change], axis=1)
    elif exclude == "abs":
        features = np.concatenate([fs_spec, fs_int, fs_bin, fs_rel, fs_change], axis=1)
    elif exclude == "change":
        features = np.concatenate([fs_spec, fs_int, fs_bin, fs_abs, fs_rel], axis=1)
    elif exclude == "rel":
        features = np.concatenate([fs_spec, fs_int, fs_bin, fs_abs, fs_change], axis=1)
    elif exclude == "rel_change":
        features = np.concatenate([fs_spec, fs_int, fs_bin, fs_abs], axis=1)
    elif exclude == "spectrum only":
        features = fs_spec
    elif exclude == "spectrum":
        features = np.concatenate([fs_int, fs_bin, fs_abs, fs_rel, fs_change], axis=1)

    return features, Y

def plot_trf(wts, chan, wts_shape=(46, 23), wts1=(0, 10), wts2=(10, 20), min_max=(0, 20), wts1_label=None,
             wts2_label=None, edges_added=True, figsize=(3.5, 5), axs=None):
    if axs is None:
        return_fig = True
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        return_fig = False

    if edges_added:
        edge = 3
    else:
        edge = 0
    min_value = np.min(wts[chan].reshape(*wts_shape)[edge:wts_shape[0] - edge, min_max[0]:min_max[1]])
    max_value = np.max(wts[chan].reshape(*wts_shape)[edge:wts_shape[0] - edge, min_max[0]:min_max[1]])
    abs_value = np.max(np.abs([min_value, max_value]))
    if abs_value < 0.01:
        abs_value = 0.01
    min_value = -1 * abs_value
    max_value = abs_value
    im1 = axs[0].imshow(
        np.fliplr(np.flipud(wts[chan].reshape(*wts_shape)[edge:wts_shape[0] - edge, wts1[0]:wts1[1]].T)),
        cmap=plt.get_cmap('RdBu_r'), aspect="auto")
    im3 = axs[1].imshow(
        np.fliplr(np.flipud(wts[chan].reshape(*wts_shape)[edge:wts_shape[0] - edge, wts2[0]:wts2[1]].T)),
        cmap=plt.get_cmap('PuOr_r'), aspect="auto")
    for im in [im1, im3]:
        im.set_clim((-1 * abs_value, abs_value))
    min_tick_value = np.trunc(np.ceil(min_value * 100)) / 100
    max_tick_value = np.trunc(np.floor(max_value * 100)) / 100
    plt.colorbar(im1, ax=axs[0], ticks=[min_tick_value, 0, max_tick_value], aspect=10)
    plt.colorbar(im3, ax=axs[1], ticks=[min_tick_value, 0, max_tick_value], aspect=10)
    if wts1_label is None:
        im1.axes.set(yticks=(0, 3, 6, 9), yticklabels=[250, 200, 150, 90], ylabel="Absolute pitch (Hz)")
    else:
        im1.axes.set(**wts1_label)
    if wts2_label is None:
        im3.axes.set(xticks=[0, wts_shape[0] - (2 * edge) - 1], xticklabels=[(wts_shape[0] - (2 * edge)) * 10, 0],
                     xlabel="Delay (ms)",
                     yticks=(0, 3, 6, 9), yticklabels=[1.7, 0.6, -0.5, -1.7], ylabel="Relative pitch (z-score)")
    else:
        im3.axes.set(**wts2_label)

    if return_fig:
        plt.gcf().tight_layout()

    return plt.gcf()

def get_all_features(all_hgs, all_times, all_names, feature_names, nn_features, all_latent_features):
    stim_resp = {}

    annotations = timit.get_timit_annotations()
    abs_bin_edges, rel_bin_edges, change_bin_edges = get_bin_edges_abs_rel_change(annotations)

    for i in range(len(feature_names)):
        stim_resp[feature_names[i]] = []

    for i in range(len(all_hgs)):   # loop through blocks
        hg = all_hgs[i]

        # neural signals & speech features
        for j in range(len(all_times[i][0])):  # loop through trials
            name = all_names[i][j]
            onset_time = all_times[i][0][j]

            if onset_time < 0:
                continue

            # high gamma response, auditory nerve, inferior colliculus
            duration = len(annotations.loc[name])
            onset_index = util.time_to_index(onset_time)
            Y = hg[:, int(onset_index): int(onset_index + duration)]
            stim_resp['hg'].append(Y.T)

            wavpath = timit.get_wavpath(name)

            # neural net features
            for k in range(len(nn_features)):
                stim_resp[nn_features[k]].append(all_latent_features[nn_features[k]][i][j])

            # speech features
            # spectrogram
            spec_features = timit.get_mel_spectrogram(name, time_bin=10, n_mels=128)[:, :duration]
            stim_resp['spectrogram'].append(spec_features.T)

            # intensity, bin pitch, abs pitch, rel pitch, pitch change
            intensity_features = np.asarray(annotations.loc[name]['zscore_intensity'])[:, np.newaxis]
            intensity_features[np.isnan(intensity_features)] = 0
            stim_resp['intensity'].append(intensity_features)

            abs_pitch = annotations.loc[name]['abs_pitch'].values
            rel_pitch = annotations.loc[name]['rel_pitch_global'].values
            pitch_change = annotations.loc[name]['abs_pitch_change'].values

            stim_abs_pitch = get_pitch_matrix(abs_pitch, abs_bin_edges)
            stim_rel_pitch = get_pitch_matrix(rel_pitch, rel_bin_edges)
            stim_pitch_change = get_pitch_matrix(pitch_change, change_bin_edges)

            binary_pitch = np.any(stim_abs_pitch, axis=1).astype(np.int)
            stim_resp['bin_pitch'].append(binary_pitch[:, np.newaxis])

            stim_resp['abs_pitch'].append(stim_abs_pitch)
            stim_resp['rel_pitch'].append(stim_rel_pitch)
            stim_resp['pitch_change'].append(stim_pitch_change)

            # phonetics, peak rate, onsets
            stim_resp['phonetics'].append(annotations.loc[name][['dorsal', 'coronal', 'labial', 'high', 'front', 
       'low', 'back', 'plosive', 'fricative', 'syllabic', 'nasal', 'voiced', 'obstruent', 'sonorant']].values)

    return stim_resp

def concatenate_all_features(stim_resp, feature_names):
    for i in range(len(stim_resp['fs_ext'])):
        length = stim_resp['fs_ext'][i].shape[0]
        for j in range(len(feature_names)):
            if stim_resp[feature_names[j]][i].shape[0] != length:
                stim_resp[feature_names[j]][i] = resample(stim_resp[feature_names[j]][i], length, axis=0)

    feat_mat = {}
    for i in range(len(feature_names)):
        feat_mat[feature_names[i]] = np.concatenate(stim_resp[feature_names[i]])
    
    return feat_mat

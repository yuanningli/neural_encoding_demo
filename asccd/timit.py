import os

timit_data_path = os.path.join(os.path.dirname(__file__), 'data', 'timit')
timit_pitch_data_path = os.path.join(timit_data_path, 'timit_pitch')
processed_timit_data_path = os.path.join(os.path.dirname(__file__), 'processed_timit_data')
neural_data_path = os.path.join(os.path.dirname(__file__), 'neural_data')
preanalysis_results_path = os.path.join(os.path.dirname(__file__), 'preanalysis_results')

import glob
import pandas as pd
import h5py
import tables
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import zscore
import librosa
from scipy.io import wavfile
import scipy.io as sio

from cl_iotools import HTK
from cl_iotools.transcriptions import praatfiles
from . import match_filter
from . import util
from . import preanalysis


def get_mel_spectrograms_for_timit_block(timit_block, time_bin=10, n_mels=128):
    wavpaths = get_wavpaths_for_timit_block(timit_block)
    spectrograms = []
    for wavpath in wavpaths:
        S = util.get_mel_spectrogram_for_wavpath(wavpath, time_bin=time_bin, n_mels=n_mels)
        spectrograms.append(S)
    return spectrograms

def get_mel_spectrogram(timit_name, time_bin=10, n_mels=128):
    wavpath = get_wavpath(timit_name)
    return util.get_mel_spectrogram_for_wavpath(wavpath, time_bin=time_bin, n_mels=n_mels)

def get_wavpath(timit_name):
    wavpath = os.path.join(timit_data_path, timit_name + ".wav")
    return wavpath

def get_wavpaths_for_timit_block(timit_block):
    timit_file_path = os.path.join(timit_data_path, "TIMIT" + str(timit_block) + ".txt")
    with open(timit_file_path) as f:
        timit_names = f.readlines()
    timit_names = [token.strip() + ".wav" for token in timit_names]
    timit_wavpaths = [os.path.join(timit_data_path, name) for name in timit_names]
    return timit_wavpaths

def get_timit_names_for_timit_block(timit_block):
    timit_file_path = os.path.join(timit_data_path, "TIMIT" + str(timit_block) + ".txt")
    with open(timit_file_path) as f:
        timit_names = f.readlines()
    timit_names = [token.strip() for token in timit_names]
    return timit_names

def get_all_timit_names():
    timit_names = []
    for block in range(1, 6):
        timit_names.extend(get_timit_names_for_timit_block(block))
    return timit_names

@util.save(preanalysis_results_path, "times", "times")
def get_times(subject, block, timit_block, anin_chan=2):
    anin_signal, anin_fs = preanalysis.get_anin(subject, block, anin_chan=anin_chan)
    return find_times(anin_signal, anin_fs, timit_block)

def find_times(anin_signal, anin_fs, timit_block=1):
    timit_wavpaths = get_wavpaths_for_timit_block(timit_block)

    times = []
    for wavpath in timit_wavpaths:
        evnts = match_filter.find_time_for_one(wavpath, anin_signal, anin_fs, 1, template_start=0, template_end=None)
        anin_signal[0, np.arange(np.int(evnts[-1][0] * anin_fs), np.int(evnts[-1][1] * anin_fs))] = 0
        times.append(evnts)

    times = np.concatenate(times)
    times = np.sort(times[:,0])[np.newaxis, :]
    return times

def load_timit_block_sentence_onset_times(subject, block, timit_block):
    timit_onsets_offsets = get_timit_sent_first_phoneme_start_times()
    timit_names = get_timit_names_for_timit_block(timit_block)
    times = preanalysis.load_times(subject, block=block)
    for i in range(len(timit_names)):
        times[0][i] = times[0][i] + timit_onsets_offsets[timit_names[i]][0]/100
    return times

phoneme_order = ['d','b','g','p','k','t','jh','sh','z','s','f','th','dh','v','w','r','l','ae','aa','ay','aw','ow','ax','uw','eh','ey','ih','ux','iy','n','m','ng']

def get_timit_syllables():
    syllable_tables = []
    timit_names = []

    syll_file_names = glob.glob(os.path.join(timit_data_path, '*.syll'))
    for syll_file in syll_file_names:
        phns = pd.read_csv(syll_file, header=None, delimiter=' ', names=['start_time', 'end_time', 'phn', 'syllable_boundary', '_', '__'] )
        phns['start_time'] = phns['start_time']/16000
        phns['end_time'] = phns['end_time']/16000
        phns['silence'] = False
        phns.loc[phns.phn == 'h#', 'silence'] = True
        phns.loc[phns.phn == 'pau', 'silence'] = True
        timit_name = syll_file.split(os.sep)[-1][:-5]

        start_times = []
        end_times = []
        syllable_phns = []

        first_syllable_started = False
        for row in phns.itertuples():
            if row.syllable_boundary > 0:
                if first_syllable_started == False:
                    start_times.append(row.start_time)
                    phns = row.phn + " "
                    first_syllable_started = True
                else:                    
                    end_times.append(row.start_time)
                    syllable_phns.append(phns)
                    start_times.append(row.start_time)
                    phns = row.phn + " "
            elif row.syllable_boundary == 0 and first_syllable_started:
                if row.silence == True:
                    end_times.append(row.start_time)
                    syllable_phns.append(phns)
                    first_syllable_started = False
                else:
                    phns = phns + row.phn + " "                    

        sylls = pd.DataFrame({'start_times': start_times, 'end_times': end_times, 'syllable_phns': syllable_phns})

        syllable_tables.append(sylls)
        timit_names.append(timit_name)

    timit_syllables = pd.concat(syllable_tables, keys=timit_names, names=['timit_name', 'syllable_index'])
    

    
    filename = os.path.join(processed_timit_data_path, 'timit_syllables.h5')
    timit_syllables.to_hdf(filename, 'timit_syllables')
    return timit_syllables  

def save_timit_phonemes():
    phoneme_tables = []
    timit_names = []

    phn_file_names = glob.glob(os.path.join(timit_data_path, '*.phn'))
    for phn_file in phn_file_names:
        phns = pd.read_csv(phn_file, header=None, delimiter=' ', names=['start_time', 'end_time', 'phn'] )
        phns['start_time'] = phns['start_time']/16000
        phns['end_time'] = phns['end_time']/16000

        timit_name = phn_file.split(os.sep)[-1][:-4]

        phoneme_tables.append(phns)
        timit_names.append(timit_name)

    timit_phonemes = pd.concat(phoneme_tables, keys=timit_names, names=['timit_name', 'phoneme_index'])
    
    timit_phonemes['silence'] = False
    timit_phonemes.loc[timit_phonemes.phn == 'h#', 'silence'] = True
    timit_phonemes.loc[timit_phonemes.phn == 'pau', 'silence'] = True
    
    filename = os.path.join(processed_timit_data_path, 'timit_phonemes.h5')
    timit_phonemes.to_hdf(filename, 'timit_phonemes')
    return timit_phonemes

def get_timit_phonemes():
    filename = os.path.join(processed_timit_data_path, 'timit_phonemes.h5')
    timit_phonemes = pd.read_hdf(filename, 'timit_phonemes')
    return timit_phonemes

def load_timit_syllables():
    filename = os.path.join(processed_timit_data_path, 'timit_syllables.h5')
    timit_syllables = pd.read_hdf(filename, 'timit_syllables')
    return timit_syllables

def get_timit_annotations():
    filename = os.path.join(processed_timit_data_path, 'timit_pitch_phonetic.h5')
    timit_ = pd.read_hdf(filename)
    return timit_

def get_timit_sent_first_phoneme_start_times():
    timit_onsets_offsets = {}
    phn_file_names = glob.glob(os.path.join(timit_data_path, '*.phn'))
    for phn_file in phn_file_names:
        phns = pd.read_csv(phn_file, header=None, delimiter=' ', names=['start_time', 'end_time', 'phn'] )   
        phns['start_time'] = phns['start_time']/16000
        phns['end_time'] = phns['end_time']/16000

        timit_name = phn_file.split(os.sep)[-1][:-4]

        timit_onsets_offsets[timit_name] = np.array([phns['start_time'].values[1], phns['end_time'].values[-2]]) * 100

    return timit_onsets_offsets

def get_timit_erps(subject_number):
    timit_onsets_offsets = get_timit_sent_first_phoneme_start_times()

    out = load_tables_out(subject_number)
    number_of_trials = 0
    for trial in out:
        number_of_trials += trial.ecog.shape[2]

    Y_mat_onset = np.zeros((256, 300, number_of_trials))
    Y_mat_offset = np.zeros((256, 250, number_of_trials))

    females = np.zeros((number_of_trials))

    count = 0
    for trial in out:
        timit_name = trial._v_attrs['timit_name'][0].decode('UTF-8')
        onset = int(np.round(timit_onsets_offsets[timit_name][0]) + 50)
        offset = int(np.round(timit_onsets_offsets[timit_name][1]) + 50)

        ecog = trial.ecog.read()
        for i in range(trial.ecog.shape[2]):
            ecog_trial = ecog[:, :, i]
            if(onset+250 < ecog_trial.shape[1]):

                Y_mat_onset[:, :, count] = ecog_trial[:256, onset-50:onset+250]
                Y_mat_offset[:, :, count] = ecog_trial[:256, offset-100:offset+150]
                    
                if timit_name[0] == 'f':
                    females[count] = 1
                count = count + 1
    Y_mat_onset = Y_mat_onset[:,:,0:count]
    Y_mat_offset = Y_mat_offset[:,:,0:count]
    females = females[:count]

    return Y_mat_onset, Y_mat_offset, females

def plot_timit_onset_erps_for_subject(subject_number):
    Y_mat_onset, Y_mat_offset, females = get_timit_erps(subject_number)
    gc = np.arange(256)
    fig = plot_timit_onset_erps(Y_mat_onset, females==0, females==1, gc, x_zero=50)
    return fig

def plot_timit_onset_erps(Y_mat, indexes1, indexes2, gc=np.arange(256), x_zero=None):
    fig = plot_grid_mean_ste(Y_mat, indexes1, indexes2, gc=gc, x_zero=x_zero)
    return fig

def get_average_response_to_phonemes(out, phoneme_order=phoneme_order):
    """Returns the average response over all instances of each phoneme in TIMIT
    """
    timit_phonemes = get_timit_phonemes()
    names = [i[0] for i in out.items()] #get names of sentences that were recorded for the specific subject.
    timit_phonemes = timit_phonemes[timit_phonemes.index.get_level_values(0).isin(names)]

    #response is 500ms, 100ms before phoneme onset to 400ms after phoneme onset.
    average_response = np.zeros((256, len(phoneme_order), 50))
    for p_index, phoneme in enumerate(phoneme_order):
        phonemes = timit_phonemes[timit_phonemes.phn == phoneme]
        average_response_phoneme = np.zeros((256, 50, len(phonemes)))
        for i, trial in enumerate(phonemes.iterrows()):
            timit_name = trial[0][0]
            start_time = trial[1].start_time
            start_index = int(round((start_time - 0.1)*100) + 50) #+50 to account for 500ms offset in the neural data from generating the out file
            average_response_phoneme[:,:,i] = out[timit_name]['ecog'][:][:256,start_index:start_index+50,0]
        average_response[:,p_index,:] = np.mean(average_response_phoneme,2)

    return average_response

def get_psis(out, phoneme_order=phoneme_order):
    # timit_phonemes is a dataframe containing information about phoneme onsets in timit sentences
    timit_phonemes = get_timit_phonemes()
    names = [i[0] for i in out.items()] # timit sentences that are in a given subject's out data file
    timit_phonemes = timit_phonemes[timit_phonemes.index.get_level_values(0).isin(names)]

    psis = np.zeros((256, len(phoneme_order))) 

    #Get the distribution of high-gamma values at 110ms after phoneme onset for each electrode for each phoneme.
    #The phoneme is the key used in the dict activitiy_distributions
    activity_distributions = {}
    for phoneme in phoneme_order:
        # all instances of a specific phoneme in the set of timit sentences a subject heard
        phoneme_instances = timit_phonemes[timit_phonemes.phn == phoneme]

        activity_phoneme = np.zeros((256, len(phoneme_instances)))
        for i, trial in enumerate(phoneme_instances.iterrows()):
            timit_name = trial[0][0]
            start_time = trial[1].start_time
            index = np.int(round((start_time + 0.11)*100) + 50)

            activity_phoneme[:, i] = out[timit_name]['ecog'][:][:256,index,0].flatten()

        activity_distributions[phoneme] = activity_phoneme

    phonemes = set(phoneme_order)

    for p_index, phoneme1 in enumerate(phoneme_order):
        dist1 = activity_distributions[phoneme1]

        for phoneme2 in phonemes - set([phoneme1]):
            dist2 = activity_distributions[phoneme2]

            for chan in np.arange(256):
                z_stat, p_value = stats.ranksums(dist1[chan,:], dist2[chan,:])
                if(p_value < 0.001):
                    psis[chan, p_index] = psis[chan, p_index] + 1

    return psis.T

def save_average_response_psis_for_subject_number(subject_number, average_response, psis):
    filename = os.path.join(results_path, "EC" + str(subject_number) + "_timit_average_response_psis.mat")
    sio.savemat(filename, {'average_response': average_response, 'psis': psis})

def load_average_response_psis_for_subject_number(subject_number):
    filename = os.path.join(results_path, "EC" + str(subject_number) + "_timit_average_response_psis.mat")
    data = sio.loadmat(filename)
    return data['average_response'], data['psis']


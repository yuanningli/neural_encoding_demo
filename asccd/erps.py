import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_timelocked_activity(times, hg, hz=100, back=20, forward=100):
    times = np.array(times)
    times = times[times * hz - back > 0]
    times = times[times * hz + forward < hg.shape[1]]
    Y_mat = np.zeros((hg.shape[0], int(back + forward), len(times)), dtype=float)

    for i, seconds in enumerate(times):
        index = int(np.round(seconds * hz))
        Y_mat[:, :, i] = hg[:, int(index-back):int(index+forward)]

    return Y_mat

def get_mean_and_ste(to_average):
    """Takes chans x timepoints x trials and averages over trials returning chans x timepoints
    This function returns the average and ste over the third dimension (axis=2) and returns
    an array of the first two dimensions
    """
    average = np.nanmean(to_average, axis=2)
    ste = np.nanstd(to_average, axis=2)/np.sqrt(np.shape(to_average)[2])

    min_value = np.nanmin([average-ste])
    max_value = np.nanmax([average+ste])
    return average, ste, min_value, max_value


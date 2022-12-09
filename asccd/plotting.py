import os

subject_brain_data_path = os.path.join(os.path.dirname(__file__), 'subjects', 'brain_imaging')

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
import matplotlib
matplotlib.rcParams['font.size'] = 16

from . import preanalysis
from . import erps

tone_colors_light = [(0.38, 0.66, 0.53), (0.87, 0.45, 0.24), (0.51, 0.50, 0.74), (0.89, 0.32, 0.58)]
tone_colors = [(0.28, 0.56, 0.43), (0.77, 0.35, 0.14), (0.41, 0.40, 0.64), (0.80, 0.22, 0.49)]

def get_tone_color(tone):
    if tone == 0:
        return (0.4, 0.4, 0.4)
    elif tone in [1, 2, 3, 4]:
        return tone_colors_light[tone-1]
    else:
        return (1, 1, 1)
        
def nansem(a, axis=1):
    return np.nanstd(a, axis=1)/np.sqrt(a.shape[axis])

def plot_filled_sem(a, xvals=None, ax=None, color=None, alpha=None, ylim=None, xlabel="Time (s)", ylabel="Relative pitch (z-score)"):
    if ax is None:
        fig, ax = plt.subplots()
    mean = np.nanmean(a, axis=1)
    sem = nansem(a, axis=1)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.5)

    if alpha is None:
        alpha = 0.9
    if xvals is None:
        xvals = np.arange(mean.shape[0])
    
    if color is not None:
        h = ax.fill_between(xvals, mean-sem, mean+sem, color=color, alpha=alpha)
    else:
        h = ax.fill_between(xvals, mean-sem, mean+sem, alpha=alpha)
        
    if ylim is not None:
        ax.set(ylim=ylim)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(xvals[0], xvals[-1]))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return h

def plot_sem(a, xvals, ax=None, color=None, ylim=None, xlabel="Time (s)", ylabel="Relative pitch (z-score)"):
    if ax is None:
        fig, ax = plt.subplots()
    mean = np.nanmean(a, axis=1)
    sem = nansem(a, axis=1)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.5)

    if color is not None:
        h = ax.plot(xvals, sem, color=color)
    else:
        h = ax.plot(xvals, sem)
        
    if ylim is not None:
        ax.set(ylim=ylim)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(xvals[0], xvals[-1]))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return h

def reshape_grid(vals, subject='EC166', temporal_grid_only=True):
    if subject == 'EC166' and temporal_grid_only:
        channel_order = get_channel_order(subject=subject, temporal_grid_only=temporal_grid_only)
        return vals[np.array(channel_order)].reshape(8, 16)
    else:
        print("not implemented")

def center_colormap(im):
    val = np.max(np.abs(im.get_clim()))
    im.set_clim((-1*val, val))
    im.set_cmap(plt.get_cmap('RdBu_r'))
    return im

def equalize_colormaps(ims):
    cmins = [im.get_clim()[0] for im in ims]
    cmaxs = [im.get_clim()[1] for im in ims]
    for im in ims:
        im.set_clim([np.min(cmins), np.max(cmaxs)])

def plot_average_responses_to_phonemes(chan, average_responses, phoneme_order, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))

    im = ax.imshow(average_responses[chan].T, interpolation='none', aspect='auto')
    center_colormap(im)
    im.axes.set_yticks(np.arange(len(phoneme_order)))
    im.axes.set_yticklabels(phoneme_order)
    im.axes.axvline(10, color='k')
    plt.colorbar(im, ax=ax)
    return ax

def plot_grid_on_brain(subject):
    img = mpimg.imread(os.path.join(subject_brain_data_path, subject + "_brain2D.png"))
    xy = sio.loadmat(os.path.join(subject_brain_data_path, subject + "_elec_pos2D.mat"))['elecmatrix']

    n_chans = 256 if xy.shape[0] > 256 else 128

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(img)
    for i in range(n_chans):
        ax.plot(xy[i][0], xy[i][1], 'ro')
        ax.text(xy[i][0], xy[i][1], str(i))
    ax.axis("off")

    return fig

def get_brain(subject):
    img = mpimg.imread(os.path.join(subject_brain_data_path, subject + "_brain2D.png"))
    xy = sio.loadmat(os.path.join(subject_brain_data_path, subject + "_elec_pos2D.mat"))['elecmatrix']
    return img, xy

def show_chans_on_brain(subject, which_chans, color='r', show_grid=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    bbox = ax.get_window_extent()
    size = np.min([bbox.width, bbox.height])
    print(size)
    img, xy = get_brain(subject)
    n_chans = 256 if xy.shape[0] > 256 else 128

    ax.imshow(img)
    for i in range(n_chans):
        if i in which_chans:
            ax.plot(xy[i][0], xy[i][1], 'o', color='k', markersize=12*size/550)
            ax.plot(xy[i][0], xy[i][1], 'o', color=color, markersize=8*size/550)
        else:
            if show_grid:
                ax.plot(xy[i][0], xy[i][1], 'o', color='k', markersize=5*size/550)

    ax.axis("off")

    if ax is None:
        return fig
    
def brain_inset(axs, subject, response_channels, markersize=8, plotgrid=True, inset_xlim = (230, 590), inset_ylim = (432, 192)):
    ax_brain, ax_inset = axs

    img, xy = get_brain(subject)

    ax_brain.imshow(img, cmap='Greys_r')
    ax_brain.add_patch(matplotlib.patches.Rectangle((inset_xlim[0]-10, inset_ylim[1]-10), inset_xlim[1]-inset_xlim[0]+20, inset_ylim[0]-inset_ylim[1]+20, fill=False, linewidth=1))
    ax_brain.set(xticks=[], yticks=[])
    ax_brain.set_frame_on(False)

    if len(xy) >= 256:
        n_chans = 256
    else:
        n_chans = 128

    if plotgrid is True:
        for i in range(n_chans):
            ax_brain.plot(xy[i][0], xy[i][1], 'o', color=[0.2,0.2,0.2], markersize=1)

    ax_inset.set(xticklabels=[], yticklabels=[], yticks=[], xticks=[], xlim=inset_xlim, ylim=inset_ylim)
    
    ax_inset.imshow(img, cmap='Greys_r')
    for i in range(n_chans):
        if i in response_channels:
            ax_inset.plot(xy[i][0], xy[i][1], 'o', color='darkred', markersize=markersize/2)
        else:
            ax_inset.plot(xy[i][0], xy[i][1], 'o', color='black', markersize=np.minimum(2, markersize/2))

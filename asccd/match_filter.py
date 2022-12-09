import numpy as np
import scipy.signal
from scipy.signal import resample
from scipy.io import wavfile

def find_time_for_one(wav_path, anin_signal, anin_fs, nreps, template_start=2, template_end=10):
    [template_fs, template_sound] = wavfile.read(wav_path)
    if len(template_sound.shape) == 1:
        template_sound = np.atleast_2d(template_sound)
    else:
        template_sound = template_sound.T
    
    new_fs = anin_fs
    # Shouldn't have to zero-pad the sound here because template_sound should be relatively short
    if template_end is None:
        template_sound = template_sound[0, int(template_fs*template_start):]
    else:
        template_sound = template_sound[0, int(template_fs*template_start):int(template_fs*template_end)]

    template_sound = resample(template_sound.T, int((template_sound.size/template_fs)*anin_fs) )
        
    corr_thresh = 0.30 # Signals must be at least this correlated to be considered a good match

    [evnts, cc, matches, match_conv] = match_filter(template_sound, anin_signal[0], new_fs, corr_thresh, nreps, remove_bad_events=False, debug = True)
    evnts[:, 0] = evnts[:, 0] - template_start
    return evnts

def match_filter(template_sound, anin_signal, fs, corr_thresh = 0.8, nreps = 1, remove_bad_events = True, debug = False):
    """
    Find instances of the template signal [template_sound] in an analog channel signal [anin_signal].  Both must 
    be sampled at sampling rate [fs], so use resample before passing to this function.
    
    Parameters
    ---------------------------
      template_sound: n-length vector representing sound signal to search for

      anin_signal: m-length vector (where m > n) for ANIN1 or ANIN2 (usually you will want to search using ANIN2, 
                   the speaker channel)

      fs: sampling rate of the signals

      corr_thresh: Threshold for correlation between template and ANIN, used
                   to include or reject events

      nreps: Number of repetitions expected for this sound (if you know that
             it occurs a certain number of times.  Default = 1 repetition.)
      
      remove_bad_events: [True]/False.  Whether to remove event times that are below
                         [corr_thresh]. Sometimes when debugging you may want
                         to keep bad events.  Default = True.

      debug: True/[False]. Returns optional outputs specified below.

    Returns
    ---------------------------
      evnt_time: [nreps x 2]. Start and end time (in seconds) of the sound
                 [template_sound] as it is found in [anin_signal].

      cc: [nreps x 1]. Correlation between [template_sound] and [anin_signal]
          for each repetition. Used to determine whether event is a good
          match.

    Optional Outputs:
      matched_segment: [nreps x len(template_sound)].  Waveform on
                       [anin_signal] that was found as a match.  Should look
                       very similar to [template_sound].

      match_conv: [nreps x len(anin_signal)].  Result of the
                    convolution.  Usually only returned for debugging 
                    purposes.

    Written 2015 by Liberty Hamilton

    Example
    --------------------------
    >>> import scipy.io
    >>> dat = scipy.io.loadmat('/Users/liberty/Documents/UCSF/changrepo/matlab/preprocessing/EventDetection/sample_sounds.mat')
    >>> corr_thresh = 0.8
    >>> nreps = 12
    >>> [ev,cc] = match_filter(dat['template_sound'], dat['anin_signal'], dat['fs'], corr_thresh, nreps)
    Found a match for sentence (62.000-64.000), rep 1, r=0.890
    Found a match for sentence (168.000-170.000), rep 2, r=0.894
    Found a match for sentence (106.000-108.000), rep 3, r=0.873
    Found a match for sentence (199.000-201.000), rep 4, r=0.903
    Found a match for sentence (211.000-213.000), rep 5, r=0.845
    Found a match for sentence (90.000-93.000), rep 6, r=0.925
    Found a match for sentence (121.000-124.000), rep 7, r=0.940
    Found a match for sentence (148.000-150.000), rep 8, r=0.950
    Found a match for sentence (10.000-12.000), rep 9, r=0.968
    Found a match for sentence (33.000-35.000), rep 10, r=0.969
    Could not find a match for rep 11, best correlation was r=0.114
    Removing non-matching events with corr < 0.80
    Could not find a match for rep 12, best correlation was r=0.114
    Removing non-matching events with corr < 0.80
    """

    # Initialize variables
    evnt_time = [] # Start time and end time for each repetition of this template sound
    cc = [] # Correlation coefficient between template and the match found in anin_signal (helps determine whether event detection was successful)
    match_conv = [] # Convolution between template and anin
    matched_segment = [] # Matching segment (should look like template)

    signal = np.copy(anin_signal)

    for r in np.arange(nreps):
        # Perform circular convolution
        match_conv.append(scipy.signal.fftconvolve(template_sound[::-1], signal, mode = 'full'))

        # sort by maximum of the convolution, this tells you where the END
        # of the events occurred. We sort in case there is more than one
        # example of the sentence detected in the TDT signal
        end_time = np.argmax(match_conv[-1])
        start_time = end_time - template_sound.shape[0]
        
        if start_time < 0:
            start_time = 1
            print('Start time was negative! This is likely bad.')
        
        # Append the start and end times
        evnt_time.append( [ start_time, end_time ] )

        # Get the segment that matches according to the convolution
        matched_segment.append(signal[np.arange(np.int(evnt_time[-1][0]), np.int(evnt_time[-1][1]))])
        
        # correlation between sentence and the "match"
        cc_tmp = np.corrcoef(template_sound.ravel(), matched_segment[-1].ravel()) 
        cc.append(cc_tmp[0,1])
        
        # If the correlation is good enough, consider this a true match
        if cc[-1] > corr_thresh: # Find last element in cc list
            signal[np.arange(np.int(evnt_time[-1][0]), np.int(evnt_time[-1][1]))] = 0
            #print('Found a match for sentence (%4.3f-%4.3f), rep %d, r=%3.3f'%(evnt_time[-1][0]/fs, evnt_time[-1][1]/fs, r + 1, cc[-1]))
        else:
            print('Could not find a match for rep %d, best correlation was r=%3.3f'%(r + 1, cc[-1]))
            if remove_bad_events:
                print('Removing non-matching events with corr < %2.2f'%(corr_thresh))
                evnt_time.pop()
                matched_segment.pop()
                match_conv.pop()
                cc.pop()

    # convert event times from samples to seconds
    evnt_time = np.array(evnt_time)/np.float(fs) 
    
    if debug:
        return evnt_time, cc, matched_segment, match_conv
    else:
        return evnt_time, cc

if __name__ == "__main__":
    import doctest
    doctest.testmod()
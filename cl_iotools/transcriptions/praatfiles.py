from .labfile import labread

from collections import OrderedDict
import numpy as np
import textgrid

__all__ = ['lab2textgrid', 'write_textgrid']

def lab2textgrid(labfile, textgridfile=None, label_name='phoneme'):
    """Creates a Praat TextGrid file for a given .lab phoneme transcription.

    """
    labels = labread(labfile, add_column_for_start_time=True, as_df=True)

    tiers = {}
    tiers[label_name] = labels.values

    if textgridfile is None:
        if labfile.split(".")[-1] == 'lab':
            textgridfile = labfile[:-4]
        else:
            textgridfile = labfile

    write_textgrid(textgridfile, tiers)

def write_textgrid(filename, tiers, xmin=None, xmax=None):
    """Writes a Praat TextGrid file given a dict tiers

    tiers can be an OrderedDict for a specific order of tiers.
    The keys are the tier names and the values are a (n_labels x 3) ndarray containing
    the start_time, end_time, and label for each interval label (sentence, word, phoneme, etc.).

    """
    if filename.split('.')[-1] != 'TextGrid':
        filename = filename + '.TextGrid'

    if type(tiers) == OrderedDict:
        tiers = OrderedDict(tiers)
    else:
        tiers = OrderedDict(sorted(tiers.items(), key=lambda t: len(t[1])))

    xmins = np.array([tiers[k][0, 0] for k in tiers])
    if xmin is None:
        xmin = np.min(xmins)

    xmaxs = np.array([tiers[k][-1, 1] for k in tiers])
    if xmax is None:
        xmax = np.max(xmaxs)

    with open(filename, "w") as f:
        f.write('File type = "ooTextFile"\nObject class = "TextGrid"\n\n')
        f.write("xmin = %0.3f\nxmax = %0.3f\ntiers? <exists>\n" % (xmin, xmax))
        f.write("size = %d\nitem []:\n" % len(tiers))

        for i, (tier_name, tier) in enumerate(tiers.items()):
            f.write('\titem [%d]:\n\t\tclass = "IntervalTier"\n' % (i+1))
            f.write('\t\tname = "%s"\n' % tier_name)
            f.write('\t\txmin = %0.3f\n\t\txmax = %0.3f\n' % (xmins[i], xmaxs[i]))
            f.write('\t\tintervals: size = %d\n' % len(tier))

            for row_index, row in enumerate(tier):
                f.write('\t\tintervals [%d]\n' % (row_index + 1))
                f.write('\t\t\txmin = %0.3f\n\t\t\txmax = %0.3f\n\t\t\ttext = "%s"\n' % tuple(row))

def read_textgrid(tg_filepath):
    """
    Parses a TextGrid and returns it as a Python dictionary

    Returns:
    (dict) {
        'TextGrid': (str) name of TextGrid
        'start':    (float) global start time
        'end':      (float) global end time
        'tiers': (OrderedDict) {
            'NAME OF TIER':
                (list) [
                    [ (str) 'ANNOTATION', (float) start, (float) end ],
                    [ (str) 'ANNOTATION', (float) start, (float) end ],
                    ...
                ]
            ,
            ...
        }
    }

    """
    data = {}
    with open(tg_filepath, 'r') as f:
        # skipping the beginning metadata
        filetype = next(f).rstrip()
        print(filetype)
        # ooTextFile short is the default output of p2fa
        if filetype == 'File type = "ooTextFile short"':
            print("Found short!")
            for _ in range(2):
                next(f)
            # extracting data about the whole Grid
            start = float(next(f))
            end = float(next(f))
            data['TextGrid'] = tg_filepath
            data['start'] = start
            data['end'] = end
            interval_tier_mark = next(f).strip()
            # interval_tier_mark is a boolean for whether there exist tiers
            if interval_tier_mark == "<exists>":
                num_tiers = int(next(f))
                tier_data = OrderedDict()
                for i in range(num_tiers):
                    next(f)
                    tier_name = next(f).strip().strip('"')
                    tier_start = float(next(f))
                    tier_end = float(next(f))
                    segment_count = int(next(f))
                    segment_list = []
                    for j in range(segment_count):
                        segment_start = float(next(f))
                        segment_end = float(next(f))
                        segment_name = next(f).strip().strip('"')
                        segment_list.append([segment_name, segment_start, segment_end])
                    tier_data[tier_name] = segment_list
                data['tiers'] = tier_data
        # ooTextFile is the default output of Praat
        elif filetype == 'File type = "ooTextFile"':
            print("found long!")
            tg = textgrid.TextGrid.fromFile(tg_filepath)
            data = {}
            data['TextGrid'] = tg_filepath
            data['start'] = tg.minTime
            data['end'] = tg.maxTime
            data['tiers'] = OrderedDict()
            for i in tg:
                newtier = []
                for j in i:
                    newtier.append([j.mark, float(j.minTime), float(j.maxTime)])
                data['tiers'][i.name] = newtier
        else:
            print("didn't find it!")
    return data

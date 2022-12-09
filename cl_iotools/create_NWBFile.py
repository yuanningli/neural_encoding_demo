import argparse
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.io as sio

from pynwb import NWBFile, TimeSeries, get_build_manager
from pynwb.misc import IntervalSeries
from pynwb.ecephys import ElectricalSeries, Device
from form.backends.hdf5 import HDF5IO

from HTK import readHTK

# Establish the assumptions about file paths
raw = "RawHTK"
analog = "Analog"
artifacts = "Artifacts"
meshes = "Meshes"
elec_metadata_file = "elecs/TDT_elecs_all.mat"
audio_file = "%s/ANIN1.htk" % analog
stim_file1 = "%s/ANIN2.htk" % analog
stim_file2 = "%s/ANIN3.htk" % analog
bad_time_file = "%s/badTimeSegments.mat" % artifacts

desc = 'convert Raw ECoG data (in HTK) to NWB'
epi = 'The following directories must be present: %s, %s, %s, and %s' % (raw, analog, artifacts, elec_metadata_file)

parser = argparse.ArgumentParser(usage='%(prog)s data_dir out.nwb', description=desc, epilog=epi)
parser.add_argument('data_dir', type=str, help='the directory containing Raw ECoG data files')
parser.add_argument('out', type=str, help='the path to the NWB file to write to')
parser.add_argument('-s', '--scale', action='store_true', default=False, help='specifies whether or not to scale sampling rate')

args = parser.parse_args()

# Get the paths to all HTK files and sort them
def wav_path_key(path):
    num = path[path.find('Wav')+3:path.rfind('.')]
    return (int(num[0]), int(num[1:]))

htk_paths = sorted(glob.glob("%s/%s/Wav*.htk" % (args.data_dir, raw)), key=wav_path_key)

audio_file = "%s/%s" % (args.data_dir, audio_file)
stim_file1 = "%s/%s" % (args.data_dir, stim_file1)
stim_file2 = "%s/%s" % (args.data_dir, stim_file2)
bad_time_file = "%s/%s" % (args.data_dir, bad_time_file)

# Get metadata for all electrodes
elecs_metadata = sio.loadmat("%s/%s" % (args.data_dir, elec_metadata_file))
elec_grp_xyz_coord = elecs_metadata['elecmatrix']
anatomy = elecs_metadata['anatomy']
elec_grp_loc = [str(x[3][0]) if len(x[3]) else "" for x in anatomy]
elec_grp_dev = [str(x[2][0]) for x in anatomy]
elec_grp_desc = [str(x[1][0]) for x in anatomy]
elec_grp_name = [str(x[0][0]) for x in anatomy]
anatomy = {'loc': elec_grp_loc, 'dev': elec_grp_dev, 'desc': elec_grp_desc, 'name': elec_grp_name}
elec_grp_df = pd.DataFrame(anatomy)

# Create the NWB file object
session_description = 'conversion of %s' % args.data_dir
nwbfile = NWBFile(args.out, session_description, args.data_dir, datetime.now(), datetime.now(), "experimenter", "experiment name",
                  institution='University of California, San Francisco',
                  lab='Chang Lab')

for group in elec_grp_df.groupby("dev"):
    # Create devices
    device_name = group[0]
    device = nwbfile.create_device(device_name)

    # Create electrode groups
    n = len(group[1])
    name = group[1]['name']
    description = group[1]['desc']
    location = group[1]['loc']
    if n < len(elec_grp_xyz_coord):
        coord = elec_grp_xyz_coord[:n]
    elif n == len(elec_grp_xyz_coord):
        coord = elec_grp_xyz_coord
    else:
        coord = elec_grp_xyz_coord
        for i in range(n - len(elec_grp_xyz_coord)):
            coord.append([np.nan, np.nan, np.nan])
    electrode_group = nwbfile.create_electrode_group(device_name, description, location, [''], coord, [0], "ECoG", "", device)

    # Read electrophysiology data from HTK files and add them to NWB file
    htk = readHTK(htk_paths[0])
    htk_data = np.concatenate([readHTK(htk_paths[i], scale_s_rate=args.scale)['data'] for i in group[1].index.values])
    ts_desc = "data generated from electrode group %s, sampled at %0.6f Hz" % (device_name, htk['sampling_rate'])
    nwbfile.add_raw_timeseries(ElectricalSeries(device_name, "source", htk_data, electrode_group, starting_time=0.0, rate=htk['sampling_rate'], description=ts_desc, conversion=0.001))

# Add audio recording from room
audio_htk = readHTK(audio_file, scale_s_rate=args.scale)
ts_name = 'ANIN1'
ts_desc = "audio recording from microphone in room"
ts_source = 'microphone in room'
nwbfile.add_raw_timeseries(TimeSeries('ANIN1', ts_source, audio_htk['data'][0], 'audio unit', starting_time=0.0, rate=audio_htk['sampling_rate'], description=ts_desc))

# Add audio stimulus 1
stim_htk = readHTK(stim_file1, scale_s_rate=args.scale)
ts_name = 'ANIN2'
ts_desc = "audio stimulus 1"
ts_source = 'the first stimulus source'
nwbfile.add_stimulus(TimeSeries('ANIN2', ts_source, stim_htk['data'][0], 'audio unit', starting_time=0.0, rate=stim_htk['sampling_rate'], description=ts_desc))

# Add audio stimulus 2
stim_htk = readHTK(stim_file2, scale_s_rate=args.scale)
ts_name = 'ANIN3'
ts_desc = "audio stimulus 2"
ts_source = 'the second stimulus source'
nwbfile.add_stimulus(TimeSeries('ANIN3', ts_source, stim_htk['data'][0], 'audio unit', starting_time=0.0, rate=stim_htk['sampling_rate'], description=ts_desc))

# Add bad time segments
bad_time = sio.loadmat(bad_time_file)['badTimeSegments']
ts_name = 'badTimeSegments'
ts_source = bad_time_file               # this should be something more descriptive
ts_desc = 'bad time segments'           # this should be something more descriptive
bad_timepoints_ts = IntervalSeries(ts_name, ts_source, description=ts_desc)
for start, stop in bad_time:
    bad_timepoints_ts.add_interval(start, stop)

if len(bad_time) > 0:
    nwbfile.add_raw_timeseries(bad_timepoints_ts)

# Export the NWB file
manager = get_build_manager()
path = nwbfile.filename

io = HDF5IO(path, manager)
io.write(nwbfile)
io.close()

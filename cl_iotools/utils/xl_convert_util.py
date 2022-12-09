__author__ = 'dharshan'

import os
import abc
import shutil
import datetime
import subprocess
from win32process import DETACHED_PROCESS

import h5py
import numpy as np

from ..xltek.xlstudy import XLStudy

class ConvertUtilBase(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, srcdir, outdir):
        pass
    
    @abc.abstractmethod
    def select_study(self):
        pass
    
    @abc.abstractmethod
    def _set_study(self, studypath):
        pass
    
    @abc.abstractmethod
    def compile_study_info(self):
        pass

class H5DCtoNWB_ConvertUtil(ConvertUtilBase):
    """Placeholder with code from SUBNETS data standardization demo"""
    def nwb_convert(filepath):
        try:
            from nwb import nwb_file
            from nwb import nwb_utils as ut
        except ImportError as e:
            print ("Unable to convert data: Missing NWB support package...")
            return
        
        """Converts DC's HDF5 format for clinical data to the NWB format"""
        
        out_file_name = os.path.basename(filepath)[:-3] + '.nwb'
        subj_id = os.path.basename(filepath).split('_')[0]
        study_id = os.path.basename(filepath).split('_')[1]
        fileno = os.path.basename(filepath).split('_')[2][:-3]
        
        settings = {}
        
        settings['file_name'] = out_file_name
        settings['identifier'] = ut.create_identifier("UCSF EMU Data: Subject=({:s}) Study=({:s}) File#=({:s})".format(subj_id,study_id,fileno))
        settings['mode'] = 'w' # rewrite enabled
        settings['extensions'] = ['extensions/e-interval.py']
        
        hf = h5py.File(filepath,'r')
        
        loctz = pytz.timezone('America/Los_Angeles')
        start_time = loctz.localize(datetime.datetime.fromtimestamp(hf['/timestamp vector'][0]))
        settings['start_time'] = start_time.isoformat()
        
        settings['description'] = 'test file for UCSF EMU data'
        
        f = nwb_file.open(**settings)
        #print("Opened file: {:s}".format(settings['file_name']))
        
        test_grp = f.make_group("<ElectricalSeries>", "ecog", path='/acquisition/timeseries', attrs={'source': 'UCSF EMU Recording'})
        test_grp.set_dataset("data", hf['/ECoG Array'].value, attrs={'unit': 'microvolt'})
        times = test_grp.set_dataset("timestamps", hf['/timestamp vector'].value - hf['/timestamp vector'][0],
                                     attrs={'sampling_rate': hf['/ECoG Array'].attrs['Sampling Rate']})
        elc_idx = test_grp.set_dataset("electrode_idx", hf['/channel indices'].value)
        
        
class XLtoNWB_ConvertUtil(ConvertUtilBase):
    pass

class XLtoH5_ConvertUtil(ConvertUtilBase):
    def __init__(self):
        self.select_study()
        return

    def select_study(self):
        fpath = filedialog.askopenfilename(filetypes=[('EEG Study Info', '.eeg'),
                                                      ('All Files', '.*')],
                                           title='Select XLTEK ECoG Study')

        self.study = XLStudy(os.path.dirname(fpath))

        return

    def _set_study(self, studypath): # FOR DEBUGGING (w/o GUI)
        self.study = XLStudy(studypath)
        return

    def _seek_h5_file(self, seg_id):
        try:
            return h5py.File(os.path.join(self.study.study_dir,
                                          (seg_id + '.h5')))
        except:
            return None

    def _get_time_axis(self, seg_id):
        (toc_file, raw_file) = self.study.raw_data_segments[seg_id]
        snc_file = self.study.snc_file

        sstamp_arr = []
        for te in toc_file:
            sstamp_arr.extend(range(te.samplestamp, (te.samplestamp + te.sample_span)))
        sstamp_arr = np.array(sstamp_arr)

        tstamp_arr = []
        for si in sstamp_arr:
            snc_ref = snc_file.snc_data[np.argmin(np.array([np.abs(se.sample_stamp - si) for se in snc_file]))]
            delta_t = datetime.timedelta(seconds=((si - snc_ref.sample_stamp) * 1.0/raw_file.header.sample_freq))
            tstamp_arr.append(snc_ref.sample_time + delta_t)

        return (sstamp_arr, tstamp_arr)

    def add_h5_metadata(self): # STEP 2!
        for ii,seg_id in enumerate(self.study.raw_data_segments):
            print('Processing: Segment [%s]' % (seg_id))
            h5_file = self._seek_h5_file(seg_id)
            (ss_arr, ts_arr) = self._get_time_axis(seg_id)

            print('\tCreating dimension scale datasets...')
            ecog_dataset = h5_file['ECoG Array']
            if 'samplestamp vector' in h5_file:
                print('\t\tWarning: OVERWRITING "samplestamp vector" ...')
                h5_file['samplestamp vector'][:] = ss_arr
            else:
                h5_file.create_dataset('samplestamp vector', data=ss_arr)

            if 'timestamp vector' in h5_file:
                print('\t\tWarning: OVERWRITING "timestamp vector" ...')
                h5_file['timestamp vector'][:] = np.array([ts.timestamp() for ts in ts_arr])
            else:
                h5_file.create_dataset('timestamp vector', data=np.array([ts.timestamp() for ts in ts_arr]))

            if 'channel indices' in h5_file:
                print('\t\tWarning: OVERWRITING "channel indices" ...')
                h5_file['channel indices'][:] = np.arange(ecog_dataset.shape[1])
            else:
                h5_file.create_dataset('channel indices', data=np.arange(ecog_dataset.shape[1]))

            print('\tCreating dimension scales...')
            ecog_dataset.dims.create_scale(h5_file['samplestamp vector'], 'sample axis')
            ecog_dataset.dims.create_scale(h5_file['timestamp vector'], 'time axis')
            ecog_dataset.dims.create_scale(h5_file['channel indices'], 'channel axis')

            print('\tAttaching dimension scales...')
            ecog_dataset.dims[0].attach_scale(h5_file['samplestamp vector'])
            ecog_dataset.dims[0].attach_scale(h5_file['timestamp vector'])
            ecog_dataset.dims[1].attach_scale(h5_file['channel indices'])

            h5_file.flush()
            h5_file.close()
            print('\tDone: Segment[%d of %d] ...\n' % (ii+1, len(self.study.raw_data_segments)))

        return


class XLBuildStudyError(Exception):
    pass


class XLRDConverterExeError(Exception):
    pass


class XLRDConvertError(Exception):
    pass


class XLRDStepTwoError(Exception):
    pass


class XLERDtoH5_ConvertUtil(object):
    """Implements the 2-step Conversion Process for XLTEK EEG Raw Data (ERD) files"""
    def __init__(self, erd_input_path, output_dir):
        try:
            self.study = XLStudy(os.path.dirname(erd_input_path))
        except:
            raise XLBuildStudyError('Unable to build XLStudy from ERD file path')
        
        self.input_erd = erd_input_path
        print(self.input_erd)
        self.output_dir = output_dir
        self.output_h5 = os.path.join(output_dir,
                                      '{:s}.h5'.format('.'.join(os.path.basename(erd_input_path).split('.')[:-1])))
        
        return
    
    def _get_time_axis(self, seg_id):
        (toc_file, raw_file) = self.study.raw_data_segments[seg_id]
        snc_file = self.study.snc_file
        
        sstamp_arr = []
        for te in toc_file:
            sstamp_arr.extend(range(te.samplestamp, (te.samplestamp + te.sample_span)))
        sstamp_arr = np.array(sstamp_arr)
        
        tstamp_arr = []
        for si in sstamp_arr:
            snc_ref = snc_file.snc_data[np.argmin(np.array([np.abs(se.sample_stamp - si) for se in snc_file]))]
            delta_t = datetime.timedelta(seconds=((si - snc_ref.sample_stamp) * 1.0/raw_file.header.sample_freq))
            tstamp_arr.append(snc_ref.sample_time + delta_t)
        
        return (sstamp_arr, tstamp_arr)
        
    def _add_h5_metadata(self, seg_id, h5_path): # STEP 2!
        with h5py.File(h5_path) as h5_file:
            (ss_arr, ts_arr) = self._get_time_axis(seg_id)
            ecog_dataset = h5_file['ECoG Array']
            
            h5_file.create_dataset('samplestamp vector', data=ss_arr)
            h5_file.create_dataset('timestamp vector', data=np.array([ts.timestamp() for ts in ts_arr]))
            h5_file.create_dataset('channel indices', data=np.arange(ecog_dataset.shape[1]))
            
            ecog_dataset.dims.create_scale(h5_file['samplestamp vector'], 'sample axis')
            ecog_dataset.dims.create_scale(h5_file['timestamp vector'], 'time axis')
            ecog_dataset.dims.create_scale(h5_file['channel indices'], 'channel axis')
            
            ecog_dataset.dims[0].attach_scale(h5_file['samplestamp vector'])
            ecog_dataset.dims[0].attach_scale(h5_file['timestamp vector'])
            ecog_dataset.dims[1].attach_scale(h5_file['channel indices'])
        
        return
    
    def run_step1_xlfilesdk(self):
        # find step1 console app
        try:
            # see if the location is stored in an environment variable
            xlrd_converter_exe = os.getenv('XLRD_CONVERT_APP')
        except:
            pass

        if not (xlrd_converter_exe):
            # if not, then try an assumed (hard-coded) location
            xlrd_converter_exe = r'C:\XLTEK_DC\XLTEK_SDKs\RawDataFileFormat\File SDK\File SDK\File Format Example\RawDataReader\Release\xlrd_h5_converter.exe'

        print(xlrd_converter_exe)
        # check if path exists and is executable
        if os.path.exists(xlrd_converter_exe):
            with open(xlrd_converter_exe, 'rb') as candidate_exe:
                exe_magic_num = candidate_exe.read(2)
                if not ((exe_magic_num == b'MX') or (exe_magic_num == b'MZ') or (exe_magic_num == b'ZM')):
                    raise XLRDConverterExeError('Bad Executable: {:s} [magic_num={!s}]'.format(xlrd_converter_exe, exe_magic_num))
        else:
            raise XLRDConverterExeError('Executable not found: {:s}'.format(xlrd_converter_exe))
        
        # run step1
        try:
            subprocess.check_call([xlrd_converter_exe, self.input_erd], shell=True)
        except Exception as e:
            raise XLRDConvertError('xlrd_h5_converter.exe FAILED -- exited with error: {!s}'.format(e))
        
        step_one_result = '{:s}.h5'.format('.'.join(self.input_erd.split('.')[:-1]))
        if not os.path.exists(step_one_result):
            raise XLRDConvertError('xlrd_h5_converter.exe FAILED to produce output HDF5 file')
        
        # move result to output_dir
        try:
            shutil.move(step_one_result, self.output_h5)
        except Exception as e:
            raise XLRDConvertError('Unable to move xlrd_h5_converter result to output dir -- {!s}'.format(e))
        
        return
    
    def run_step2_metadatatool(self):
        try:
            seg_id = '.'.join(os.path.basename(self.input_erd).split('.')[:-1])
            self._add_h5_metadata(seg_id, self.output_h5)
        except Exception as e:
            raise XLRDStepTwoError('Unable to attach metadata to raw data HDF5 file -- {!s}'.format(e))
        
        return

def run_s2():
    conv_util = XLtoH5_ConvertUtil()
    print('Selected study [%s] for processing...\n' % (conv_util.study._study_id))

    conv_util.add_h5_metadata()
    print('\nProcessing study [%s] completed!' % (conv_util.study._study_id))

    return

# -*- coding:utf-8 -*-

"""TODO: Doc this
"""

import os
import sys
import datetime

import tkinter as tk
import tkinter.filedialog as tkfdlg
import tkinter.messagebox as tkmsgb
import tkinter.ttk as ttk
from multiprocessing import Pool, cpu_count

import numpy as np

from ..xltek.xlstudy import XLStudy
from ..utils.xl_convert_util import XLERDtoH5_ConvertUtil

try:
    from emu_sess_autopopulate import _DB_XLStudy
except ImportError:
    # assumes relative location of this file and the cldb-framework:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'cldb-framework','scripts'))
    from emu_sess_autopopulate import _DB_XLStudy


def process_xlerdfile_wrapper(iospec):
    print(iospec)
    erd_file_path, output_dir = iospec.split('|')
    try:
        print('before init')
        conv_tool = XLERDtoH5_ConvertUtil(erd_file_path, output_dir)
        print('done init, before step1')
        conv_tool.run_step1_xlfilesdk()
        print('done step1, before step2')
        conv_tool.run_step2_metadatatool()
        print('done step2')
    except Exception as e:
        status = 'FAILURE!'
        log_msg = '(error) Unable to convert file: "{:s}" -- {!s}'.format(erd_file_path, e)
        return (status, log_msg)

    status = 'SUCCESS!'
    output_file_path = os.path.join(output_dir,
                                    '{:s}.h5'.format('.'.join(os.path.basename(erd_file_path).split('.')[:-1])))
    log_msg = 'Done: {:s} -> {:s}'.format(erd_file_path, output_file_path)
    return (status, log_msg)


class EmuXLRawData_ListFilesError(Exception):
    pass


class EmuXLRawData_CorruptStudyError(Exception):
    pass


class _ProgressFrame(tk.LabelFrame):
    def __init__(self, parent, controller, nlines=1, **kwargs):
        super().__init__(parent, text='Processing status:', **kwargs)
        
        self.controller = controller
        
        self.progress = tk.DoubleVar(self, 0)
        self.prog_bar = ttk.Progressbar(self, variable=self.progress, maximum=100)
        self.prog_bar.pack(padx=25, pady=5, fill=tk.X, expand=True)
        
        self.msgs = [ '' for ii in range(nlines) ]
        self.text = tk.StringVar(self, '\n'.join(self.msgs))
        
        self.messages = tk.Message(self, textvariable=self.text, width=500,
                                   anchor=tk.W)
        self.messages.pack(padx=25, pady=5, fill=tk.X, expand=True)
    
    def update_msgs(self, new_msg):
        self.msgs.pop(0)
        self.msgs.append("[{!s}] {:s}".format(datetime.datetime.now(), new_msg))
        self.text.set('\n'.join(self.msgs))
        self.update_idletasks()
        
    def update_prog(self, new_val):
        self.progress.set(new_val)
        self.update_idletasks()


class _MainPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        topframe = tk.Frame(self)
        tk.Label(topframe, text='Subject ID: ', anchor=tk.W).pack(side=tk.LEFT)
        self.subj_id = tk.StringVar(self) 
        
        subjid_entry = tk.Entry(topframe, textvariable=self.subj_id)
        subjid_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        topframe.pack(fill=tk.X, expand=True)
        
        midframe = tk.Frame(self)
        tk.Label(midframe, text='Study: ', anchor=tk.W).pack(side=tk.LEFT)
        self.study_path = tk.StringVar(self)
        
        study_entry = tk.Entry(midframe, textvariable=self.study_path)
        study_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.study_brws = tk.Button(midframe, text='Browse', command=self.study_select)
        self.study_brws.pack(side=tk.LEFT)
        midframe.pack(fill=tk.X, expand=True)
        
        btnframe = tk.Frame(self)
        self.skip_db_step = tk.IntVar(self)
        opt_frame = tk.LabelFrame(btnframe, text='Options:')
        skip_btn = tk.Checkbutton(opt_frame, text='Skip Database Step', variable=self.skip_db_step)
        skip_btn.pack()
        opt_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.proc_btn = tk.Button(btnframe, text='Begin Processing', command=self.db_study)
        self.proc_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.exit_btn = tk.Button(btnframe, text='Cancel (Exit)', command=sys.exit)
        self.exit_btn.pack(side=tk.LEFT, fill= tk.BOTH, expand=True)
        btnframe.pack(fill=tk.X, expand=True)
        
        self.progress = _ProgressFrame(self, controller=self, nlines=3)
        self.progress.pack(fill=tk.X, expand=True, anchor=tk.S)
    
    def launch_action(self):
        return
    
    def study_select(self):
        self.study_path.set(tkfdlg.askopenfilename(filetypes=[('EEG Files','.eeg')]))
    
    def validate(self):
        self.progress.prog_bar.start()
        
        subj_id = self.subj_id.get()
        study_path = self.study_path.get()
        
        self.progress.update_msgs('''
        Parameters:
        \tSubject ID: {:s}
        \tStudy Path: {:s}
        '''.format(subj_id,
                   study_path))
        
        valid = True
        
        if (subj_id == ''):
            valid = False
            self.progress.update_msgs('(error) Bad subject ID')
        
        if not os.path.exists(study_path):
            valid = False
            self.progress.update_msgs('(error) Study path error')
        
        self.progress.prog_bar.stop()
        self.progress.update_prog(0)
        
        return valid
    
    def db_study(self):
        if (self.validate()):
            if not (self.skip_db_step.get()):
                try:
                    study_path = os.path.dirname(self.study_path.get())
                    
                    self.progress.prog_bar.start()
                    util = _DB_XLStudy(study_path,
                                       self.subj_id.get())
                    self.progress.update_msgs('(info) study logged in database')
                    
                    util.log_all()
                    self.progress.update_msgs('(info) study segments logged in database')
                    
                    self.progress.prog_bar.stop()
                    self.progress.update_prog(100)
                except Exception as e:
                    errmsg = 'Unable to database study -- {!s}'.format(e)
                    self.progress.update_msgs('(error) {:s}'.format(errmsg))
                    
                    self.progress.prog_bar.stop()
                    self.progress.update_prog(100)
                    
                    tkmsgb.showwarning(title='Database Error', message='{:s}. Please attempt to resolve these issues before trying again.')

            # if (form is valid, form is processed, and no errors) OR (skip_db_step is set)
            # then move to the next frame
            self.controller.study_path = self.study_path.get()
            self.controller.show_page('_ConvertPage')
            
        return


class _SelectFilesFrame(tk.LabelFrame):
    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, text='Select files to convert:', **kwargs)
        self.controller = controller
        
        self.segment_list = ttk.Treeview(self)
        
        self.segment_list['columns'] = ('start_datetime', 'end_datetime')
        self.segment_list.heading(0, text='file start time')
        self.segment_list.heading(1, text='file end time')
        
        self.segment_list.pack(fill=tk.BOTH, expand=True)
    
    def populate_files(self, study, pframe):
        pframe.prog_bar.start()
        try:
            nsegs = len(study.raw_data_segments)
            for si,raw_data_seg in enumerate(study.stc_file):
                seg_id = raw_data_seg.segment_name
                start_stamp = raw_data_seg.start_stamp
                end_stamp = raw_data_seg.end_stamp
                
                fs = study.raw_data_segments[seg_id][1].header.sample_freq
                snc_data = study.snc_file.snc_data
                sync_ref = snc_data[np.argmin(
                    np.array([np.abs(se.sample_stamp - start_stamp) for se in snc_data])
                )]
                delta_t = datetime.timedelta(seconds=((start_stamp - sync_ref.sample_stamp) * 1.0/fs))
                
                start_dt = sync_ref.sample_time + delta_t
                end_dt = start_dt + datetime.timedelta(seconds=((end_stamp - start_stamp) * 1.0/fs))
                
                self.segment_list.insert('', si,
                                         iid=study.raw_data_segments[seg_id][1].erd_path,
                                         text='_'.join(seg_id.split('_')[1:]),
                                         values=(start_dt, end_dt))
                
                pframe.prog_bar.stop()
                pframe.update_prog((100.0 * (si+1))/nsegs)
            
            pframe.prog_bar.stop()
            pframe.update_prog(100)
        
            # show the newly created table (in the current page)
            self.controller.show_page('_SelectFilesFrame')
            
        except Exception as e:
            pframe.prog_bar.stop()
            pframe.update_prog(100)
            
            errmsg = 'Unable to populate raw data filelist -- {!s}'.format(e)
            pframe.update_msgs('(error) {:s}'.format(errmsg))
            tkmsgb.showerror(title='EmuXLRawData_ListFilesError', message=errmsg)
            raise EmuXLRawData_ListFilesError(errmsg)
    
    def get_selection(self):
        return self.segment_list.selection()


class _ConvertPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        container = tk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)
        
        self.pages = {}
        self.pages['_ProgressFrame'] = _ProgressFrame(parent=container, controller=self,
                                                      nlines=3)
        self.pages['_ProgressFrame'].grid(row=0, column=0, sticky=tk.NSEW, ipadx=10, ipady=10)
        self.pages['_SelectFilesFrame'] = _SelectFilesFrame(parent=container, controller=self)
        self.pages['_SelectFilesFrame'].grid(row=0, column=0, sticky=tk.NSEW, ipadx=10, ipady=10)
            
        self.show_page('_ProgressFrame')
        
        # Processing options
        opt_frame = tk.LabelFrame(self, text='Options:')
        opt_frame.pack(fill=tk.BOTH, expand=True,
                       ipadx=10, ipady=15)
        output_opt_frame = tk.Frame(opt_frame)
        output_opt_frame.pack(padx=5, fill=tk.X, expand=True)
        
        tk.Label(output_opt_frame, text='Output directory: ', anchor=tk.W).pack(side=tk.LEFT)
        self.output_path = tk.StringVar(self)
        tk.Entry(output_opt_frame,
                 textvariable=self.output_path).pack(side=tk.LEFT,
                                                     fill=tk.BOTH,
                                                     padx=5, pady=5,
                                                     expand=True)
        tk.Button(output_opt_frame,
                  text='Browse',
                  command=self.output_select).pack(side=tk.LEFT)
        
        midframe = tk.Frame(opt_frame)
        midframe.pack(fill=tk.BOTH, expand=True, ipadx=5, ipady=5)
        dupl_mode_box = tk.LabelFrame(midframe, text='Duplicates:')
        dupl_mode_box.pack(pady=5, ipady=5, fill=tk.BOTH, side=tk.LEFT, expand=True)
        self.duplicates_mode = tk.StringVar(self)
        
        tk.Radiobutton(dupl_mode_box, text='Skip', anchor=tk.W,
                       variable=self.duplicates_mode, value='skip').pack(anchor=tk.W, pady=5)
        tk.Radiobutton(dupl_mode_box, text='Overwrite', anchor=tk.W,
                       variable=self.duplicates_mode, value='ovwt').pack(anchor=tk.W, pady=5)
        self.duplicates_mode.set('skip')
        
        par_box = tk.LabelFrame(midframe, text='Parallel:')
        par_box.pack(pady=5, fill=tk.BOTH, side=tk.RIGHT, expand=True)
        self.n_procs = tk.IntVar(self)
        
        tk.Scale(par_box, label='# of Procs', orient=tk.HORIZONTAL, variable=self.n_procs,
                 from_=1, to=cpu_count()).pack(padx=10, pady=10, expand=True)
        
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Button(btn_frame, text='Begin Processing',
                  command=self.launch_processing).pack(side=tk.LEFT)
        tk.Button(btn_frame, text='Cancel (Exit)',
                  command=sys.exit).pack(side=tk.RIGHT)
        
        return
        
    def output_select(self):
        self.output_path.set(tkfdlg.askdirectory(initialdir=self.controller.study_path))
    
    def launch_action(self):
        self.pages['_SelectFilesFrame'].populate_files(
            XLStudy(os.path.dirname(self.controller.study_path)),
            self.pages['_ProgressFrame'])
    
    def validate(self):
        output_dir = self.output_path.get(); #import pdb; pdb.set_trace()
        filelist = []; log_msg = ''
        if not os.path.isdir(output_dir):
            return filelist, False, 'Output Directory Does Not Exist'
        
        pre_filelist = self.pages['_SelectFilesFrame'].segment_list.selection()
        if (pre_filelist == ''):
            return filelist, False, 'No Files Selected for Conversion'
        
        for f_in in pre_filelist:
            f_out = os.path.join(output_dir,
                                 '.'.join(os.path.basename(f_in).split('.')[:-1] + ['h5']))
            
            if not os.path.exists(f_in):
                raise EmuXLRawData_CorruptStudyError('Specified file does not exist: {:s}'.format(f_in))
            
            if not os.path.exists(f_out):
                filelist.append("{:s}|{:s}".format(f_in,output_dir))
            elif self.duplicates_mode.get() == 'overwrite':
                try:
                    os.remove(f_out)
                    filelist.append("{:s}|{:s}".format(f_in,output_dir))
                except:
                    raise Exception('Unable to clear files for overwrite mode')
        
        return filelist, True, 'Form data is valid'
    
    def launch_processing(self):
        try:
            filelist, valid, log_msg = self.validate()
        except Exception as e:
            tkmsgb.showerror(title='Error during validation', message='Encountered error: {!s}'.format(e))
            #import pdb; pdb.set_trace()
            return
        
        if not valid:
            tkmsgb.showwarning(title='Unable to process', message=log_msg)
            return
        
        self.pages['_ProgressFrame'].update_prog(5.0)
        self.show_page('_ProgressFrame')

        import time; time.sleep(1)
        try:
            proc_pool = Pool(processes=self.n_procs.get())
            for pr, rv in enumerate(proc_pool.imap_unordered(process_xlerdfile_wrapper, filelist)):
                self.pages['_ProgressFrame'].update_msgs('[{!s}] {!s}'.format(rv[0], rv[1]))
                self.pages['_ProgressFrame'].update_prog(5.0 + ((95.0 * (pr + 1))/len(filelist)))

            proc_pool.close()
            proc_pool.join()
        except Exception as e:
            print(e)
            self.pages['_ProgressFrame'].update_msgs('||ERROR|| FAILURE: {!s} <<*** PROCESSING FAILED!!! ***>>'.format(e))
            self.pages['_ProgressFrame'].update_prog(100)
            return

        self.pages['_ProgressFrame'].update_msgs('*** PROCESSING COMPLETE!!! ***')
        self.pages['_ProgressFrame'].update_prog(100)
        return
    
    def show_page(self, page_id):
        page = self.pages[page_id]
        page.tkraise()
        self.update_idletasks()


class EmuDataGuiApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        container = tk.Frame(self, relief=tk.GROOVE)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                       padx=25, pady=25)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.pages = {}
        for F in (_MainPage, _ConvertPage):
            page_id = F.__name__
            page = F(parent=container, controller=self)
            self.pages[page_id] = page
            
            page.grid(row=0, column=0, sticky=tk.NSEW)
        
        self.show_page('_MainPage')
    
    def show_page(self, page_id):
        page = self.pages[page_id]
        page.tkraise()
        page.launch_action()
        self.update_idletasks()
    

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    
    app = EmuDataGuiApp()
    app.mainloop()

if __name__ == "__main__":
    main()

################################################################################
#                                                                              #
#  Module:      edffile                                                        #
#  Purpose:     reader/writer and file-format specification objects for the    #
#                European Data Format for physiological time-series data       #
#                                                                              #
#  Author:      Dharshan Chandramohan                                          #
#                                                                              #
#  Created:     13 Jan 2015                                                    #
#  Copyright:   (c) Dharshan Chandramohan 2015                                 #
#                                                                              #
#  References:  http://www.edfplus.info/specs/edf.html                         #
#               http://www.edfplus.info/specs/edfplus.html                     #
#                                                                              #
################################################################################

import struct

import numpy as np

class EDFHeader(object):
    """ EDF+ file header """
    
    _BASEHDRSIZE = 256;
    
    # Header: (fieldname, (offset, size))
    _hdrspec = [ ('EDFVERSION', (0,8)),
                 ('LOCALPATID', (8,80)),
                 ('LOCALRECID', (88,80)),
                 ('RSTARTDATE', (168,8)),
                 ('RSTARTTIME', (176,8)),
                 ('HDRRECBYTS', (184,8)),
                 ('RSRVDSPACE', (192,44)),
                 ('NDATARECDS', (236,8)),
                 ('DATARECDUR', (244,8)),
                 ('NDATASGNLS', (252,4)) ] # ns
    # Channel Data: (fieldname, size)
    _chanspec = [ ('CHANNELLBL',16),
                  ('TRANSDCTYP',80),
                  ('PHYSICLDIM',8),
                  ('PHYSICLMIN',8),
                  ('PHYSICLMAX',8),
                  ('DIGITALMIN',8),
                  ('DIGITALMAX',8),
                  ('PREFLTRING',80),
                  ('NUMSAMPLES',8), # nr
                  ('CHNRSVDSPC',32) ]
    
    @classmethod
    def hdrfields(cls):
        return [ fldspec[0] for fldspec in cls._hdrspec ]

    @classmethod
    def fldoffset(cls,field):
        return dict(cls._hdrspec).get(field,(None,None))[0]

    @classmethod
    def fldlength(cls,field):
        return dict(cls._hdrspec).get(field,(None,None))[1]
    
    def __init__(self,datafile,**kwargs):
        if 'r' in datafile.mode:
            self.header_record = {}
            self.read_from_file(datafile)
        
        if 'w' in datafile.mode:
            # TODO do some stuff with keyword args...
            # Format stuff as strings so it will be easy to write...
            raise NotImplementedError
        
        return
           
    def read_from_file(self,datafile):
        """ Read EDF+ Header Record from file """
        # Read the header record
        for spc_field,(spc_offset,spc_size) in EDFHeader._hdrspec:
            datafile.seek(spc_offset)
            self.header_record[spc_field] = datafile.read(spc_size)
        
        # Read header fields for each "signal" (eg. channel)
        chspc_offset = EDFHeader._BASEHDRSIZE
        self.header_record['SIGNALSPEC'] = [{} for sig_idx in range(int(self.header_record['NDATASGNLS']))]
        for chspc_field,chspc_size in EDFHeader._chanspec:
            for sig_idx in range(int(self.header_record['NDATASGNLS'])):
                datafile.seek(chspc_offset)
                self.header_record['SIGNALSPEC'][sig_idx][chspc_field] = datafile.read(chspc_size)
                chspc_offset += chspc_size
                
        return
    
    def write_to_file(self,datafile):
        # TODO handle writing the header
        # do some error/size checks then write the header_record dictionary
        raise NotImplementedError
    
    
class EDFFile(object):
    """File-like object representing an EDF file"""
    
    def __init__(self, filename, mode='r'):
        self._edfname = filename
        self._openmode = mode
        
        if not ( (self._openmode == 'r') or (self._openmode == 'w') or (self._openmode == 'r+') ):
            raise ValueError("Mode string must be either 'r' or 'w' or 'r+'.")
        
        fmode = self._openmode+'b' # EDFs are binary files
        self._edffile = open(self._edfname,fmode)
        
        self._edfheader = EDFHeader(self._edffile) # read header
        self.header_record = self._edfheader.header_record
        
        # TODO: Next parse out some important data from the header
        # NOTE: header is written in ascii characters
        #       converting this data will make it more useful
        
        # eg. parse date, num channels/signals, num samples, etc.
        # ...

        return

    # Lazy read methods (save memory):
    def read_data(self):
        # Extract channel data:
        # TODO: Handle annotations
        self.channel_data = [{'INFO':sigspec,'DATA':[]} for sigspec in self.header_record['SIGNALSPEC']]
        
        for clip in range(int(self.header_record['NDATARECDS'])):
            for cdata in self.channel_data:
                nr = int(cdata['INFO']['NUMSAMPLES'])
                fmt = '<' + 'h'*nr
                cdata['DATA'].append(struct.unpack(fmt,self._edffile.read(struct.calcsize(fmt))))
        
        def rescale(x,a,b,c,d):
            return c + (((x-a)/(b-a))*(d-c))
        
        for cc in self.channel_data:
            a = np.double(cc['INFO']['DIGITALMIN'])
            b = np.double(cc['INFO']['DIGITALMAX'])
            c = np.double(cc['INFO']['PHYSICLMIN'])
            d = np.double(cc['INFO']['PHYSICLMAX'])
            cc['DATA'] = np.array([rescale(np.double(x),a,b,c,d) for x in np.array(cc['DATA']).flatten()])

        return

    def anonymize(self):
        if self._openmode == 'r+':
            self._edffile.seek(EDFHeader.fldoffset('LOCALPATID'))
            self._edffile.write(b'X X X X'.ljust(EDFHeader.fldlength('LOCALPATID')))
        else:
            raise IOError('file opened with incorrect mode')
        
        return
        
    @classmethod
    def open(cls, filename, mode='r'):
        return cls(filename, mode)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    def close(self):
        self._edffile.close()
    




import numpy as np
import math
import random
import json
import librosa
import scipy.fft, scipy.signal
import os

import sofa

class BaseRoom:
    def __init__(self, **kwargs):
        # kwargs specify properties of a room type, e.g., size range, list of sofas, T60 range
        # A single room will be initialized by choosing specific values from these properties
        self.room_info = {} # Contains all information of the selected room
                            # A room object can be initialized using only this
        raise NotImplementedError("Subclasses must implement this!")

    def get_nchan(self) -> int: # return number of mic in the mic array
        raise NotImplementedError("Subclasses must implement this!")
        
    def get_position(self,
                     event_duration=None, # passed by spatial_sound_scene, optional
                     mode='point', **kwargs) -> list:
        position = [] # [npos, 3], any type of position usable by the synthesize function, and serializable to a JSON file
        # return position
        raise NotImplementedError("Subclasses must implement this!")
        
    def synthesize(self,
                   source_signal, # [wlen]
                   sr,
                   source_position, # [npos, 3], from get_position function
                   return_option = { # other options are possible
                       'rir': True,
                       'dry': True,
                   }):
        soundscape = {}
        soundscape['waveform'] = [] # [nchan, wlen]
        
        if return_option.get('dry', False):
            soundscape['waveform_dry'] = [] # [nchan, wlen]
        
        if return_option.get('rir', False):
            soundscape['rir'] = [] # [npos, nchan, rir_len]
            if return_option.get('dry', False):
                soundscape['rir_dry'] = [] # [npos, nchan, rir_len]
        
        # return soundscape
        raise NotImplementedError("Subclasses must implement this!")

    # Subclasses only need to override generate_metadata and from_metadata if they contain non-serializable types (e.g., numpy arrays).
    # In such cases, the data must be converted to a serializable type in generate_metadata and converted back to the original type in from_metadata.
    def generate_metadata(self, metadatapath=None):
        if metadatapath is not None:
            with open(metadatapath, "w") as outfile:
                json.dump(self.room_info, outfile, indent=4)
        return self.room_info
    
    @classmethod
    def from_metadata(cls, metadata):
        if isinstance(metadata, str):
            with open(metadata) as f: room_info = json.load(f);
        elif isinstance(metadata, dict): room_info = metadata
        else: raise ValueError(f"metadata should be room_info dict or path to json file. Recived: {metadata}")
        instance = cls.__new__(cls)
        instance.room_info = room_info
        return instance

    # util
    def _get_direct_path_rir(self,
                             RIRs, # np.array, [..., RIR len]
                             direct_time_sp, # time in sample
                             direct_range_sp): # [6, 50]
        
        RIRs_direct = np.zeros_like(RIRs)
        abs_direct_range = np.stack([np.maximum(direct_time_sp - direct_range_sp[0], 0.0),
                                     np.minimum(direct_time_sp + direct_range_sp[1], RIRs.shape[-1] - 1)], axis=2).astype(int)
        for i in range(RIRs.shape[0]):
                for j in range(RIRs.shape[1]):
                    RIRs_direct[i, j, abs_direct_range[i, j, 0]: abs_direct_range[i, j, 1]+1] = RIRs[i, j, abs_direct_range[i, j, 0]: abs_direct_range[i, j, 1]+1]
        return RIRs_direct

class SofaRoom(BaseRoom):
    def __init__(self,
                 path, # path to a sofa file, or path to directory containing sofa files
                 direct_range_ms = [6, 50], # [6, 50] ms
                ):
        # TODO: relative path vs absolute path
        assert direct_range_ms[0] > 0
        if os.path.isdir(path):
            all_sofa_files = [f for f in os.listdir(path) if f.endswith('.sofa')] # sort all_sofa_files for reproducible
            assert all_sofa_files, f'No sofa file found in {path}'
            sofa_path = os.path.join(path, random.choice(all_sofa_files))
        else:
            sofa_path = path # path to a sofa file
        
        sofafile = sofa.Database.open(sofa_path, mode='r', parallel=False)
        dims = sofafile.Data.IR.dimensions() # ('M', 'R', 'N')
        shape = sofafile.Data.IR.shape # ( ... , 4, 48000)
        dim_sizes = dict(zip(dims, shape))
        sofa_sr = int(sofafile.Data.SamplingRate.get_values()[0])
        
        self.room_info = {
            'sofa_path': sofa_path,
            'sr': sofa_sr,
            'nchan': dim_sizes['R'], # receiver
            'nrir': dim_sizes['M'], # from 0 to nrir-1
            'rir_len': dim_sizes['N'],
            'direct_range_ms': direct_range_ms,
        }
        sofafile.close()

        
    def get_nchan(self): # return number of mic
        return self.room_info['nchan']

    def get_all_positions(self):
        sofafile = sofa.Database.open(self.room_info['sofa_path'], mode='r', parallel=False)
        position = sofafile.Source.Position.get_values(system="cartesian")
        sofafile.close()
        return position # [[x,y,z]]

        
    def get_position(self,
                     mode='point', # TODO, choose the RIR most close to the position
                     **kwargs): # event_duration = None,# unused
        if mode != 'point':
            raise NotImplementedError("Sofa room only support mode = 'point'")

        sofafile = sofa.Database.open(self.room_info['sofa_path'], mode='r', parallel=False)

        selected_index = random.randint(0, self.room_info['nrir'] - 1)
        position = sofafile.Source.Position.get_values(indices={'M': selected_index}, system="cartesian")
        position = [position.tolist()]

        sofafile.close()
        
        return position # [[x,y,z]]
        
    def synthesize(self,
                   source_signal, # [wlen, ]
                   sr,
                   source_position, # [npos, 3]
                   return_option):
        return_dry = return_option.get('dry', False)
        return_dry_channel = return_option.get('dry_channel', None)
        return_rir = return_option.get('rir', False)
        soundscape = {}

        # only one source position
        source_position = np.array(source_position)
        assert source_position.shape[0] == 1, 'sofa room does not support moving sources'
        source_position = source_position[0]

        sofafile = sofa.Database.open(self.room_info['sofa_path'], mode='r', parallel=False)

        # Get position index and then get the corresponding RIR
        all_positions = sofafile.Source.Position.get_values(system="cartesian")
        index = np.where(np.all(np.isclose(all_positions, source_position, atol=1e-6), axis=1))[0]
        RIRs = sofafile.Data.IR.get_values(indices = {'M': index}, dim_order=None) # [1, nchan, rir len]

        # resample (if needed) and normalize RIR
        if self.room_info['sr'] != sr:
            RIRs = librosa.resample(RIRs, orig_sr = self.room_info['sr'], target_sr = sr, axis = -1);
        RIRs = RIRs / np.sqrt(np.mean(np.sum(RIRs**2, axis=-1)))

        # synthesize
        soundscape['waveform'] = scipy.signal.fftconvolve(source_signal[None, :], RIRs[0], mode="full", axes=1) # [nchan, wlen + rir len]
        

        if return_dry:
            if return_dry_channel is None:
                direct_time = np.argmax(np.abs(RIRs), axis = -1) # [1, nchan]
                RIRs_direct = self._get_direct_path_rir(
                                    RIRs=RIRs, # np.array, [1, nchan, rir len]
                                    direct_time_sp = direct_time, # [...]
                                    direct_range_sp = [int(self.room_info['direct_range_ms'][0]*sr/1000),
                                                       int(self.room_info['direct_range_ms'][1]*sr/1000)]
                )
            else:
                direct_time = np.argmax(np.abs(RIRs[:, return_dry_channel:return_dry_channel+1, :]), axis = -1) # [1, nchan]
                RIRs_direct = self._get_direct_path_rir(
                                    RIRs=RIRs[:, return_dry_channel:return_dry_channel+1, :], # np.array, [..., RIR len]
                                    direct_time_sp = direct_time, # [...]
                                    direct_range_sp = [int(self.room_info['direct_range_ms'][0]*sr/1000),
                                                       int(self.room_info['direct_range_ms'][1]*sr/1000)]
                )
            soundscape['waveform_dry'] = scipy.signal.fftconvolve(source_signal[None, :], RIRs_direct[0], mode="full", axes=1) # [nchan, wlen + rir len]
            
        if return_rir:
            soundscape['rir'] = RIRs # [nposition, nchan, rir_len]
            if return_dry: soundscape['rir_dry'] = RIRs_direct

        sofafile.close()
        return soundscape


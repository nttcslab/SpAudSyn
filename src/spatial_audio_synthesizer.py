import os
import random
import warnings
import librosa
import numpy as np
import json
import wave
import math

from .utils import get_files_list, get_labels, find_event_time, trim_signal, initialize_config, source_file_filter

class SpAudSyn:
    def __init__(
        self,
        duration, # in second
        sr, # 32000
        max_event_overlap,
        max_event_dur,
        ref_db,
        foreground_dir,
        background_dir=None,
        interference_dir=None,
        room_config=None,
        verbose=False,
    ):
        self.config = locals().copy(); self.config.pop('self') # push all parameters into a dict
        
        self.fg_events = []
        self.bg_events = []
        self.int_events = []
        self.fg_labels = get_labels(foreground_dir) # get all foreground events' label
        if interference_dir:
            self.int_labels = get_labels(interference_dir) # get all interference events' label
        if room_config is not None:
            self.set_room(room_config)
        if verbose:
            print('Foreground labels:', ', '.join(self.fg_labels))
            # if interference_dir: print('Interference labels:', ', '.join(self.int_labels))
    
    #======================================================
    # METADATA
    #======================================================
    def generate_metadata(self, metadata_path=None):
        self.fg_events = sorted(self.fg_events, key=lambda x: x['event_time'])
        self.int_events = sorted(self.int_events, key=lambda x: x['event_time'])
        metadata = {'config': self.config}
        metadata['fg_events'] = [event for event in self.fg_events]
        metadata['bg_events'] = [event for event in self.bg_events]
        metadata['int_events'] = [event for event in self.int_events]
        metadata['room'] = self.room_metadata
        if metadata_path is not None:
            with open(metadata_path, "w") as outfile:
                json.dump(metadata, outfile, indent=4)
        return metadata
    
    @classmethod
    def from_metadata(cls, metadata): # metadata should be dict or path to json file
        if isinstance(metadata, str):
            with open(metadata) as f: metadata = json.load(f);
        instance = cls.__new__(cls)
        instance.config = metadata['config']
        instance.fg_events = metadata['fg_events']
        instance.bg_events = metadata['bg_events']
        instance.int_events = metadata['int_events']
        instance.fg_labels = get_labels(instance.config['foreground_dir']) # get all foreground events' label
        if instance.config['interference_dir']:
            instance.int_labels = get_labels(instance.config['interference_dir']) # get all interference events' label
        instance.room = initialize_config(metadata['room'])
        instance.room_metadata = metadata['room']
        return instance

    #======================================================
    # ADD SOUND SCENE COMPONENTS
    #======================================================
    def set_room(self, room_config):
        self.room = initialize_config(room_config)
        room_info = self.room.generate_metadata()
        self.config['room_config'] = room_config # update room config
        self.room_metadata = {
            'module': room_config['module'],
            'main': room_config['main']  + '.from_metadata',
            'args': {'metadata': room_info},
        }
    
    def add_event(
        self,
        label=None,
        source_file=None,
        source_time=None,
        event_time=None,
        event_position=None,
        snr=None,

        max_try = 1,
        trim_amplitude = None,
        min_event_duration = None,
    ):
        """
        Adds a sound event to the foreground event list (`self.fg_events`).
    
        Parameters
        ----------
        label : dict, optional
            Method to select the label for the event.
            Default: `{'method': 'choose'}` (randomly select from all foreground labels)
            Options:
            - `{'method': 'choose'}`: randomly choose from all labels
            - `{'method': 'choose', 'value': [label1, label2, ...]}`: randomly choose from the provided list
            - `{'method': 'choose_wo_replacement'}`: choose a label not already added to the mixture
            - `{'method': 'const', 'value': label}`: use a fixed label
    
        source_file : dict, optional
            Method to select the source audio file.
            Default: `{'method': 'choose'}` (randomly select a file from label folder)
            Options:
            - `{'method': 'choose'}`: choose randomly from all files of the label
            - `{'method': 'choose', 'value': [path1, path2, ...]}`: randomly choose from a provided list
            - `{'method': 'choose_wo_replacement', 'exclusion_folder_depth': 0}`: choose a file not already used
            - `{'method': 'const', 'value': path}`: use a fixed file path
    
        source_time : dict, optional
            Start time within the source file.
            Default: `{'method': 'choose'}` (start randomly if source is longer than max_event_dur)
            Options:
            - `{'method': 'choose'}`: random start
            - `{'method': 'const', 'value': t}`: fixed start time in seconds
    
        event_time : dict, optional
            Start time of the event in the mixture.
            Default: `{'method': 'choose'}` (find valid time respecting max_event_overlap)
            Options:
            - `{'method': 'choose'}`: automatically select valid start time
            - `{'method': 'const', 'value': t}`: fixed start time in seconds
    
        event_position : dict, optional
            Spatial position of the event in the room.
            Default: `{'method': 'choose', 'get_position_args': {}}` (use room's position function)
            Options:
            - `{'method': 'choose', 'get_position_args': kwargs}`: call `self.room.get_position(**kwargs)`
            - `{'method': 'const', 'value': position}`: fixed position array
    
        snr : dict, optional
            Signal-to-noise ratio for the event relative to reference dB.
            Default: `{'method': 'uniform', 'range': [5, 20]}`
            Options:
            - `{'method': 'uniform', 'range': [min, max]}`: random uniform SNR in dB
            - `{'method': 'choose', 'value': [val1, val2, ...]}`: randomly pick from list
            - `{'method': 'const', 'value': val}`: fixed SNR in dB
    
        max_try : int, optional
            Maximum attempts to satisfy constraints such as trimming, min duration, and valid mixture time.
            Default: 1
    
        trim_amplitude : float or None, optional
            Threshold for trimming low-amplitude segments of the source. 
            Default: None (no trimming)
    
        min_event_duration : float or None, optional
            Minimum allowed duration after trimming. If segment is shorter, it retries or skips.
            Default: None
    
        Returns
        -------
        None
    
        Notes
        -----
        - `self.room` must be set before adding events (via `room_config` or `set_room`).
        - Setting `trim_amplitude` or `min_event_duration` loads the audio, which can slow processing.
        """
        label = {'method': 'choose'} if label is None else label
        source_file = {'method': 'choose'} if source_file is None else source_file
        source_time = {'method': 'choose'} if source_time is None else source_time
        event_time = {'method': 'choose'} if event_time is None else event_time
        event_position = {'method': 'choose', 'get_position_args': {}} if event_position is None else event_position
        snr = {'method': 'uniform', 'range': [5, 20]} if snr is None else snr
        
        self._add_event(
            target_event_list=self.fg_events,
            event_dir=self.config['foreground_dir'],
            all_event_labels=self.fg_labels,
            role='foreground',
            label=label,
            source_file=source_file,
            source_time=source_time,
            event_time=event_time,
            event_position=event_position,
            snr=snr,
            max_try = max_try,
            trim_amplitude = trim_amplitude,
            min_event_duration = min_event_duration,
        )

    def add_interference(
        self,
        label=None,
        source_file=None,
        source_time=None,
        event_time=None,
        event_position=None,
        snr=None,

        max_try = 1,
        trim_amplitude = None,
        min_event_duration = None,
    ):
        """
        Adds a sound event to a interference event list (self.int_events).
        Parameters are same as add_event
        """
        label = {'method': 'choose'} if label is None else label
        source_file = {'method': 'choose'} if source_file is None else source_file
        source_time = {'method': 'choose'} if source_time is None else source_time
        event_time = {'method': 'choose'} if event_time is None else event_time
        event_position = {'method': 'choose', 'get_position_args': {}} if event_position is None else event_position
        snr = {'method': 'uniform', 'range': [5, 20]} if snr is None else snr

        if not self.config.get('interference_dir'):
            warnings.warn(f'No interference_dir specified, No interference added')
            return

        self._add_event(
            target_event_list=self.int_events,
            event_dir=self.config['interference_dir'],
            all_event_labels=self.int_labels,
            role='interference',
            label=label,
            source_file=source_file,
            source_time=source_time,
            event_time=event_time,
            event_position=event_position,
            snr=snr,
            max_try = max_try,
            trim_amplitude = trim_amplitude,
            min_event_duration = min_event_duration,
        )
    
    def _add_event(
        self,
        target_event_list,
        event_dir,
        all_event_labels,
        role,

        label,
        source_file,
        source_time,
        event_time,
        event_position,
        snr,

        max_try = 1,

        trim_amplitude = None,
        min_event_duration = None,
    ):
        if not hasattr(self, 'room'):
            warnings.warn(f'NO EVENT ADDED: before adding event, room must be set using either room_config option in constructor or set_room function!')
            return
        
        # Select label
        # No retry, assuring the uniform distribution of classes
        if label['method'] == "choose_wo_replacement": # {'method': 'choose_wo_replacement'}
            added_labels = set(e['label'] for e in target_event_list)
            remaining_labels = set(all_event_labels) - added_labels
            if not remaining_labels:
                warnings.warn(f'NO EVENT ADDED: No remaining labels are available to choose from.')
                return
            label_ = random.choice(list(remaining_labels))
        elif label['method'] == "choose":
            if isinstance(label.get('value'), list): # {'method': 'choose', 'value': [lb1, lb2,...]}
                assert set(label['value']) <= set(all_event_labels), f'Invalid label list: {label["value"]}.'
                label_ = random.choice(label['value'])
            else: # {'method': 'choose'}
                label_ = random.choice(all_event_labels)
        elif label['method'] == "const" and 'value' in label: # {'method': 'const', 'value': label}
            assert label['value'] in all_event_labels, f'Invalid label: {label["value"]}.'
            label_ = label['value']
        else:
            warnings.warn(f'NO EVENT ADDED: Invalid label option: {label}.')
            return

        itry = 0
        for itry in range(max_try):
        # Retry if a suitable source segment cannot be found or if no valid position in the mixture is available.
            # Select source file
            if source_file['method'] == "choose":
                if isinstance(source_file.get('value'), list): # {'method': 'choose', 'value': [path1, path2,...]}
                    source_file_ = random.choice(source_file['value'])
                else: # {'method': 'choose'}
                    source_file_ = random.choice(get_files_list(os.path.join(event_dir, label_), extension='.wav', recursive=True))
            elif source_file['method'] == "choose_wo_replacement": # {'method': 'choose_wo_replacement', 'exclusion_folder_depth': 0}
                exclusion_folder_depth = source_file.get('exclusion_folder_depth', 0)
                added_source_files = set(e['source_file'] for e in target_event_list if e['label'] == label_)
                remain_wav_files = source_file_filter(event_dir, label_, added_source_files, exclusion_folder_depth=exclusion_folder_depth)
                if not remain_wav_files:
                    warnings.warn(f'NO EVENT ADDED: No remaining source files to choose from.')
                    return
                source_file_ = random.choice(remain_wav_files)
            elif source_file['method'] == "const" and 'value' in source_file: # {'method': 'const', 'value': path}
                source_file_ = source_file['value']
            else:
                warnings.warn(f'NO EVENT ADDED: Invalid source_file option: {source_file}.')
                return

            event_duration_ = librosa.get_duration(path=source_file_)

            # Select start time and duration of source: actual part of source used
            if source_time['method'] == "choose": # {'method': 'choose'}
                if event_duration_ > self.config['max_event_dur']:
                    source_time_ = random.uniform(0, event_duration_ - self.config['max_event_dur'])
                else:
                    source_time_ = 0
            elif source_time['method'] == "const" and 'value' in source_time: # {'method': 'const', 'value': time}
                source_time_ = source_time['value']
            else:
                warnings.warn(f'NO EVENT ADDED: Invalid source_time option: {source_time}.')
                return

            if event_duration_ > self.config['max_event_dur']: # adjust duration
                event_duration_ = self.config['max_event_dur']

            if trim_amplitude is not None or min_event_duration is not None:
                if trim_amplitude is None: trim_amplitude = 0.0
                if min_event_duration is None or min_event_duration < 1e-6: min_event_duration = 1e-6 # avoid all-zeros

                x, sr = librosa.load(
                    source_file_,
                    sr=None,
                    offset=source_time_,
                    duration=event_duration_,
                    mono=True,
                ) # (wlen, )
                if np.max(np.abs(x)) == 0: # all-zero segment, retry or terminate
                    if itry < max_try - 1:
                        if self.config['verbose']: print(f'All-zero segment is selected \n"{source_file_}" from {source_time_}, duration {event_duration_}. Try again {itry + 1}.')
                        continue
                    else:
                        warnings.warn(f'NO EVENT ADDED: All-zero signal.\n"{source_file_}" from {source_time_}, duration {event_duration_}.')
                        return

                # Trim signal
                trim_start_sample, trim_end_sample = trim_signal(x, trim_amplitude = trim_amplitude)
                trim_start_sec = trim_start_sample/sr
                after_trim_duration_sec = (trim_end_sample - trim_start_sample + 1) / sr

                # Check signal length after triming
                if after_trim_duration_sec < min_event_duration: # too-short segment, retry or terminate
                    if itry < max_try - 1:
                        if self.config['verbose']: print(f'Too-short segment is selected \n"{source_file_}" from {source_time_}, duration {event_duration_}. Try again {itry + 1}.')
                        continue
                    else:
                        warnings.warn(f'NO EVENT ADDED: Too-short segment is selected \n"{source_file_}" from {source_time_}, duration {event_duration_}.')
                        return

                source_time_ += trim_start_sec
                event_duration_ = after_trim_duration_sec

            # Find the time position in the mixture for event so that the event overlap does not exceed max_event_overlap
            if event_time['method'] == 'choose': # {'method': 'choose'} # TODO: other options
                # event_time = ("uniform", 0, self.config['duration'])
                event_time_ = find_event_time(
                    mixture_duration = self.config['duration'],
                    event_duration = event_duration_,
                    max_overlap = self.config['max_event_overlap'],
                    existing_events = target_event_list,
                )
                if event_time_ is None:
                    if itry < max_try - 1:
                        if self.config['verbose']: print(f'No valid start time found for sound event "{source_file_}". Try again {itry}.')
                        continue
                    else:
                        warnings.warn(f'NO EVENT ADDED: No valid start time found for sound event "{source_file_}". Try using fewer events or increasing max_event_overlap.')
                        return
            elif event_time['method'] == 'const' and 'value' in event_time: # {'method': 'const', 'value': time}
                event_time_ = event_time['value']
            else:
                warnings.warn(f'NO EVENT ADDED: Invalid event_time option: {event_time}.')
                return

            break # successfully find the source

        # Set event spatial positions, selected using a function from the room
        if event_position['method'] == "choose" and 'get_position_args' in event_position: # {'method': 'choose', 'get_position_args': {}}
            event_position_ = self.room.get_position(event_duration = event_duration_, # for future use, e.g., moving sources
                                                     **event_position['get_position_args'])
        elif event_position['method'] == "const" and 'value' in event_position: # {'method': 'const', 'value': position}
            event_position_ = event_position['value']
        else:
            warnings.warn(f'NO EVENT ADDED: Invalid event_position option: {event_position}.')
            return

        if snr['method'] == "uniform" and isinstance(snr.get('range'), list): # {'method': 'uniform', 'range': [5, 10]}
            snr_ = random.uniform(*snr['range'])
        elif snr['method'] == "choose" and isinstance(snr.get('value'), list): # {'method': 'choose', 'value': [5, 10, 15]}
            snr_ = random.choice(snr['value'])
        elif snr['method'] == 'const' and 'value' in snr: # {'method': 'const', 'value': 5}
            snr_ = snr['value']
        else:
            warnings.warn(f'NO EVENT ADDED: Invalid snr option: {snr}.')
            return

        if source_file_.startswith(event_dir): # Remove the base path
            source_file_ = os.path.relpath(source_file_, start=event_dir)

        target_event_list.append({
                'label': label_,
                'source_file': source_file_,
                'source_time': source_time_,
                'event_time': event_time_,
                'event_duration': event_duration_,
                'event_position': event_position_,
                'snr': snr_,
                'role': role,
        })

    def add_background(self,
        source_file=None,
        ):
        source_file = {'method': 'choose'} if source_file is None else source_file

        if not hasattr(self, 'room'):
            warnings.warn(f'NO BACKGROUND ADDED: before adding background, room must be set using either room_config option in constructor or set_room function!')
            return

        if not self.config.get('background_dir'):
            warnings.warn(f'NO BACKGROUND ADDED: No background_dir specified')
            return

        if source_file['method'] == "choose":
            if isinstance(source_file.get('value'), list):
                source_file_ = random.choice(source_file['value'])
            else:
                bg_file_list = get_files_list(self.config['background_dir'], '.wav', recursive=True)
                if not bg_file_list:
                    warnings.warn(f'NO BACKGROUND ADDED: No valid background files')
                    return
                source_file_ = random.choice(bg_file_list)
        elif source_file['method'] == 'const' and 'value' in source_file:
            source_file_ = source_file['value']
        else:
            warnings.warn(f'NO BACKGROUND ADDED: Invalid source_file option: {source_file}.')
            return

        room_nchans = self.room.get_nchan()
        with wave.open(source_file_, 'rb') as f:
            # validate number of channels
            nchans = f.getnchannels()
            if nchans != room_nchans: # TODO: add 1 channel noise
                warnings.warn(f'NO BACKGROUND ADDED: Background channel mismatch.\n{source_file_} has {nchans} channels, while impulse responses have {room_nchans} channels.')
                return
            # calculate duration
            sr = f.getframerate()
            wlen = f.getnframes()
            ambient_noise_duration = wlen / float(sr)

        # ambient_noise_duration = librosa.get_duration(path=source_file_)
        if ambient_noise_duration > self.config['duration']:
            source_time_ = math.floor(random.uniform(0, ambient_noise_duration - self.config['duration']))
        else:
            source_time_ = None

        # remove the directory from source_file path
        if source_file_.startswith(self.config['background_dir']):
            source_file_ = os.path.relpath(source_file_, start=self.config['background_dir'])

        self.bg_events.append({
                'label': None,
                'source_file': source_file_,
                'source_time': source_time_,
                'event_time': 0, # across mixture
                'event_duration': self.config['duration'], # mixture length
                'event_position': None,
                'snr': 0, # background level is always 0 dB
                'role': 'background',
        })

    #======================================================
    # SYNTHESIZE
    #======================================================
    def _synthesize_one_background(self, mixture, event, return_option):
        return_original_source = return_option.get('original', False)
        return_metadata = return_option.get('metadata', False)
        return_waveform = return_option.get('waveform', False)
        reobj = {}
            
        if return_metadata: reobj['metadata'] = event;
        
        # process relative path
        if event['source_file'].startswith(self.config['background_dir']):
            source_file_full = event['source_file']
        else:
            source_file_full = os.path.join(self.config['background_dir'], event['source_file'])

        if event['source_time'] is not None:
            ambient, _ = librosa.load(
                source_file_full,
                sr=self.config['sr'],
                offset=event['source_time'],
                duration=event['event_duration'],
                mono=False, # default True
            ) # [4, duration]
        else:  # repeat ambient file until scape duration
            ambient, _ = librosa.load(source_file_full, sr=self.config['sr'], mono=False)
            total_samples = int(self.config['duration'] * self.config['sr'])
            repeats = -(-total_samples // ambient.shape[-1])
            ambient = np.tile(ambient, (1, repeats))[:, :total_samples]

        if return_original_source: reobj['waveform_ori'] = ambient;

        if np.max(np.abs(ambient)) == 0:
            warnings.warn(f"{source_file_full} is an all-zero signal")
            # TODO: process all zeros signal
            scaled_ambient = ambient
        else:
            energy_db = 10*np.log10(np.mean(np.square(ambient)))
            event_scale = 10 ** ((self.config['ref_db'] + event['snr'] - energy_db) / 20)
            scaled_ambient = event_scale * ambient # mixture will be updated directly
        mixture += scaled_ambient
        if return_waveform: reobj['waveform'] = scaled_ambient
            
        return reobj
    
    def _synthesize_one_event(self, mixture, event, return_option):
        return_original_source = return_option.get('original', False)
        return_metadata = return_option.get('metadata', False)
        return_wet = return_option.get('wet', False)
        return_dry = return_option.get('dry', False)

        reobj = {}
        if return_metadata: reobj['metadata'] = event
        
        source_dir = self.config['foreground_dir'] if event['role'] == 'foreground' else self.config['interference_dir']
        if event['source_file'].startswith(source_dir): source_file = event['source_file']
        else: source_file = os.path.join(source_dir, event['source_file'])
        # load signal
        x, _ = librosa.load(
            source_file,
            sr=self.config['sr'],
            offset=event['source_time'],
            duration=event['event_duration'],
            mono=True,
        )
        if return_original_source: reobj['waveform_ori'] = x
        
        if np.max(np.abs(x)) == 0:
            warnings.warn(f"Warning: {source_file} is an all-zero signal")
            # TODO: process all zeros signal
        else:
            x = x / np.max(np.abs(x))
            
        sync_event = self.room.synthesize(
            source_signal=x,
            sr=self.config['sr'],
            source_position=event['event_position'],
            return_option=return_option)

        if np.max(np.abs(x)) == 0:
            event_scale = 1 # for return_dry
        else:
            energy_db = 10*np.log10(np.mean(np.square(sync_event['waveform'])))
            event_scale = 10 ** ((self.config['ref_db'] + event['snr'] - energy_db) / 20)
            sync_event['waveform'] = event_scale * sync_event['waveform']

        # add to mixture
        onsamp = int(event['event_time'] * self.config['sr'])
        max_length = min(onsamp + sync_event['waveform'].shape[-1], mixture.shape[-1])
        mixture[..., onsamp : max_length] += sync_event['waveform'][..., : max_length - onsamp]

        if return_wet:
            zero_padded_source = np.zeros_like(mixture)
            zero_padded_source[..., onsamp : max_length] = sync_event['waveform'][..., : max_length - onsamp]
            sync_event['waveform'] = zero_padded_source
        else: sync_event.pop('waveform')

        if return_dry:
            sync_event['waveform_dry'] = event_scale * sync_event['waveform_dry']
            zero_padded_source = np.zeros((sync_event['waveform_dry'].shape[0], mixture.shape[1]), dtype=mixture.dtype)
            zero_padded_source[..., onsamp : max_length] = sync_event['waveform_dry'][..., : max_length - onsamp]
            sync_event['waveform_dry'] = zero_padded_source

        reobj.update(sync_event)

        return reobj

    def synthesize(self, fg_return=None, int_return=None, bg_return=None):
        fg_return = {} if fg_return is None else fg_return
        int_return = {} if int_return is None else int_return
        bg_return = {} if bg_return is None else bg_return
        
        nchannel = self.room.get_nchan()
        mixture = np.zeros((nchannel, int(self.config['duration'] * self.config['sr'])))

        self.fg_events = sorted(self.fg_events, key=lambda x: x['event_time'])
        self.int_events = sorted(self.int_events, key=lambda x: x['event_time'])

        fg_events = []
        int_events = []
        bg_events = []

        # add background events
        for event in self.bg_events:
            sync_noise = self._synthesize_one_background(mixture, event, bg_return)
            bg_events.append(sync_noise)
        
        # add foregound events
        for event in self.fg_events:
            sync_event = self._synthesize_one_event(mixture, event, fg_return)
            fg_events.append(sync_event)

        for event in self.int_events:
            sync_event = self._synthesize_one_event(mixture, event, int_return)
            int_events.append(sync_event)
            
        output = {
            'mixture': mixture,
            'fg_events': fg_events,
            'int_events': int_events,
            'bg_events': bg_events,
        }
        return output

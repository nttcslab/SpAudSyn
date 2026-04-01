import glob
import os
import sys
import random
import numpy as np
import importlib

def get_files_list(path, extension, recursive=False):
    """
    extension including dot, e.g., '.sofa'
    use '*' for everything
    """
    if recursive:
        pattern = os.path.join(path, f"**/*{extension}")
    else:
        pattern = os.path.join(path, f"*{extension}")

    return glob.glob(pattern, recursive=recursive)

def get_labels(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name[0] != '.']

def find_event_time(mixture_duration, event_duration, max_overlap, existing_events):
    """
    Find a valid start time for a new event within a mixture so that at any point, the event overlap does not exceed max_overlap.
    Overlap errors within an increment are negligible
    Parameters:
        mixture_duration (float)
        event_duration (float)
        max_overlap (int)
        existing_events (list of dict): each element is a dict with keys:
            - 'event_time' (float): Start time of the event.
            - 'event_duration' (float): Duration of the event.
    Returns:
        float or None:
            - A valid start time for the new event
            - None otherwise
    """
    assert event_duration <= mixture_duration, f'event duration ({event_duration}) must be smaller than mixture duration ({mixture_duration})'
    if len(existing_events) < max_overlap: # number of overlap cannot exceed max_overlap regardess start_time
        start_time = random.uniform(0, mixture_duration - event_duration)
        return float(start_time)
    elif mixture_duration == event_duration: # no place to add
        return None

    points = [] # start and end points of existing event
    for e in existing_events:
        points.append((e['event_time'], +1)) # +1 to n overlap
        points.append((e['event_time'] + e['event_duration'], -1)) # -1 to n overlap
    points.sort(key=lambda x: (x[0], -x[1]))
    points.append((mixture_duration, max_overlap))

    current_overlap = 0
    valid_segments = []
    valid_start = 0  # start of current free segment, -1 for invalid, always start from 0

    for time, delta in points:
        current_overlap += delta
        if current_overlap >= max_overlap: # from here, not valid
            if valid_start is not None: # if current is valid
                if time - valid_start >= event_duration:
                    valid_segments.append((valid_start, time)) # add valid interval
                valid_start = None # current is not valid
        elif valid_start is None: valid_start = time
            
    if not valid_segments:
        return None

    # Pick random valid interval and a start time within it
    start, end = random.choice(valid_segments)
    start_time = start + random.uniform(0, end - start - event_duration)
    return start_time

def trim_signal(signal, trim_amplitude=0.0, margin_samples=0):
    """
    Find the start and end indices of a signal above a given amplitude threshold,
    with optional margin (in samples) around the detected region.
    Margin is only applied if margin_samples > 0 and trim_amplitude > 0.

    Parameters:
        signal (numpy.ndarray): 1D array
        trim_amplitude (float, optional): threshold to detect signal
        margin_samples (int, optional): number of samples to extend before start and after end

    Returns:
        tuple: (start, end)
        If no sample exceeds the threshold, returns (0, 0).
    """
    mask = np.abs(signal) > trim_amplitude
    if not np.any(mask):
        return 0, 0

    start = np.argmax(mask)
    end = len(signal) - 1 - np.argmax(mask[::-1])

    # Apply margin only if conditions are met
    if margin_samples > 0 and trim_amplitude > 0:
        start = max(0, start - margin_samples)
        end = min(len(signal) - 1, end + margin_samples)

    return start, end

def initialize_config(module_cfg, reload=False):
    if reload and module_cfg["module"] in sys.modules: module = importlib.reload(sys.modules[module_cfg["module"]])
    else: module = importlib.import_module(module_cfg["module"])
    func = module
    for part in module_cfg["main"].split("."): func = getattr(func, part)
    return func(**module_cfg["args"]) if 'args' in module_cfg.keys() else func()

def source_file_filter(event_dir,
                       label,
                       added_source_files,
                       exclusion_folder_depth):
    """
    Return a list of source files under (event_dir/label) that do not
    come from the same sound source as those in `added_source_files`.

    Parameters
    ----------
    event_dir : str
        Root sound event directory.
    label : str
        Subfolder inside event_dir.
    added_source_files : list[str]
        Relative paths starting with `label/...`.
    exclusion_folder_depth : int
        Exclusion depth relative to (event_dir/label/level1/level2...):
            0 → remove only exact files
            1 → remove same first-level subfolder
            2 → remove same second-level subfolder
            etc.

    Returns
    -------
    list[str]
        Filtered list of absolute file paths.
    """

    basefolder = os.path.join(event_dir, label)
    all_wav_files = get_files_list(basefolder,
                                   extension=".wav",
                                   recursive=True)
    
    # if there are no added source files, filter nothing
    if not added_source_files: return all_wav_files

    # process added source files: Convert added files (relative) to absolute
    added_abs = {os.path.join(event_dir, f) for f in added_source_files}

    if exclusion_folder_depth == 0:
        return [f for f in all_wav_files if f not in added_abs]

    # Build forbidden prefixes relative to basefolder
    forbidden = set()
    for f in added_abs:
        try: rel = os.path.relpath(f, basefolder)
        except ValueError: continue

        parts = rel.split(os.sep) # including filename.wav
        if len(parts) > exclusion_folder_depth:
            forbidden.add(os.sep.join(parts[:exclusion_folder_depth]))

    # if there are no forbidden folder, just return
    if not forbidden: return [f for f in all_wav_files if f not in added_abs]

    result = []
    for f in all_wav_files:
        if f in added_abs: continue

        try: rel = os.path.relpath(f, basefolder)
        except ValueError: continue

        parts = rel.split(os.sep)

        if len(parts) > exclusion_folder_depth and \
           os.sep.join(parts[:exclusion_folder_depth]) in forbidden:
            continue

        result.append(f)

    return result

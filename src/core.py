""" 
=================================
Description
=================================
This file explicitly has helper functions.
This includes:
- Searching for files recursively within provided directories
- Smoothing / filtering functions
"""

# =================================
# Imports
# =================================

import os
import numpy as np
import glob
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter


# =================================
# UFile-Finding
# =================================

def find_files_by_pattern(
        src_dir:str, 
        pattern:str="*.csv"
):
    """ 
    Given a directory, use glob.glob to recursively search for any files matching a provided pattern.
    """
    return sorted(glob.glob(os.path.join(src_dir, pattern)))


# =================================
# Savitsky-Golay Smoothing Filter
# =================================

def savgol_smoothing(
    _df:pd.DataFrame,
    features:object,
    timestamp_col:str = "unix_ms",
    timestamp_in_milli:bool = True,
    window=7,
    poly:int=2,
    output_col_suffix:str = None
) -> pd.DataFrame:
    # Copy to prevent mutation
    df = _df.copy()
    # Get time characteristics
    t = df[timestamp_col].to_numpy()
    if timestamp_in_milli: t = t / 1000.0
    dt = np.mean(np.diff(t))
    # Smooth each column provided
    for c in features:
        output_col = c+output_col_suffix if output_col_suffix is not None else c
        df[output_col] = savgol_filter(df[c], window, poly, delta=dt)
    # Return!
    return df


# =================================
# Low-pass Smoothing Filter
# =================================

def low_pass_smoothing(
    _df:pd.DataFrame,
    features:object,
    timestamp_col:str = "unix_ms",
    timestamp_in_milli:bool = True,
    cutoff:float = 2.0,
    output_col_suffix:str = None
):
    # Helper function
    def lowpass(data, cutoff, fs, order=4):
        b, a = butter(order, cutoff / (fs / 2), btype='low')
        return filtfilt(b, a, data)
    
    # Copy the df to prevent mutation
    df = _df.copy()
    # Get time characteristics
    t = df[timestamp_col].to_numpy()
    if timestamp_in_milli: t = t / 1000.0
    fs = 1.0 / np.median(np.diff(t))
    # Smooth each column provided
    for c in features:
        output_col = c+output_col_suffix if output_col_suffix is not None else c
        df[output_col] = lowpass(df[c], cutoff=cutoff, fs=fs)
    # Return!
    return df

# =================================
# Calculate midpoints of a numpy array
# =================================

def calculate_midpoints(
    x
):
    assert len(x) >= 2, "To calculate midpoint, the query array must be at least 2 items long."
    mix_x = 0.5 * (x[:-1] + x[1:])
    return mix_x


# =================================
# Get trial details from a trial dataframe
# =================================

def data_from_trial(
    trial_df:pd.DataFrame, 
    trial_id:int,
    start_colname:str = 'sim_unix_ms',
    end_colname:str = "end_unix_ms"
):
    trial_row = trial_df[trial_df['trial_id'] == trial_id].iloc[0]
    trial_name = trial_row['trial_name']
    start_ms = trial_row[start_colname]
    end_ms = trial_row[end_colname]
    return trial_name, start_ms, end_ms
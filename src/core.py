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
    traj_df:pd.DataFrame,
    timestamp_col:str = "unix_ms",
    timestamp_in_milli:bool = True,
    cols_to_smooth = ['x_pos', 'y_pos'],
    window=7,
    poly:int=2
) -> pd.DataFrame:
    # Copy to prevent mutation
    df = traj_df.copy()
    # Get time characteristics
    t = df[timestamp_col].to_numpy()
    if timestamp_in_milli: t = t / 1000.0
    dt = np.mean(np.diff(t))
    # Smooth each column provided
    for c in cols_to_smooth:
        df[c] = savgol_filter(df[c], window, poly, delta=dt)
    # Return!
    return df


# =================================
# Low-pass Smoothing Filter
# =================================

def low_pass_smoothing(
    traj_df:pd.DataFrame,
    timestamp_col:str = "unix_ms",
    timestamp_in_milli:bool = True,
    cols_to_smooth = ['x_pos', 'y_pos'],
    cutoff:float = 2.0
):
    # Helper function
    def lowpass(data, cutoff, fs, order=4):
        b, a = butter(order, cutoff / (fs / 2), btype='low')
        return filtfilt(b, a, data)
    
    # Copy the df to prevent mutation
    df = traj_df.copy()
    # Get time characteristics
    t = df[timestamp_col].to_numpy()
    if timestamp_in_milli: t = t / 1000.0
    fs = 1.0 / np.median(np.diff(t))
    # Smooth each column provided
    for c in cols_to_smooth:
        df[c] = lowpass(df[c], cutoff=cutoff, fs=fs)
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


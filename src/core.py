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


    def lowpass_with_nans(data, cutoff, fs):
        data = np.asarray(data)
        result = np.full_like(data, np.nan, dtype=float)
        valid = ~np.isnan(data)
        if not np.any(valid): 
            return result

        # Find contiguous valid segments
        idx = np.where(valid)[0]
        splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for seg in splits:
            if len(seg) < 5:  # too short for filtfilt
                continue

        segment_data = data[seg]
        try:
            filtered = lowpass(segment_data, cutoff, fs)
            result[seg] = filtered
        except Exception:
            # fallback: leave as NaN if filtering fails
            pass
        # Return
        return result
    
    # Copy the df to prevent mutation
    df = _df.copy()
    # Get time characteristics
    t = df[timestamp_col].to_numpy()
    if timestamp_in_milli: t = t / 1000.0
    fs = 1.0 / np.median(np.diff(t))
    # Smooth each column provided
    for c in features:
        output_col = c+output_col_suffix if output_col_suffix is not None else c
        df[output_col] = lowpass_with_nans(df[c], cutoff=cutoff, fs=fs)
    # Return!
    return df

def lowpass_smoothing(
    _df:pd.DataFrame,
    features:object,
    timestamp_col:str = "unix_ms",
    timestamp_in_milli:bool = True,
    cutoff:float = 2.0,
    output_col_suffix:str = None,
    order:int = 4,
):
    def lowpass_with_nans(data, cutoff, fs, order):
        data = np.asarray(data, dtype=float)

        # Output initialized as NaN
        result = np.full_like(data, np.nan, dtype=float)

        # Design filter once
        b, a = butter(order, cutoff / (fs / 2), btype='low')

        valid_mask = ~np.isnan(data)

        if not np.any(valid_mask):
            return result

        # Find contiguous valid segments
        idx = np.where(valid_mask)[0]
        splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

        min_len = 3 * order  # safe minimum for filtfilt

        for seg in splits:
            segment_data = data[seg]

            # If too short → just copy raw data
            if len(seg) < min_len:
                result[seg] = segment_data
                continue

            try:
                filtered = filtfilt(b, a, segment_data)

                # 🔴 Critical: detect silent NaN failures
                if np.isnan(filtered).any():
                    result[seg] = segment_data  # fallback
                else:
                    result[seg] = filtered

            except Exception:
                # Any failure → fallback to raw
                result[seg] = segment_data

        # 🔴 Safety check: ensure no valid data was lost
        original_valid = np.sum(valid_mask)
        result_valid = np.sum(~np.isnan(result))

        if result_valid < original_valid:
            print(f"⚠️ Warning: lost {original_valid - result_valid} valid points during filtering")

        return result

    # --- MAIN ---
    df = _df.copy()

    # Compute sampling frequency
    t = df[timestamp_col].to_numpy()
    if timestamp_in_milli:
        t = t / 1000.0

    fs = 1.0 / np.median(np.diff(t))

    # Apply smoothing
    for c in features:
        output_col = c + output_col_suffix if output_col_suffix else c

        df[output_col] = lowpass_with_nans(
            df[c].to_numpy(),
            cutoff=cutoff,
            fs=fs,
            order=order
        )

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
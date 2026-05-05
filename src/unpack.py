""" 
=================================
Description
=================================
This file explicitly focuses on unpacking certain data.
This includes:
- Unzipping (hence "unpacking") participant `.zip` files
- Extracting (hence "unpacking") timestamp columns from any `.csv` file
- Combining multiple calibration files (hence "unpacking")
- Inferring trials from calibration and eye data (hence "unpacking")
- Copying and moving offset files (hence "unpacking")
- Extracting only confederate pedestrians (hence "unpacking")
"""

# =================================
# Imports
# =================================

import os
import shutil
import numpy as np
import pandas as pd
import zipfile
# ---------------------
from . import core


# =================================
# Unpacking (or unzipping) `.zip` files that represent participant data
# =================================

def unzip_pdir(
    src_filepath:str,       # The input zip file
    out_dirpath:str = None  # Where do we save the files?
):
    # Prepare outpath
    if out_dirpath is None: 
        out_dirpath = os.path.dirname(src_filepath)
    
    # Ensure that the outpath exists
    os.makedirs(out_dirpath, exist_ok=True)

    # Read archive
    with zipfile.ZipFile(src_filepath) as archive:
        names = archive.namelist()

        # Account for Mac OS X
        def is_valid(name:str):
            return not (
                name.startswith('_MACOSX') or
                os.path.basename(name).startswith("._")
            )
        valid_names = [n for n in names if is_valid(n)]

        # Detect common to-level folder
        top_levels = [name.split('/')[0] for name in valid_names if name.strip()]
        common_root = None 
        if len(set(top_levels)) == 1:
            common_root = top_levels[0]

        for _file in valid_names:
            target = _file
            # Strip the common root folder, if it exists
            if common_root:
                parts = _file.split('/', 1)
                if len(parts) > 1: 
                    target = parts[1]
                else:
                    continue
            if not target:
                continue

            target_path = os.path.join(out_dirpath, target)

            # Create directories as needed
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if not _file.endswith('/'):
                with archive.open(_file) as source, open(target_path, 'wb') as dest:
                    dest.write(source.read())


# =================================
# Unpack Timestamp columns from any dataframe
# =================================

def timestamp_columns(
    filepath:str,
    ts_colnames,
    outpath:str = None
) -> pd.DataFrame:
    """
    Given the eye data (or some file with trial-length timestamps and frames),
    Generate a new DataFrame purely containing timestamps.
    - Returns:
        The timestamp dataframe
    """
    df = pd.read_csv(filepath)
    df = df[ts_colnames]
    if outpath is not None:
        outdir = os.path.split(outpath)[0]
        os.makedirs(outdir, exist_ok=True)
        df.to_csv(outpath, index=False)
    return df


# =================================
# Combine calibration files into a singular set of calibration and trial dataframes
# =================================

_CALIBRATION_COLUMNS = [
    'unix_ms', 
    'frame', 
    'rel_timestamp',
    'event', 
    'overlap_counter'
]

def calibrations(
    src_dir:str,
    pattern:str = 'calibration_*.csv',
    outpath:str = None
) -> pd.DataFrame:
    """ 
    Given a participant's directory, identify and concatenate all 
    calibration files.
    - Returns:  
        The concatenated dataframe of all calibration data
    """

    def handle_cal_file(
        _F:str
    ) -> pd.DataFrame:
        # Reading the DF
        df = pd.read_csv(_F)                            # Read the file
        df = df.iloc[:, :len(_CALIBRATION_COLUMNS)]     # Correction
        df.columns = _CALIBRATION_COLUMNS               # Apply new columns
        # Adding new columns
        fname = os.path.splitext(os.path.split(_F)[1])[0]
        cid = fname.split('_')[-1]
        if not cid.isdigit(): cid = 0
        df['trial_id'] = cid
        # Reorganize column names
        df = df.loc[:, ['trial_id']+_CALIBRATION_COLUMNS]
        # Return new dataframe
        return df

    files = core.find_files_by_pattern(src_dir, pattern=pattern)
    dfs = [handle_cal_file(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if outpath is not None:
        outdir = os.path.split(outpath)[0]
        os.makedirs(outdir, exist_ok=True)
        df.to_csv(outpath, index=False)
    return df


# =================================
# Infer trials based on calibration data
# =================================

_TRIAL_NAMES = [
    "Trial-ApproachAudio Start", 
    "Trial-BehindAudio Start", 
    "Trial-Behind Start", 
    "Trial-AlleyRunnerAudio Start", 
    "Trial-AlleyRunner Start", 
    "Trial-Approach Start"
]
_TRIAL_TARGET_POSITIONS = [
    (6.78, -2),     # 0th trial: calibration
    (-6.78, -2),    # 1st trial at index 1
    (6.78, -2),     # 2nd trial
    (-6.78, -2),    # 3rd trial
    (6.78, -2),     # 4th trial
    (-6.78, -2),    # 5th trial
    (6.78, -2),     # 6th trial|
    (-6.78, -2),    # 7th trial: extra
]
_TRIAL_GOAL_AXES = [
    np.arctan2(0,1),    # 0th trial: calibration
    np.arctan2(0,-1),   # 1st trial at index 1
    np.arctan2(0,1),    # 2nd trial
    np.arctan2(0,-1),   # 3rd trial
    np.arctan2(0,1),    # 4th trial
    np.arctan2(0,-1),   # 5th trial
    np.arctan2(0,1),    # 6th trial
    np.arctan2(0,-1),   # 7th tria,: extra 
]

def trials(
    cdf_path:str,
    edf_path:str,
    outpath:str = None
) -> pd.DataFrame:
    # Helper functions
    def recombine_calibration(
        src:str
    ) -> pd.DataFrame:
        df = pd.read_csv(src)               # Read csv file
        df = df[df['trial_id']!=0]          # Remove 0th trial
        df = df[df['event']!='Overlap']     # Remove any rows with 'Overlap'
        df = df.drop(columns=['rel_timestamp','overlap_counter'])   # Drop useless columns
        # Pivot event values into columns
        out = df.pivot(index="trial_id", columns="event", values=["unix_ms", "frame"])
        # Flatten multi-index columns
        out.columns = [ f"{val}_{evt.lower()}" for val, evt in out.columns ]
        # Reset index
        out = out.reset_index()
        # Rename to desired schema
        out = out.rename(columns={
            "unix_ms_start": "cal_unix_ms",
            "frame_start": "cal_frame",
            "unix_ms_end": "sim_unix_ms",
            "frame_end": "sim_frame",
        })
        # Reorder columns
        out = out[['trial_id','cal_unix_ms','cal_frame','sim_unix_ms','sim_frame']]
        # The "end_unix_ms" and "end_frame" is the start of the calibration of the next trial
        out["end_unix_ms"] = out["cal_unix_ms"].shift(-1)
        out["end_frame"] = out["cal_frame"].shift(-1)
        # Drop the 0th and 7th trial, if they exist
        out = out[out['trial_id'].between(1,6)]
        # Return
        return out

    def recombine_eye(
        src:str
    ) -> pd.DataFrame:
        # Read data, and only keep only relevant events
        eye = pd.read_csv(src)
        eye = eye[eye['event'].isin(_TRIAL_NAMES)]
        # Rename columns and generate other columns
        ends = eye[['event', 'unix_ms', 'frame']].rename(columns={'event':'trial_name', 'unix_ms':'sim_unix_ms', 'frame':'sim_frame'})
        ends = ends.sort_values('sim_unix_ms')
        ends['trial_id'] = range(1, len(ends.index) + 1)
        ends["trial_name"] = ends["trial_name"].str.replace(" Start", "", regex=False)
        ends["trial_name"] = ends["trial_name"].str.replace("Trial-", "", regex=False)
        ends["trial_audio"] = ends["trial_name"].str.contains("Audio", na=False)
        # Drop unnecessary columns
        ends = ends[['trial_id','trial_name','trial_audio']]
        # Return
        return ends

    # Extract both the calibration and eye data
    cdf = recombine_calibration(cdf_path)
    edf = recombine_eye(edf_path)

    # Merge on left
    trials = cdf.merge(edf, on="trial_id", how="left")
    trials = trials[[
        'trial_id','trial_name','trial_audio',
        'cal_unix_ms','cal_frame',
        'sim_unix_ms','sim_frame',
        'end_unix_ms','end_frame'
    ]]
    # Add trial details such as goal axis and goal point
    trials["goal_x"] = trials["trial_id"].map(
        lambda t: _TRIAL_TARGET_POSITIONS[int(t)][0]
    )
    trials["goal_y"] = trials["trial_id"].map(
        lambda t: _TRIAL_TARGET_POSITIONS[int(t)][1]
    )
    trials["goal_axis"] = trials["trial_id"].map(
        lambda t: _TRIAL_GOAL_AXES[int(t)]
    )
    # if outpath is provided, we save. Then we return the trial DF
    if outpath is not None:
        trials.to_csv(outpath, index=False)
    return trials


# =================================
#  Copy and Move Offsets (dunno if this is actually needed, but still)
# =================================

def offsets(
    pid:str
) -> bool:
    in_offset_filepath = f'./data/offsets/{pid}/offsets.csv'
    out_offset_filepath = f'./data/processed/{pid}/offsets.csv'
    # check: does input offset exist?
    if not os.path.isfile(in_offset_filepath):
        return False
    # Copy file
    shutil.copy2(in_offset_filepath, out_offset_filepath)
    return True


# =================================
# Deriving confederate agents from pedestrians
# =================================

def confederates(
    src_path:str,
    frame_ts_map:pd.DataFrame = None,
    outpath:str = None
) -> pd.DataFrame:
    # Read raw pedestrian data
    pdf = pd.read_csv(src_path)
    # Temporarily, get only the pedestrians that existed at some point on the same sidewalk (by z-position)
    ydf = pdf[(pdf['Label']=='Pedestrian') & (pdf['pos_y']<=0.0)]
    # Get the unique IDs of pedestrians
    confederate_ids = ydf['id'].unique()
    # Extract only confederate rows
    df = ydf[ydf['id'].isin(confederate_ids)]
    # Define the output columns
    output_columns = ['frame','agent_id','x_pos','y_pos','x_for','y_for']
    # If a timestamp-frame mapper is provided, then we match
    if frame_ts_map is not None:
        df = pd.merge(
            df, 
            frame_ts_map,
            how="left",
            on="frame"
        )
        output_columns = ['unix_ms'] + output_columns
    # Rename needed columns, drop unecessary ones
    df = df.rename(columns={ 'pos_x':'x_pos', 'pos_y':'y_pos', 'for_x':'x_for', 'for_y':'y_for', 'id':'agent_id' })
    df = df[output_columns]
    # Save plot if prompted; then return the dataframe
    if outpath is not None:
        outdir = os.path.split(outpath)[0]
        os.makedirs(outdir, exist_ok=True)
        df.to_csv(outpath, index=False)
    return df
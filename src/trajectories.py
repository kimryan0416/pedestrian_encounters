
# =================================
# Imports
# =================================

import numpy as np
import pandas as pd
# ----------------------
from . import core

# =================================
# Extracting raw trajectories from eye data
# =================================

def extract_trajectory(
    eye:pd.DataFrame,
    smooth_method:str = "low_pass"
) -> pd.DataFrame:
    # Data Cleanup. 
    #   We are attempting to fill up all NaN values if present
    eye["event"] = eye["event"].str.strip().fillna('')
    eye['gaze_target_name'] = eye['gaze_target_name'].convert_dtypes().fillna('')
    # Extract rows only relevant to participant.
    #   This operation requires that we remove all events and that we don't track weird issues like the person being placed at (0,0)
    participant_df = eye[
        (~eye['event'].str.match(r".+ Start$", na=False)) 
        & (
            (eye['head_position_x'] != 0.0) 
            & (eye['head_position_z'] != 0.0)
        )]
    # Extract columns only relevant to movement + rename them to make more sense
    # Length = N
    df = participant_df[['head_position_x','head_position_z', 'head_direction_x', 'head_direction_z', 'unix_ms', 'frame']] \
        .rename(columns={
            'head_position_x':'x_pos',
            'head_position_z':'y_pos',
            'head_direction_x':'x_for',
            'head_direction_z':'y_for'
        })
    # Sort by frame
    df = df.sort_values('frame').reset_index(drop=True)
    # Smoothen if prompted
    if smooth_method == "low_pass":
        df = core.low_pass_smoothing(df)
    elif smooth_method == "savgol":
        df = core.savgol_smoothing(df)
    # cleanup NaN that still happen to remain
    df = df.dropna()
    return df

# `traj` is a 4D list. Each entry comprises of an (x,y,t,f) tuple.
def extract_velocity(
    traj:pd.DataFrame
) -> pd.DataFrame:
    df = traj.copy()
    df = df.sort_values('unix_ms').reset_index(drop=True)
    df["dt"] = df["unix_ms"].diff() / 1000.0
    df["dx"] = df["x_pos"].diff()
    df["dy"] = df["y_pos"].diff()
    df["vx"] = df["dx"] / df["dt"]
    df["vy"] = df["dy"] / df["dt"]
    df = df[['vx', 'vy', 'dt', 'unix_ms', 'frame']]
    df.loc[0, ["dt", "vx", "vy"]] = 0
    return df

# `velocity` is a 3D list. Each entry comprises of an (vx,vy,dt) tuple
def extract_acceleration(
    velocity:pd.DataFrame
) -> pd.DataFrame:
    df = velocity.copy()
    # Compute velocity differences
    df["dx"] = df["vx"].diff()
    df["dy"] = df["vy"].diff()
    # Use dt from the *current* timestep
    df["ax"] = df["dx"] / df["dt"]
    df["ay"] = df["dy"] / df["dt"]
    # Get only relevants
    df = df[['ax', 'ay', 'dt', 'unix_ms', 'frame']]
    # Clean invalid values (tiny or bad dt already handled upstream ideally)
    df.loc[0, ["ax", "ay"]] = 0
    return df

def extract_scalars(
    velocity_df: pd.DataFrame,
    acceleration_df: pd.DataFrame
) -> pd.DataFrame:
    # Merge on time to align properly
    df = pd.merge_asof(
        velocity_df.sort_values("unix_ms"),
        acceleration_df.sort_values("unix_ms"),
        on="unix_ms",
        direction="nearest"
    )
    # 1. Speed
    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
    # 2. Unit velocity vector (v_hat)
    df["v_mag"] = df["speed"]
    df["v_hat_x"] = 0.0
    df["v_hat_y"] = 0.0
    valid = df["v_mag"] > 1e-6
    df.loc[valid, "v_hat_x"] = df.loc[valid, "vx"] / df.loc[valid, "v_mag"]
    df.loc[valid, "v_hat_y"] = df.loc[valid, "vy"] / df.loc[valid, "v_mag"]
    # 3. Tangential acceleration (projection)
    df["force"] = (
        df["ax"] * df["v_hat_x"] +
        df["ay"] * df["v_hat_y"]
    )
    # Clean first rows (derivative artifacts)
    df.loc[:2, ["force"]] = 0

    speed = df[["speed", "dt_x", "unix_ms", "frame_x"]].rename(columns={'dt_x':'dt', 'frame_x':'frame'})
    force = df[['force', 'dt_y', 'unix_ms', 'frame_y']].rename(columns={'dt_y':'dt', 'frame_y':'frame'})

    return speed, force

# =================================
# Imports
# =================================

import numpy as np
import pandas as pd
# ----------------------
from . import core, unpack


# =================================
# Extracting raw trajectories from eye data
# =================================

def extract_trajectory(
    eye:pd.DataFrame,
    trial_df:pd.DataFrame = None,
    trial_ts_colname:str = 'cal_unix_ms'
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
    
    # Sort by frame, and then cleanup any NaNs that still appear
    df = df.sort_values('frame').reset_index(drop=True)
    df = df.dropna()

    # Calculate delta time, delta X, and delta Y to calculate velocity and speed
    df["dt"] = df["unix_ms"].diff() / 1000.0
    df["dx"] = df["x_pos"].diff()
    df["dy"] = df["y_pos"].diff()
    df["x_vel"] = df["dx"] / df["dt"]
    df["y_vel"] = df["dy"] / df["dt"]
    df.loc[0, ["dt", "x_vel", "y_vel"]] = 0
    df["speed"] = np.sqrt(df["x_vel"]**2 + df["y_vel"]**2)
    
    # We re-calculate `dx` and `dy` to calculate acceleration
    df["dx"] = df["x_vel"].diff()
    df["dy"] = df["y_vel"].diff()
    df["x_accel"] = df["dx"] / df["dt"]
    df["y_accel"] = df["dy"] / df["dt"]
    df.loc[0, ["x_accel", "y_accel"]] = 0
    # We have to be a bit careful with acceleration, but we can calculate force too
    df["v_hat_x"] = 0.0
    df["v_hat_y"] = 0.0
    valid = df["speed"] > 1e-6
    df.loc[valid, "v_hat_x"] = df.loc[valid, "x_vel"] / df.loc[valid, "speed"]
    df.loc[valid, "v_hat_y"] = df.loc[valid, "y_vel"] / df.loc[valid, "speed"]
    # 3. Tangential acceleration (projection)
    df["force"] = df["x_accel"] * df["v_hat_x"] + df["y_accel"] * df["v_hat_y"]
    # Clean first rows (derivative artifacts)
    #df.loc[:2, ["force"]] = 0

    # Drop useless columns `dx`,`dy`, `v_hat_x`, 'v_hat_y
    df = df.drop(columns=['dx', 'dy', 'v_hat_x', 'v_hat_y'])

    # If a trial dataframe is provided, we merge them
    if trial_df is not None and trial_ts_colname is not None:
        df = pd.merge_asof( df, trial_df, left_on='unix_ms', right_on=trial_ts_colname, direction='backward' )
        df['trial_id'] = np.where( df['unix_ms'] <= df['end_unix_ms'], df['trial_id'], np.nan )
        df['trial_id'] = df['trial_id'].astype('Int64')
        # Find first occurrence of 1 and last occurrence of 6. Then fill those before 1 with 0 and after 6 with 7
        t = df["trial_id"]
        first_one_idx = t[t == 1].index.min()
        df.loc[:first_one_idx, "trial_id"] = df.loc[:first_one_idx, "trial_id"].fillna(0)
        df.loc[:first_one_idx, "goal_axis"] = df.loc[:first_one_idx, "goal_axis"].fillna(unpack._TRIAL_GOAL_AXES[0])
        last_six_idx = t[t == 6].index.max()
        df.loc[last_six_idx:, "trial_id"] = df.loc[last_six_idx:, "trial_id"].fillna(7)
        df.loc[last_six_idx:, "goal_axis"] = df.loc[last_six_idx:, "goal_axis"].fillna(unpack._TRIAL_GOAL_AXES[7])

    # Return
    return df


# =================================
# Calculating Velocity
# =================================

# `traj` is a 4D list. Each entry comprises of an (x,y,t,f) tuple.
def extract_velocity(
    traj:pd.DataFrame,
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


# =================================
# Calculating Acceleration
# =================================

# `velocity` is a 3D list. Each entry comprises of an (vx,vy,dt) tuple
def extract_acceleration(
    velocity:pd.DataFrame,
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
    # Return
    return df


# =================================
# Calculating Speed and Force
# =================================

def extract_scalars(
    velocity_df: pd.DataFrame,
    acceleration_df: pd.DataFrame,
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

    # Return
    return speed, force

# =================================
# Calculating Heading
# =================================

def extract_movement_heading(
    move_df:pd.DataFrame,
    x_vel_col:str,
    y_vel_col:str,
    speed_col:str,
    x_headdir_col:str,
    y_headdir_col:str,
    speed_threshold:float = 0.005,
    unwrap:bool = False,
    calc_degrees:bool = True
) -> pd.DataFrame:
    # Copy the dataframe
    df = move_df.copy()
    # Derive movement heading as radians
    df['move_heading'] = np.where(
        df[speed_col] > speed_threshold,
        np.arctan2(df[y_vel_col], df[x_vel_col]),
        np.nan
    )
    # For NaN cases, we create 2 alternatives: persistent (assume from last known movement), and intent (derived from forward direction)
    #df['move_heading_pers'] = df['move_heading'].ffill().bfill()
    df['move_heading_intent'] = df['move_heading'].fillna(np.arctan2(df[y_headdir_col], df[x_headdir_col]))
    # Unwrap if needed
    if unwrap:
        df['move_heading'] = np.unwrap(df['move_heading'])
        #df['move_heading_pers'] = np.unwrap(df['move_heading_pers'])
        df['move_heading_intent'] = np.unwrap(df['move_heading_intent'])
    # Convert the radians to degrees
    if calc_degrees:
        df['move_heading_deg'] = np.degrees(df['move_heading'])
        #df['move_heading_pers_deg'] = np.degrees(df['move_heading_pers'])
        df['move_heading_intent_deg'] = np.degrees(df['move_heading_intent'])
    # Return
    return df

def infer_movement_relative_dir(
    mov_df:pd.DataFrame,
    heading_cols,
) -> pd.DataFrame:
    # Helper function
    """
    def wrap_pi(a):
        if pd.isna(a):
            return np.nan
        return (a + np.pi) % (2*np.pi) - np.pi
    """
    def wrap_pi_vec(a):
        return (a + np.pi) % (2*np.pi) - np.pi
    # Prevent mutation
    df = mov_df.copy()
    # Calculate per heading col
    for col in heading_cols:
        """
        df[f"{col}_rel_goal"] = df.apply(lambda row: wrap_pi(row[col] - row['goal_axis']), axis=1)
        # Note: this will be between 1 and 0. 1 = directly moving toward goal axis, 0 = moving in the opposite direction
        df[f"{col}_toward_goal"] = np.where(
            df[f"{col}_rel_goal"].notna(),
            (1.0 + np.cos(df[f"{col}_rel_goal"])) / 2.0,
            np.nan
        )
        """
        rel = wrap_pi_vec(df[col] - df['goal_axis'])
        df[f"{col}_rel_goal"] = rel
    # Return!
    return df

# =================================
# Calculating Confederate Proxemics
# =================================

def calculate_proxemics(
    move_df:pd.DataFrame,
    confed_df:pd.DataFrame,
    radius:float = 5.0,                             # The radius of effect around the user. We normalize to this scale (0:5 -> 0:1).
    max_distance:float = np.sqrt(5**2 + 10**2),     # The maximum possible distance, to handle edge cases
    keep_confederate_details:bool=True
) -> pd.DataFrame:
    # Do a left merge on confeds based on frame, which would be more accurate at this point.
    df = pd.merge(
        move_df,
        confed_df.rename(columns={'unix_ms':'c_unix_ms', 'x_pos':'c_x_pos', 'y_pos':'c_y_pos', 'x_for':'c_x_for', 'y_for':'c_y_for', 'move_heading':'c_move_heading'}),  # ignore frame and agent_id
        how="left",
        on="frame"
    )
    # We calculate the distance. We set to maximum distance if any are NaN (e.g. there's no confederate agent provided)
    df["cx-x_diff"] = df["c_x_pos"] - df["x_pos"]
    df["cy-y_diff"] = df["c_y_pos"] - df["y_pos"]
    df['distance_to_confederate'] = np.sqrt( df['cx-x_diff']**2 + df['cy-y_diff']**2 )
    #df['distance_to_confederate'] = df['distance_to_confederate'].fillna(max_distance)
    # We also calculate the position of the confederate relative to the movement heading of the participant
    df["_dx"] = np.cos(df["move_heading_intent"])
    df["_dy"] = np.sin(df["move_heading_intent"])
    df["_dd"] = np.sqrt(df['_dx']**2 + df['_dy']**2)
    df["_dx_norm"] = df["_dx"] / df["_dd"]
    df["_dy_norm"] = df["_dy"] / df["_dd"]
    df["cx-x_diff_norm"] = df["cx-x_diff"] / df['distance_to_confederate']
    df["cy-y_diff_norm"] = df["cy-y_diff"] / df['distance_to_confederate']
    df["dot"] = df["_dx_norm"] * df["cx-x_diff_norm"] + df["_dy_norm"] * df["cy-y_diff_norm"]
        # > 0 = confederate in front of the participiant
        # < 0 confederate is behind the participant
        # == 0 = confederate is on the side of the participant
    df["cross"] = df["_dx_norm"] * df["cy-y_diff_norm"] - df["_dy_norm"] * df["cx-x_diff_norm"]
        # > 0 → confederate is on left
        # < 0 → confederate is on right
        # == 0 → perfectly aligned
    # drop irrelevant rows
    to_drop = ['_dx', '_dy', '_dx_norm', '_dy_norm', '_dd', 'cx-x_diff', 'cy-y_diff', 'cx-x_diff_norm', 'cy-y_diff_norm', 'c_unix_ms']
    if not keep_confederate_details:
        to_drop = to_drop + ['c_x_pos', 'c_y_pos', 'c_x_for', 'c_y_for', 'c_move_heading']
    df = df.drop(columns=to_drop)
    # Return!
    return df
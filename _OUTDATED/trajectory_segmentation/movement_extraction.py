import os
import numpy as np
import pandas as pd
import argparse

from cluster_analysis import assign_segments_from_zero_crossings, assign_segments_by_heading_changepoints, assign_segments_from_values, assign_indices_from_timestamp_intervals
from movement_plot import plot_speed_acceleration, plot_segmented_trajectory
from .traj_analysis.filtering import lowpass
import matplotlib.pyplot as plt
import mplcursors

_VIEWER_FOV_DEGREES = 90
_VIEWER_RADIUS = 2
_NAME_CATEGORY_DICT = {
    'NorthSidewalk': ['Environment', 'Sidewalk', 'North', 'NorthSidewalk'], 
    'SouthSidewalk': ['Environment', 'Sidewalk', 'South', 'SouthSidewalk'],
    'RoadEast': ['Environment', 'Road', 'East', 'RoadEast'], 
    'RoadWest': ['Environment', 'Road', 'West', 'RoadWest'],
    'RoadCrosswalk': ['Environment', 'Road', 'Crosswalk', 'RoadCrosswalk'],
    'NorthWalkingPole': ['Environment', 'Pole', 'North', 'NorthWalkingPole'], 
    'SouthWalkingPole': ['Environment', 'Pole', 'South', 'SouthWalkingPole'],
    'NorthCarSignal': ['Environment', 'Car_Signal', 'North', 'NorthCarSignal'],
    'SouthCarSignal': ['Environment', 'Car_Signal', 'South', 'SouthCarSignal'],
    'NorthBuildings_NoColliders': ['Environment', 'Buildings', 'North', 'NorthBuildings'],
    'SouthBuildings_NoColliders': ['Environment', 'Buildings', 'South', 'SouthBuildings'], 
    'NE_Tree_10': ['Environment', 'Tree', 'NE', 'NE_Tree_10'], 
    'NE_Tree_30': ['Environment', 'Tree', 'NE', 'NE_Tree_30'], 
    'NE_Tree_50': ['Environment', 'Tree', 'NE', 'NE_Tree_50'], 
    'NE_Tree_70': ['Environment', 'Tree', 'NE', 'NE_Tree_70'],
    'NW_Tree_10': ['Environment', 'Tree', 'NW', 'NW_Tree_10'], 
    'NW_Tree_30': ['Environment', 'Tree', 'NW', 'NW_Tree_30'], 
    'NW_Tree_50': ['Environment', 'Tree', 'NW', 'NW_Tree_50'], 
    'NW_Tree_70': ['Environment', 'Tree', 'NW', 'NW_Tree_70'],
    'SE_Tree_10': ['Environment', 'Tree', 'SE', 'SE_Tree_10'], 
    'SE_Tree_30': ['Environment', 'Tree', 'SE', 'SE_Tree_30'], 
    'SE_Tree_50': ['Environment', 'Tree', 'SE', 'SE_Tree_50'], 
    'SE_Tree_70': ['Environment', 'Tree', 'SE', 'SE_Tree_70'],
    'SW_Tree_10': ['Environment', 'Tree', 'SW', 'SW_Tree_10'], 
    'SW_Tree_30': ['Environment', 'Tree', 'SW', 'SW_Tree_30'], 
    'SW_Tree_50': ['Environment', 'Tree', 'SW', 'SW_Tree_50'], 
    'SW_Tree_70': ['Environment', 'Tree', 'SW', 'SW_Tree_70'],
    'ApproachAgent': ['Pedestrian', 'Confederate', 'Approach', 'ApproachAgent'], 
    'BehindRunner': ['Pedestrian', 'Confederate', 'Behind', 'BehindRunner'], 
    'AlleywayRunner': ['Pedestrian', 'Confederate', 'Alleyway', 'AllewayRunner'],
    '0': ['Pedestrian', 'Bystander', '', '0'], 
    '1': ['Pedestrian', 'Bystander', '', '1'], 
    '2': ['Pedestrian', 'Bystander', '', '2'], 
    '3': ['Pedestrian', 'Bystander', '', '3'], 
    '4': ['Pedestrian', 'Bystander', '', '4'], 
    '5': ['Pedestrian', 'Bystander', '', '5'], 
    '6': ['Pedestrian', 'Bystander', '', '6'], 
    '7': ['Pedestrian', 'Bystander', '', '7'], 
    '8': ['Pedestrian', 'Bystander', '', '8'], 
    '9': ['Pedestrian', 'Bystander', '', '9'],
    '10': ['Pedestrian', 'Bystander', '', '10'], 
    '11': ['Pedestrian', 'Bystander', '', '11'], 
    '12': ['Pedestrian', 'Bystander', '', '12'], 
    '13': ['Pedestrian', 'Bystander', '', '13'], 
    '14': ['Pedestrian', 'Bystander', '', '14'], 
    '15': ['Pedestrian', 'Bystander', '', '15'], 
    '16': ['Pedestrian', 'Bystander', '', '16'], 
    '17': ['Pedestrian', 'Bystander', '', '17'], 
    '18': ['Pedestrian', 'Bystander', '', '18'], 
    '19': ['Pedestrian', 'Bystander', '', '19'],
    '20': ['Pedestrian', 'Bystander', '', '20'], 
    '21': ['Pedestrian', 'Bystander', '', '21'], 
    '22': ['Pedestrian', 'Bystander', '', '22'], 
    '23': ['Pedestrian', 'Bystander', '', '23'], 
    '24': ['Pedestrian', 'Bystander', '', '24'], 
    '25': ['Pedestrian', 'Bystander', '', '25'], 
    '26': ['Pedestrian', 'Bystander', '', '26'], 
    '27': ['Pedestrian', 'Bystander', '', '27'], 
    '28': ['Pedestrian', 'Bystander', '', '28'], 
    '29': ['Pedestrian', 'Bystander', '', '29'],
}

# Calculate midpoints of a numpy array
def calculate_midpoints(x):
    assert len(x) >= 2, "To calculate midpoint, the query array must be at least 2 items long."
    mix_x = 0.5 * (x[:-1] + x[1:])
    return mix_x

# `eye_df` is a Pandas dataframe, usually read from an `eye.csv` file
def extract_trajectory(eye_df:pd.DataFrame):
    # Clean up 
    eye["event"] = eye["event"].str.strip().fillna('')
    eye['gaze_target_name'] = eye['gaze_target_name'].convert_dtypes().fillna('')
    # Extract rows only relevant to participant
    participant_df = eye[
        (~eye['event'].str.match(r".+ Start$", na=False)) 
        & (
            (eye['head_position_x'] != 0.0) 
            & (eye['head_position_z'] != 0.0)
        )]
    # Extract columns only relevant to movement + rename them to make more sense
    positions = participant_df[[
        'unix_ms', 'rel_timestamp', 'frame', 
        'head_position_x','head_position_y','head_position_z',
        'head_direction_x', 'head_direction_y', 'head_direction_z',
    ]].rename(columns={
        'head_position_x':'participant_pos_x',
        'head_position_y':'participant_pos_y',
        'head_position_z':'participant_pos_z',
        'head_direction_x':'participant_for_x',
        'head_direction_y':'participant_for_y',
        'head_direction_z':'participant_for_z'
    })
    # Extract raw positional data + time
    x_pos = positions['participant_pos_x'].to_numpy()   # X-axis positions. Length = N
    z_pos_raw = positions['participant_pos_z'].to_numpy()   # Z-axis positions. Length = N
    t = positions['unix_ms'].to_numpy()                 # Timestamps. Length = N
    frames = positions['frame'].to_numpy()
    # Smooth the position data with a low-pass filter to remove head-bob oscillations
    dt_sample_rate = np.median(np.diff(t)) / 1000.0
    fs = 1.0 / dt_sample_rate
    z_pos = lowpass(z_pos_raw, fs)
    # Combine into a single array
    traj = np.vstack((x_pos, z_pos, t, frames)).T
    # Extract the forward position
    positions_df = positions[['participant_pos_x', 'participant_pos_z', 'participant_for_x', 'participant_for_z', 'unix_ms', 'frame']]
    # Return both the trajectory and smoothing window
    return traj, positions_df, fs

# `traj` is a 3D list. Each entry comprises of an (x,y,t) tuple.
def extract_velocity(traj):
    dx = np.diff(traj[:,0:2], axis=0)               # Delta position. Length = N-1
    dt = np.diff(traj[:,2]) / 1000.0                # Delta time. Length = N-1
    v = dx / dt[:,None]                             # Velocity. Length = N-1
    velocity = np.column_stack([v[:,0], v[:,1], dt])
    return velocity

# `velocity` is a 3D list. Each entry comprises of an (vx,vy,dt) tuple
def extract_acceleration(velocity):
    dv = np.diff(velocity[:,0:2], axis=0)   # Delta velocity. Length = N-2
    dt = velocity[1:,2]                     # Delta time. Length = N-2
    a = dv / dt[:,None]                     # Acceleration. Length = N-2
    acceleration = np.column_stack([a[:,0], a[:,1], dt])
    return acceleration

# Given both velocity and acceleration, let's extract their scalar values
def extract_scalars(velocity, acceleration, smooth_fs:float=None):
    # Calculate raw speed
    speed = np.linalg.norm(velocity, axis=1)    # Length: N-1
    if smooth_fs is not None:
        speed = lowpass(speed, fs=fs)

    # Calculate speed appropriate for acceleration
    v_mid = calculate_midpoints(velocity)   # Length: N-2
    speed_mid = np.linalg.norm(v_mid, axis=1)   # Length: N-2
    if smooth_fs is not None:
        speed_mid = lowpass(speed_mid, fs=fs)

    # Calculate v_hat; only calculate based on valid entries 
    # to avoid division of 0 during accel_tangent calc.
    v_hat = np.zeros_like(v_mid)
    valid = speed_mid > 1e-6
    v_hat[valid] = v_mid[valid] / speed_mid[valid, None]

    # Calculate a_tangent
    a_tangent = np.zeros(len(acceleration))
    a_tangent[valid] = np.sum(acceleration[valid] * v_hat[valid], axis=1)
    if smooth_fs is not None:
        a_tangent = lowpass(a_tangent, fs=fs)

    # Return
    return speed, a_tangent

# Calculate moments when data crosses zero. Interpolates across time if needed.
def calculate_zero_crossings(t, x):
    sign = np.sign(x)
    crossings = np.where(sign[:-1] * sign[1:] < 0)[0]
    # linear interpolation
    t0 = t[crossings]
    t1 = t[crossings + 1]
    a0 = x[crossings]
    a1 = x[crossings + 1]
    t_cross = t0 - a0 * (t1 - t0) / (a1 - a0)
    return t_cross

# Returns times where motion enters or exits a stationary state
def calculate_stationary_events(
    t,
    speed,
    accel,
    v_thresh=0.01,
    a_thresh=0.01,
    min_duration=200
):
    stationary = (np.abs(speed) < v_thresh) & (np.abs(accel) < a_thresh)
    events = []
    start = None
    for i, is_stat in enumerate(stationary):
        if is_stat and start is None:
            start = i
        elif not is_stat and start is not None:
            duration = t[i] - t[start]
            if duration >= min_duration:
                events.append(t[start])
                events.append(t[i])
            start = None
    # Handle case where stationary lasts to end
    if start is not None:
        duration = t[-1] - t[start]
        if duration >= min_duration:
            events.append(t[start])
            events.append(t[-1])
    return np.array(events)

# Combines `calculate_zero_crossings` and `calcualte_stationary_events` into a single wrapper
def calculate_zero_events(
    t,
    accel,
    speed,
    v_thresh=0.05,
    a_thresh=0.05,
    min_stationary_time=200
):
    zc = calculate_zero_crossings(t, accel)
    stat = calculate_stationary_events(
        t,
        speed,
        accel,
        v_thresh,
        a_thresh,
    
        min_stationary_time
    )
    # Merge + sort + deduplicate
    events = np.unique(np.concatenate([zc, stat]))
    return events

# Custom helper function. Process raycast_hit_name into different segments, based on delimiters
def raycast_processing(raw_value):
    # Case 1: the cell value is empty
    if len(raw_value) == 0: return ['No Eye Hit','No Eye Hit','No Eye Hit','No Eye Hit']
    # Case 2: the value has an entry in `_NAME_CATEGORY_DICT`
    if raw_value in _NAME_CATEGORY_DICT:
        return _NAME_CATEGORY_DICT[raw_value]
    # Default: attempt to extract value from filename
    divided = raw_value.split("-")
    values = divided[0].split('.')
    values.append(divided[1])
    return values

# Extract eye-tracking gaze target intervals
def extract_gaze_targets(eye:pd.DataFrame, trial_start_unix_ms):
    # Read, ensure that "events" column is stripped of whitespace, and that "gaze_target_name" is valid-typed
    eye["event"] = eye["event"].str.strip().fillna('')
    eye['gaze_target_name'] = eye['gaze_target_name'].convert_dtypes().fillna('')

    # Split `raycast_hit_name` into different subdivisions, based on some delimiters in the name.
    eye[['gaze_target_category','gaze_target_type','gaze_target_subtype','gaze_target_id']] = eye.apply(lambda r: raycast_processing(r['gaze_target_name']), axis='columns', result_type='expand')
    # Check: some items have `Vehicel` instead of `Vehicle` in the 'gaze_target_category' column.
    eye.loc[eye['gaze_target_category'] == 'Vehicel', 'gaze_target_category'] = 'Vehicle'  
   
    # Segment the eye data into gazes with start-end timestamps
    eye["segment_id"] = ( (eye["gaze_target_category"] != eye["gaze_target_category"].shift()) | (eye["event"] != eye["event"].shift()) ).cumsum()
    gazes = eye.groupby("segment_id").agg(
            target=("gaze_target_type", "first"),
            start_unix_ms=("unix_ms", "min"),
            end_unix_ms=("unix_ms", "max"),
        ).reset_index(drop=True)
    gazes['rel_start_unix_ms'] = gazes['start_unix_ms'] - trial_start_unix_ms
    gazes['rel_end_unix_ms'] = gazes['end_unix_ms'] - trial_start_unix_ms
    gazes['duration_ms'] = gazes['end_unix_ms'] - gazes['start_unix_ms']
    # Remove segments with "No Eye Hit" as the target. Also remove observations shorter than 2 frame (roughly 16 milliseconds with 60 FPS)
    gazes = gazes[ (gazes['target'] != 'No Eye Hit') & (gazes['duration_ms'] >= 32) ]
    # Return
    return gazes

# Extract pedestrian data
def extract_pedestrian_data(peds_df:pd.DataFrame, participant_df:pd.DataFrame, trial_start_unix_ms):
    # Preparation.
    pos_df = (peds_df[peds_df['Label'] == 'Pedestrian'])                        # Get only their positions    
    ped_par_df = pd.merge(left=pos_df, right=participant_df, on='frame', how='left')    # Combine pedestrian and participant dfs via merge

    # Calculate distance
    ped_par_df['distance'] = np.hypot(
        ped_par_df['pos_x'] - ped_par_df['participant_pos_x'],
        ped_par_df['pos_y'] - ped_par_df['participant_pos_z']
    )

    # Calculate horizontal (along XZ plane), perpendicular distance of agent to forward vector of participant's viewing direction
    par_pos = ped_par_df[["participant_pos_x", "participant_pos_z"]].to_numpy()     # participant position (XZ)
    par_dir = ped_par_df[["participant_for_x", "participant_for_z"]].to_numpy()     # participant forward (XZ)
    ped_pos = ped_par_df[["pos_x", "pos_y"]].to_numpy()                             # pedestrian position (XZ)
    d_norm = par_dir / np.linalg.norm(par_dir, axis=1, keepdims=True)               # Normalizing view direction
    v = ped_pos - par_pos                                                           # Vector from participant to pedestrian
    v_norm = v / np.linalg.norm(v, axis=1, keepdims=True)                           # Normalizing vector from participant to pedestrian
    proj = np.sum(v * d_norm, axis=1, keepdims=True) * d_norm
    perp = v - proj
    cos_theta = np.sum(d_norm * v_norm, axis=1)                                     # Dot product = cos(theta)
    ped_par_df["view_distance"] = np.linalg.norm(perp, axis=1)
    
    # Positive → target on left side; Negative → target on right side
    ped_par_df["signed_view_distance"] = (d_norm[:, 0] * v[:, 1] - d_norm[:, 1] * v[:, 0])
    ped_par_df["in_front"] = cos_theta > 0
    
    # Now, calculate if a pedestrian is VISIBLE to the participant
    half_fov_rad = np.deg2rad(_VIEWER_FOV_DEGREES / 2)
    cos_half_fov = np.cos(half_fov_rad)
    ped_par_df["visible"] = (cos_theta >= cos_half_fov) & ped_par_df["in_front"]
    ped_par_df["side"] = np.where(ped_par_df['pos_y']<0, True, False)

    # From `in_front`, `visible`, and `signed_view_distance` generate viewpoint segments
    # The issue is that there can be overlaps (i.e. you can be seeing multiple things at the same time)
    # To offset this, rather than grouping by line,we need to follow these steps
    
    # Isolate the visible agents only (in the view frustrum, in front of the user)
    visibility_mask = ped_par_df["visible"] & ped_par_df["in_front"]
    visible = ped_par_df[visibility_mask].sort_values(["id", "frame"]).reset_index(drop=True)
    # Derive the direction of the object relative to the view frustum horizon
    visible["svd_sign"] = np.sign(visible["signed_view_distance"])
    # Identify segments from breaks in frame, ID, and sign
    frame_break = visible["frame"].diff() != 1
    id_break = visible["id"].ne(visible["id"].shift())
    sign_break = visible["svd_sign"].ne(visible["svd_sign"].shift())
    visible["segment_start"] = ( frame_break | id_break | sign_break)
    visible["segment_id"] = visible["segment_start"].cumsum()
    # Generate `view_df` from segmented data.
    view_df = (
        visible.groupby("segment_id").agg(
            id=("id", "first"),
            label=("Label", "first"),
            svd_sign=("svd_sign", "first"),
            start_frame=("frame", "first"),
            end_frame=("frame", "last"),
            start_unix_ms=("unix_ms", "first"),
            end_unix_ms=("unix_ms", "last"),
            duration_ms=("unix_ms", lambda x: x.iloc[-1] - x.iloc[0]),
        )
        .reset_index(drop=True)
    )
    view_df['rel_start_unix_ms'] = view_df['start_unix_ms'] - trial_start_unix_ms
    view_df['rel_end_unix_ms'] = view_df['end_unix_ms'] - trial_start_unix_ms

    # Return
    return ped_par_df, view_df

def interpret_distances(participant_df:pd.DataFrame, ped_par_df:pd.DataFrame):
    # Generate a new dataframe from `ped_par_df` based on frames
    # For each frame, first filter out all agents outside of `_VIEWER_RADIUS`
    # Then`count N agents within each frame and the closest agent's distance
    frame_stats = (
        ped_par_df
            .loc[ped_par_df["distance"] <= _VIEWER_RADIUS]
            .groupby("frame")
        .agg(
            n_agents=("distance", "size"),
            min_distance=("distance", "min"),
        )
        .reset_index()
    )

    # Merge with participant_df by frame
    par_df = pd.merge(left=participant_df, right=frame_stats, on='frame', how='left')
    par_df["n_agents"] = par_df["n_agents"].fillna(0).astype(int)
    par_df["min_distance"] = par_df["min_distance"].fillna(-1)

    # Return par_df
    return par_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', help="The relative path to the participant's directory", type=str)
    parser.add_argument('trial_id', help="The trial number that ought to be observed", type=int)
    parser.add_argument('-tf', '--trial_filename', help="The expected filename of the trial csv file", type=str, default='trials.csv')
    parser.add_argument('-ef', '--eye_filename', help="The expected filename of the eye-tracking file", type=str, default='eye.csv')
    parser.add_argument('-pf', '--ped_filename', help="The expected filename of the pedestrian file", type=str, default='pedestrians.csv')
    parser.add_argument('-ws', '--window_size', help="The minimum number of points expected for the sliding window operation", type=int, default=7)
    parser.add_argument('-et', '--error_threshold', help="The error threshold (in meters) between the extrapolated, estimated current position and actual position along hte trajectory", type=float, default=5)
    parser.add_argument('-fnan', '--fill_nan', help='How should we fill empty values in our velocity and acceleration?', type=str, choices=['prepend', 'append'], default=None)
    args = parser.parse_args()

    trial_src = os.path.join(args.src_dir, args.trial_filename)
    assert os.path.exists(trial_src), f"Trial file \"{args.trial_filename}\" doesn't appear to exist..."
    trials_df = pd.read_csv(trial_src)
    trial_row = trials_df.loc[trials_df['trial_id'] == args.trial_id]
    trial = trial_row.iloc[0].to_dict()

    trial_dir = os.path.join(args.src_dir, str(trial['trial_id']))
    assert os.path.exists(trial_dir), f"Trial directory \"{trial['trial_id']}\" doesn't appear to exist..."
    eye_filepath = os.path.join(trial_dir, args.eye_filename)
    assert os.path.exists(eye_filepath), f"Gaze file \"{args.eye_filename}\" not found..."
    eye = pd.read_csv(eye_filepath)

    # Basic trajectories
    traj, participant_df, fs = extract_trajectory(eye)
    velocity = extract_velocity(traj)
    acceleration = extract_acceleration(velocity)
    speed, force = extract_scalars(velocity, acceleration, smooth_fs=fs)
    t = traj[:,2]
    frames = traj[:,3]
    frame_timestamp_mapper = {frames[i]:t[i] for i in range(len(t))}
    timestamp_index_mapper = {t[i]:i for i in range(len(t))}
    speed_time = np.column_stack([speed, t[1:]])
    force_time = np.column_stack([force, t[2:]])

    #zero_crossings = calculate_zero_crossings(t[2:], force)
    zero_crossings = calculate_zero_events(t[2:], force, speed[1:], v_thresh=0.1, a_thresh=0.05, min_stationary_time=1000)
    move_accel_segments = assign_segments_from_zero_crossings(t[2:], zero_crossings)
    
    # Direction changes
    move_heading = np.unwrap(np.arctan2(velocity[:,1], velocity[:,0]))
    move_dir_segments, change_points = assign_segments_by_heading_changepoints(move_heading, penalty=6)

    # Extract gaze targets from eye data
    gaze_targets = extract_gaze_targets(eye, t[0])
    confederate_gazes = gaze_targets[gaze_targets['target'] == 'Confederate']
    confederate_gaze_segments = confederate_gazes[['start_unix_ms', 'duration_ms']].to_numpy()

    # Extract pedestrian data, which also needs the participant data too. This time with a timestamp based in frames
    pedestrians_raw_df = pd.read_csv(os.path.join(trial_dir, args.ped_filename))
    ped_par_df, view_df = extract_pedestrian_data(pedestrians_raw_df, participant_df, t[0])
    ped_df = ped_par_df.dropna()
    same_side_ped_df = ped_df[ped_df['side']]
    same_side_ped_ids = same_side_ped_df['id'].unique()
    visible_df = view_df[view_df['id'].isin(same_side_ped_ids)]
    visible_df['start_index'] = visible_df['start_unix_ms'].apply(lambda x: timestamp_index_mapper[x])
    visible_df['end_index'] = visible_df['end_unix_ms'].apply(lambda x: timestamp_index_mapper[x])
    visible_segments = assign_indices_from_timestamp_intervals(t, visible_df, start_ts_colname='start_index', end_ts_colname='end_index')
    print(visible_segments)
    
    distances_df = interpret_distances(participant_df, ped_df)
    n_agent_segments = distances_df['n_agents'].to_numpy()
    min_distances = distances_df['min_distance'].to_numpy()

    plot_segmented_trajectory(
        f"{os.path.basename(args.src_dir)} - T{args.trial_id}: {trial['trial_name']}",
        t[2:],
        traj[2:], 
        speed[1:], 
        force, 
        zero_crossings, 
        move_accel_segments,
        move_dir_segments[1:],
        confederate_gaze_segments,
        n_agent_segments[2:],
        min_distances[2:],
        visible_segments[2:],
        add_segment_labels_to_legend=False
    )


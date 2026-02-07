import os
import numpy as np
import argparse
import pandas as pd
from scipy.signal import butter, filtfilt
from movement_plot import plot_speed_acceleration, plot_velocity_over_windows, windowed_dbscan_inspector
from cluster_analysis import coassociation_strength, extract_segments

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Low-pass filter to reduce noise in raw data
def lowpass(signal, fs, cutoff=0.6, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal, axis=0)

# Calculate midpoints of a numpy array
def calculate_midpoints(x):
    assert len(x) >= 2, "To calculate midpoint, the query array must be at least 2 items long."
    mix_x = 0.5 * (x[:-1] + x[1:])
    return mix_x

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

    # Smooth the position data with a low-pass filter to remove head-bob oscillations
    dt_sample_rate = np.median(np.diff(t)) / 1000.0
    fs = 1.0 / dt_sample_rate
    z_pos = lowpass(z_pos_raw, fs)

    # Combine into a single array, and return both the trajectory and smoothing window
    traj = np.vstack((x_pos, z_pos, t)).T
    return traj, fs

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

"""
def extract_acceleration(velocity, dt):
    dv = np.diff(velocity[:,0:2], axis=0)       # Delta velocity. Length = N-2
    ddt = velocity[1:,2]                        # Delta Delta time. Length = N-2
    a = dv / ddt[:,None]                        # Acceleration. Length = N-2
    force = np.linalg.norm(a, axis=1)           # Acceleration Magnitude. Length = N-2
    return a, force, ddt
"""

"""
# `traj` is a 3D list. Each entry comprises of an (x,y,t) tuple
def extract_speed_acceleration(traj, fill_nan=None):
    dt = np.diff(traj[:,2]) / 1000.0        # Delta time. Length = N-1
    v = np.diff(traj[:,0:2], axis=0) / dt[:,None]   # Velocity. Length = N-1
    a = np.diff(v, axis=0) / dt[1:,None]         # Acceleration. Length = N-2

    # Callculate speed
    speed = np.linalg.norm(v[:-1], axis=1)
    valid = speed > 1e-6

    # smooth speed
    dt_sample_rate = np.median(np.diff(traj[:,2])) / 1000.0
    fs = 1.0 / dt_sample_rate
    speed = lowpass(speed, fs=fs)

    v_hat = v[:-1] / np.linalg.norm(v[:-1], axis=1, keepdims=True)
    a_tangent = np.zeros(len(a))
    a_tangent[valid] = np.sum(a[valid] * v_hat[valid], axis=1)
    a_tangent = lowpass(a_tangent, fs=fs)
    
    return speed, v, a_tangent, a
    """

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', help="The relative path to the participant's directory", type=str)
    parser.add_argument('trial_id', help="The trial number that ought to be observed", type=int)
    parser.add_argument('-tf', '--trial_filename', help="The expected filename of the trial csv file", type=str, default='trials.csv')
    parser.add_argument('-ef', '--eye_filename', help="The expected filename of the eye-tracking file", type=str, default='eye.csv')
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
    traj, fs = extract_trajectory(eye)
    velocity = extract_velocity(traj)
    acceleration = extract_acceleration(velocity)
    speed, force = extract_scalars(velocity, acceleration, smooth_fs=fs)

    #clustering_windows = windowed_dbscan(velocity[:,0:2], traj[1:,2], window_size=500, step=250, eps=0.025, min_samples=3)
    #same, total = build_coassociation(clustering_windows)
    #edges = coassociation_strength(same, total)
    #segments = extract_segments(edges, len(velocity))
    #segments_t = traj[:,2]
    #plot_velocity_over_windows(velocity[:,0:2], traj[1:,2], window_size=500, segments=segments)

    #windowed_dbscan_inspector(velocity[:,0:2], traj[1:,2])

    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")

    for k, idx in enumerate(segments):
        idx = np.array(idx)

        ax.plot(
            velocity[idx, 0],      # v_x
            velocity[idx, 1],      # v_y
            segments_t[1+idx],         # time
            color=cmap(k % cmap.N),
            linewidth=2,
            label=f"Segment {k}"
        )

    ax.set_xlabel("v_x")
    ax.set_ylabel("v_y")
    ax.set_zlabel("time (s)")
    ax.set_title("Velocity trajectory in (v_x, v_y, t) space")

    ax.legend()
    plt.show()
    """

    # For raw demonstration purposes, we plot the raw speed and acceleration
    speed_time = np.column_stack([speed, traj[1:,2]])
    force_time = np.column_stack([force, traj[2:,2]])
    zero_crossings = calculate_zero_crossings(traj[2:,2], force)
    plot_speed_acceleration(speed_time, force_time, zero_crossings=zero_crossings)
    #plot_velocity_over_windows(velocity[:,0:2], traj[1:,2])
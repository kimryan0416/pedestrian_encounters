import os
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.cluster import DBSCAN

def extract_velocity(traj):
    dx = np.diff(traj[:,0:2], axis=0)           # Delta position. Length = N-1
    dt = np.diff(traj[:,2]) / 1000.0            # Delta time. Length = N-1
    v = dx / dt[:,None]                         # Velocity. Length = N-1
    speed = np.linalg.norm(v, axis=1)           # Speed. Length = N-1
    return v, speed, dt

def extract_acceleration(velocity):
    dv = np.diff(velocity[:,0:2], axis=0)       # Delta velocity. Length = N-2
    ddt = velocity[1:,2]                        # Delta Delta time. Length = N-2
    a = dv / ddt[:,None]                        # Acceleration. Length = N-2
    force = np.linalg.norm(a, axis=1)           # Acceleration Magnitude. Length = N-2
    return a, force, ddt

def extract_speed_acceleration(traj):
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

def lowpass(signal, fs, cutoff=0.6, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal, axis=0)

def plot_speed_acceleration(speed, force, t):
    """
    velocity, speed, dt = extract_velocity(traj)
    velocity = np.vstack((velocity[:,0], velocity[:,1], dt)).T
    acceleration, force, ddt = extract_acceleration(velocity)
    print(acceleration)
    """
    # Plot speed
    fig = plt.figure(figsize=(10,5))
    plt.plot(t, speed, c='orange', zorder=2, label='speed')
    plt.plot(t, force, c='blue', zorder=1, label='acceleration')
    plt.axhline(y=0, c='black')
    for zero_crossing in zero_crossings(t, force):
        plt.axvline(x=zero_crossing, c='gray')
    plt.legend()
    plt.savefig('speed_accel.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_velocity_over_windows(v, t, window_size=1000):    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    sc = ax.scatter([], [], s=20)

    ax.set_xlabel("v_x")
    ax.set_ylabel("v_y")
    ax.set_aspect("equal")
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)

    ax_time = plt.axes([0.15, 0.18, 0.7, 0.03])
    time_slider = Slider(
        ax_time,
        "Center time (s)",
        t[0],
        t[-1],
        valinit=t[0]
    )

    ax_window = plt.axes([0.15, 0.12, 0.7, 0.03])
    window_slider = Slider(
        ax_window,
        "Window size (ms)",
        100,
        t[-1]-t[0],
        valinit=1000
    )

    def gaussian_alpha(t, t0, sigma):
        a = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
        return a / a.max()

    def update(_):
        t0 = time_slider.val
        W = window_slider.val
        half_w = W / 2

        mask = (t >= t0 - half_w) & (t <= t0 + half_w)
        v_win = v[mask]
        t_win = t[mask]

        if len(v_win) == 0:
            sc.set_offsets([])
            return

        # Gaussian sigma tied to window size
        sigma = W / 6
        alpha = gaussian_alpha(t_win, t0, sigma)

        sc.set_offsets(v_win)

        colors = np.zeros((len(v_win), 4))
        colors[:, :3] = [0.2, 0.4, 0.8]
        colors[:, 3] = alpha
        sc.set_facecolors(colors)

        ax.set_title(
            f"Velocity space\n"
            f"center={t0:.2f}s, window={W:.2f}s, samples={len(v_win)}"
        )

        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    window_slider.on_changed(update)
    update(None)
    plt.show()

def zero_crossings(t, a):
    sign = np.sign(a)
    crossings = np.where(sign[:-1] * sign[1:] < 0)[0]
    # linear interpolation
    t0 = t[crossings]
    t1 = t[crossings + 1]
    a0 = a[crossings]
    a1 = a[crossings + 1]

    t_cross = t0 - a0 * (t1 - t0) / (a1 - a0)
    return t_cross

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', help="The relative path to the participant's directory", type=str)
    parser.add_argument('trial_id', help="The trial number that ought to be observed", type=int)
    parser.add_argument('-tf', '--trial_filename', help="The expected filename of the trial csv file", type=str, default='trials.csv')
    parser.add_argument('-ef', '--eye_filename', help="The expected filename of the eye-tracking file", type=str, default='eye.csv')
    parser.add_argument('-ws', '--window_size', help="The minimum number of points expected for the sliding window operation", type=int, default=7)
    parser.add_argument('-et', '--error_threshold', help="The error threshold (in meters) between the extrapolated, estimated current position and actual position along hte trajectory", type=float, default=5)
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
    eye["event"] = eye["event"].str.strip().fillna('')
    eye['gaze_target_name'] = eye['gaze_target_name'].convert_dtypes().fillna('')
    participant_df = eye[
        (~eye['event'].str.match(r".+ Start$", na=False)) 
        & (
            (eye['head_position_x'] != 0.0) 
            & (eye['head_position_z'] != 0.0)
        )]
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
    
    x_pos = positions['participant_pos_x'].to_numpy()   # X-axis positions. Length = N
    z_pos_raw = positions['participant_pos_z'].to_numpy()   # Z-axis positions. Length = N
    t = positions['unix_ms'].to_numpy()                 # Timestamps. Length = N

    # Smooth the position data with a low-pass filter to remove head-bob oscillations
    dt_sample_rate = positions['unix_ms'].diff().median() / 1000.0
    fs = 1.0 / dt_sample_rate
    z_pos = lowpass(z_pos_raw, fs)

    # Combine into a single array
    traj = np.vstack((x_pos, z_pos, t)).T
    speed, velocity, force, acceleration = extract_speed_acceleration(traj)

    # DBScan analysis
    speed_with_time = np.column_stack([velocity[:,0], velocity[:,1], t[1:]/1000])
    db = DBSCAN( eps=0.05, min_samples=3 )
    labels = db.fit_predict(speed_with_time)
    print(labels)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        velocity[:,0],
        velocity[:,1],
        t[1:],
        c=labels,
        cmap="viridis",
        s=8
    )

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Time (s)")

    ax.set_title("DBSCAN clustering in (x, y, t) space")
    plt.colorbar(scatter, ax=ax, label="Cluster label")
    plt.show()


    # Plotting
    plot_speed_acceleration(speed, force, t[2:])
    plot_velocity_over_windows(velocity, t[1:])



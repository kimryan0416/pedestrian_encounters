import os
import time
import operator
import numpy as np
import pandas as pd
import math
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.signal import butter, filtfilt, savgol_filter
from collections import deque

def sws(traj, k:int, error_threshold:float):

    # Use DBSCAN to identify overlaps in trajectory, then provide a smoother trajectory
    def clean_trajectory(x, nn=15):
        # Clustering via DBSCAN
        dbscan_clusters = DBSCAN(eps=0.0015, min_samples=2).fit_predict(x)
        unique_labels = set(dbscan_clusters)
        filtered_unique_labels = [l for l in set(dbscan_clusters) if l != -1]

        # Combine each cluster
        clusters = []
        new_x = []
        first = 0
        for label in filtered_unique_labels:
            start_index = operator.indexOf(dbscan_clusters, label)
            end_index = len(dbscan_clusters) - operator.indexOf(reversed(dbscan_clusters), label)  # technically -1 too, but we wont for now.
            segment = x[start_index:end_index]
            avg = np.mean(segment, axis=0)
            clusters.append({
                'label':label,
                'start_index':start_index, 
                'end_index':end_index,
                'rep_point':avg,
                'segment':segment
            })
            if first != start_index:
                new_x.extend(x[first:start_index])
            new_x.append(avg)
            first = end_index
        if first != len(x):
            new_x.extend(x[first:len(x)])
        new_x = np.array(new_x)
        # Rebuilt a new trajectory

        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        fig,axes = plt.subplots(2,1)
        for lab, col in zip(unique_labels, colors):
            mask = dbscan_clusters == lab
            if lab == -1:
                axes[0].scatter(x[mask, 0], x[mask, 1], c="lightgray", alpha=0.1, s=20, label="Noise")
            else:
                axes[0].scatter(x[mask, 0], x[mask, 1], c=[col], s=20, label=f"Cluster {lab}", zorder=2)
        axes[0].scatter(x[0, 0], x[0, 1], c="green", marker='x', s=30, label="Start", zorder=3)
        axes[0].plot(x[:,0], x[:,1])
        axes[0].scatter(x[-1, 0], x[-1, 1], c="red", marker='x', s=30, label="End", zorder=3)
        axes[0].legend()
        print(new_x[:,0], new_x[:,1])
        axes[1].scatter(new_x[:,0], new_x[:,1], c='gray', alpha=0.05, s=20)
        axes[1].plot(new_x[:,0], new_x[:,1])
        axes[1].scatter(new_x[0, 0], new_x[0, 1], c="green", marker='x', s=30, label="Start", zorder=3)
        axes[1].scatter(new_x[-1, 0], new_x[-1, 1], c="red", marker='x', s=30, label="End", zorder=3)
        axes[1].legend()
        #plt.axis("equal")
        plt.show()
        return new_x

    # least-squares extrapolation
    def ls_extrapolation(x,xn:int=None):
        if xn is None: xn = len(x) # how many points in `x` to use for extrapolation?
        y = x[-xn:]
        t = np.arange(xn)
        # Solve least squares independently per dimension
        coeffs = np.linalg.lstsq(
            np.c_[t, np.ones(xn)], y, rcond=None
        )[0]
        v, b = coeffs
        return v * xn + b

    # cartesian coordinate extrapolation
    def euclidean_distance(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.hypot(dx, dy)

    def generate_error_signal(x):
        # Initialize
        e = []
        mid = math.floor(k/2)
        e.extend(np.zeros(mid))
        # iterate
        for i in range(mid, len(x)-mid):
            wind = x[i-mid:i+mid+1]                          # sliding window formation
            p_forward = ls_extrapolation(wind[:mid])            # forward extrapolation
            p_backward = ls_extrapolation(wind[-mid:][::-1])    # backward extrapolation
            p_est = (p_forward + p_backward) / 2                     # midpoint
            ei = euclidean_distance(x[mid],p_est)            # error
            e.append(ei)
        # close
        e.extend(np.zeros(mid))
        return e

    # Clean up trajectory
    new_traj = clean_trajectory(traj)
    # Initialize
    E = generate_error_signal(new_traj)
    q = deque()
    q.append((0,len(new_traj)))
    partition_indices = []

    # iterate
    while len(q) > 0:
        t = q.popleft()     # Get tuple of indices of `traj` to currenty look at
        ce = E[t[0]:t[1]]   # Extract relevant errors
        m_ce = max(ce)   # get them maximal error from the relevant errors
        m_indices = [
            i for i, v in enumerate(ce) if v == m_ce
        ]                   # Get the indices of ce where the max error was detected
        print(m_ce, m_indices)
        # If the error exceeds the threshold:
        if m_ce > error_threshold:
            # `ce` is an array of some length. its indices are with respect to that length
            # EX if `ce` is a 5-item array and we identified max values at indices 1 and 3...
            #   ... then its equivalent indices in `traj` are t[0] + each index
            #   because order is maintained here.
            # So if `first` = 0 on the first run of a 10-point traj, then 
            #   the segments are 0-1, 1-3, and 3-10 (in the first loop)
            first = t[0]
            for i in m_indices:
                partition_point = (first,t[0]+i)
                if (t[0]+i) - first < 1: continue
                q.append((first,t[0]+i))
                partition_indices.append(t[0]+i)
                first = t[0]+i+1
            if t[1] - first < 1: continue
            q.append((first,t[1]))
        """
        else:
            segments.append(t)
        """
        print(partition_indices)
    # Return segments
    return partition_indices

def lowpass(signal, fs, cutoff=0.6, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal, axis=0)


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
    print(trials_df.to_string() + "\n")
    trial_row = trials_df.loc[trials_df['trial_id'] == args.trial_id]
    trial = trial_row.iloc[0].to_dict()
    print(trial)

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

    # segment estimation
    traj = np.vstack((x_pos, z_pos)).T
    partition_indices = sws(traj, args.window_size, args.error_threshold)
    print(partition_indices)

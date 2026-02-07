import os
import numpy as np
import pandas as pd
import math
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.signal.windows import gaussian

_PRINT_INDENT = "  "
_DESCRIPTION = "Given a participant's directory and trial number, identify the events that occur within."
_VIEWER_FOV_DEGREES = 90
_VIEWER_RADIUS = 2
_GANTT_YLIMS = [-4.75, 2.25]
_REMOVE_HEADBOB = True
# category, type,       subtype,    unique_id
# Vehicle.  SportCar.   Driver-     SportCar2.Driver
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

# Custom helper function. Normalize a column
def normalize_column(col):
    _min = col.rolling('1s', center=True).min()
    _max = col.rolling('1s', center=True).max()
    return ((col - _min) / (_max - _min)).fillna(0.0).clip(0, 1)

def lowpass(signal, fs, cutoff=0.6, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal, axis=0)

def read_eye_data(src:str, trial_start_unix_ms):
    # Read, ensure that "events" column is stripped of whitespace, and that "gaze_target_name" is valid-typed
    eye = pd.read_csv(src)
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

    # Get the participant's positions. Filter out events and position=(0,0) (which is impossible given our experimental setup)
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
    positions['time'] = pd.to_datetime(positions['unix_ms'], unit='ms')
    positions = positions.set_index('time').sort_index()

    # For Gaussian smoothing for any data value of our choice
    # Whenever we want to smooth, use a rolling window with "gaussian" as the type and `sigma_samples` for the window size
    sigma_ms = 150  # We can also try 50–150 ms
    sigma_samples = sigma_ms / positions['unix_ms'].diff().median()

    # We effectively need to perform a lot of calculations for extracting transition events
    # First step: Extract positions and timestamps
    x_pos = positions['participant_pos_x'].to_numpy()   # X-axis positions. Length = N
    z_pos_raw = positions['participant_pos_z'].to_numpy()   # Z-axis positions. Length = N
    t = positions['unix_ms'].to_numpy()                 # Timestamps. Length = N

    # Smooth the position data with a low-pass filter to remove head-bob oscillations
    if _REMOVE_HEADBOB:
        dt_sample_rate = positions['unix_ms'].diff().median() / 1000.0
        fs = 1.0 / dt_sample_rate
        z_pos = lowpass(z_pos_raw, fs)
    else:
        z_pos = z_pos_raw
    positions['participant_pos_z_smooth'] = z_pos

    # Re-combine into array
    x = np.vstack((x_pos, z_pos)).T

    # Now, calculate velocity/speed and acceleration/force
    # force = acceleration magnitude. Not exactly a match, but for the sake of argument it's fine
    dx = np.diff(x, axis=0)             # Delta position. Length = N-1
    dt = np.diff(t) / 1000.0            # Delta time. Length = N-1
    v = dx / dt[:,None]                 # Velocity. Length = N-1
    speed = np.linalg.norm(v, axis=1)   # Speed. Length = N-1

    # ----------------------------
    dv = np.diff(v, axis=0)                 # Delta velocity. Length = N-2
    ddt = dt[1:]                            # Delta Delta time. Length = N-2
    a = dv / ddt[:,None]                    # Acceleration. Length = N-2
    force = np.linalg.norm(a, axis=1)       # Acceleration Magnitude. Length = N-2
    # ----------------------------
    positions['dt'] = 0.0
    positions.loc[positions.index[1:], 'dt'] = dt
    positions['speed'] = 0.0
    positions.loc[positions.index[1:], 'speed'] = speed
    positions['ddt'] = 0.0
    positions.loc[positions.index[2:], 'ddt'] = ddt
    positions["force"] = 0.0
    positions.loc[positions.index[2:], "force"] = force
    positions['force_smooth'] = positions['force'].rolling(
        window=int(6 * sigma_samples), 
        win_type='gaussian', 
        center=True, 
        min_periods=1
    ).mean(std=sigma_samples)

    # Now, we're going to identify the cause of the force change: either SPEED or DIRECTION
    eps = 1e-8
    v_hat = np.zeros_like(v)
    valid = speed > eps
    v_hat[valid] = v[valid] / speed[valid, None]    # This avoids division-by-zero when the participant is stationary.
    # Tangential acceleration magnitude (speed change)
    a_tan_mag = np.abs(np.sum(a * v_hat[1:], axis=1))
    # Normal acceleration magnitude (direction change)
    a_norm_vec = a - (np.sum(a * v_hat[1:], axis=1)[:, None] * v_hat[1:])
    a_norm_mag = np.linalg.norm(a_norm_vec, axis=1)
    # ----------------------------
    positions['speed_change'] = 0.0
    positions.loc[positions.index[2:], 'speed_change'] = a_tan_mag
    positions['speed_change_smooth'] = positions['speed_change'].rolling(
        window=int(6 * sigma_samples), 
        win_type='gaussian', 
        center=True, 
        min_periods=1
    ).mean(std=sigma_samples)
    positions['dir_change'] = 0.0
    positions.loc[positions.index[2:], 'dir_change'] = a_norm_mag
    positions['dir_change_smooth'] = positions['dir_change'].rolling(
        window=int(6 * sigma_samples), 
        win_type='gaussian', 
        center=True, 
        min_periods=1
    ).mean(std=sigma_samples)

    # Now we need to normalize all columns down to between 0 and 1
    positions['force_normalized'] = normalize_column(positions['force_smooth'])
    positions['speed_change_normalized'] = normalize_column(positions['speed_change_smooth'])
    positions['dir_change_normalized'] = normalize_column(positions['dir_change_smooth'])

    # We're going to create a new column that identifies when the `force_normalized` column is above some threshold
    # We use that transition to identify transition events.
    positions['is_transition'] = positions['force_normalized'] > 0.75
    # Group transitions based on consecutive occurrences of `is_transition`
    is_t = positions['is_transition']
    event_id = is_t.ne(is_t.shift()).cumsum()
    positions['transition_event'] = event_id.where(is_t)
    # We create labels for each transition event
    event_labels = {}
    for eid, grp in positions.groupby('transition_event'):
        if pd.isna(eid): continue
        tan_strength = grp['speed_change_normalized'].mean()
        norm_strength = grp['dir_change_normalized'].mean()
        ratio = np.log((tan_strength + 1e-6) / (norm_strength + 1e-6))
        tau = 0.4  # ≈ 50% dominance
        if ratio > tau:
            label = 'speed_change'
        elif ratio < -tau:
            label = 'direction_change'
        else:
            label = 'mixed'
        event_labels[eid] = label
    positions['transition_type'] = positions['transition_event'].map(event_labels)

    # Return
    return gazes, positions, eye, sigma_samples

def read_ped_data(src:str, participant_df:pd.DataFrame, trial_start_unix_ms):
    peds_df = pd.read_csv(src)                                                 # Read the pedestrain data
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

# Generate a distribution plot to depict the distribution curve of the speed of the participant throughout all trials
def plot_speed_distribution(par_df:pd.DataFrame, figsize=(8,5), show:bool=False):
    # Initialize figure and groups
    fig, ax = plt.subplots(figsize=figsize)
    groups = par_df['trial_id'].unique()
    colors = plt.cm.tab10.colors  # categorical color map

    # Plot based on colors and groups
    for color, g in zip(colors, groups):
        data = par_df[par_df['trial_id'] == g]['speed']
        data_no_zeros = [x for x in data if x > 0.0]
        ax.hist(
            data_no_zeros,
            bins=30,
            density=True,
            alpha=0.5,
            label=str(g),
            color=color
        )

    # Quartiles (computed over ALL data — adjust if you want per-group)
    q1 = par_df['speed'].quantile(0.25)
    q3 = par_df['speed'].quantile(0.75)
    q95 = par_df['speed'].quantile(0.95)
    ax.axvline(q1, color="black", linestyle="--", linewidth=2, label="Q1")
    ax.axvline(q3, color="black", linestyle=":", linewidth=2, label="Q3")
    ax.axvline(q95, color="black", linestyle="-", linewidth=2, label="Q0.95")

    # Labels
    ax.set_xlabel('Participant Speed')
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Distribution with Group Coloring and IQR")

    # Show or not show?
    if show: plt.show()
    else: plt.close()

# Generate a gantt chart for visualizing what a participant has looked at
def plot_gaze_gantt(gaze_dfs, details, suptitle:str, outname:str=None, show:bool=False):
    # Generate the figure
    fig, axes = plt.subplots(nrows=len(gaze_dfs), ncols=1, figsize=(12, 2*len(gaze_dfs)), sharex=True)

    # plot each df
    for i in range(len(gaze_dfs)):
        ax = axes[i]
        trial_details = details[i]
        df = gaze_dfs[i]

        # Extract targetrs
        targets = df["target"].unique()
        if 'Confederate' in targets:
            targets = [v for v in targets if v != 'Confederate'] + ['Confederate']
        target_to_y = {t: i for i, t in enumerate(targets)}

        # Extract rows inside our df
        for _, row in df.iterrows():
            ax.barh(
                y=target_to_y[row["target"]],
                width=row["duration_ms"],
                left=row["rel_start_unix_ms"]
            )
        ax.set_yticks(list(target_to_y.values()))
        ax.set_yticklabels(list(target_to_y.keys()))
        ax.set_xlabel("Relative Time (ms)")
        ax.set_ylabel("Gaze Target")
        ax.set_title(f"Trial #{trial_details['id']}: \"{trial_details['name']}\"")

    fig.suptitle(f"Gaze Behaviors: {suptitle}")
    plt.tight_layout()

    # Outputs
    if outname is not None and len(outname) > 0:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    if show: 
        plt.show()
    else: 
        plt.close()

# Generate a gantt chart for visualizing the position of viewed objects relative to the head forward direction
def plot_view_gantt(view_dfs, details, suptitle:str, outname:str=None, show:bool=False):
    # Generate the figure elements
    fig, axes = plt.subplots(nrows=len(view_dfs), ncols=1, figsize=(12, 2*len(view_dfs)), sharex=True)
    cmap = plt.cm.tab20
    sign_color_map = {
        -1: {                       # blue  (right of gaze)
            'label':'Right-side',
            'color':(0.12, 0.47, 0.71, 0.9)
        },
         1: {                       # red   (left of gaze)
            'label':'Left-side',
            'color':(0.89, 0.10, 0.11, 0.9)
         }
    }

    # plot each df
    for i in range(len(view_dfs)):
        ax = axes[i]
        trial_details = details[i]
        df = view_dfs[i]

        # Derive object ids, semgents, and other gantt chart elements
        id_label_df = (
            df[["id", "label"]]
            .drop_duplicates()
            .sort_values("id")   # or "label" if you prefer
        )
        object_ids = id_label_df["id"].to_list()
        object_labels = id_label_df["label"].to_list()
        id_color_map = { oid: cmap(i % cmap.N) for i, oid in enumerate(object_ids) }
        y_positions = { oid: i for i, oid in enumerate(object_ids) }

        # Plot gantt chart segments
        for _, row in df.iterrows():
            y = y_positions[row["id"]]
            start = row["rel_start_unix_ms"]
            duration = row["duration_ms"]
            color = sign_color_map[row["svd_sign"]]['color']
            label = row['label']
            ax.broken_barh(
                [(start, duration)],
                (y - 0.4, 0.8),
                facecolors=color,
            )
        ax.set_yticks(range(len(object_ids)))
        ax.set_yticklabels(object_labels)
        ax.set_ylabel("Object ID")
        ax.set_title(f"Trial {trial_details['id']}: {trial_details['name']}")
        ax.grid(True, axis="x")
    
    # Define the legend
    common_legend = [
        Line2D([0], [0], marker='o', color='w', label=sign_color_map[scm]['label'], markerfacecolor=sign_color_map[scm]['color'], markersize=8)
        for i, scm in enumerate(sign_color_map)
    ]
    fig.legend( handles=common_legend, loc="upper right", fontsize=7 )

    # Plotting other elements
    axes[-1].set_xlabel("Time (unix ms)")
    plt.suptitle("Object Visibility Gantt Chart (Visible + In-Front, Sign-Stable)", fontsize=14 )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if outname is not None and len(outname) > 0:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    if show: 
        plt.show()
    else: 
        plt.close()

# Generate trajectory charts for visualizing participant interactions with pedestrians
def plot_participant_trajectories(par_df, details, suptitle:str, outname:str=None, show:bool=False):
    # Color maps
    transition_type_color_map = {
        'speed_change':     (0.12, 0.47, 0.71, 1.0),  # blue
        'direction_change': (0.89, 0.10, 0.11, 1.0),  # red
        'mixed':            (0.50, 0.50, 0.50, 1.0),  # gray
    }
    highlight_styles = {
        "n_agents": {
            "mask": lambda df: df["n_agents"] > 0,
            "color": (0.12, 0.47, 0.71, 1.0),
            "title": f"Agents within R={_VIEWER_RADIUS}",
        },
        "min_distance": {
            "mask": lambda df: df["min_distance"] > -1,
            "color": (0.89, 0.10, 0.11, 1.0),
            "title": "Closest agent detected",
        },
    }

    # Initialize the figure
    fig, axes = plt.subplots( nrows=len(details), ncols=3, figsize=(12, 2 * len(details)), sharex=True, sharey=True )

    # Iterate through details to plot each
    for i, trial_details in enumerate(details):
        # get raw data
        df = par_df[par_df["trial_id"] == trial_details["id"]]
        x = df["participant_pos_x"].to_numpy()
        z = df["participant_pos_z_smooth"].to_numpy()
        start_unix_ms = trial_details['start_unix_ms']
        sample_mask = ((df["unix_ms"] - start_unix_ms) // 1000).diff().fillna(1).astype(bool)
        u = df.loc[sample_mask, "participant_for_x"].to_numpy()
        v = df.loc[sample_mask, "participant_for_z"].to_numpy()

        # --------------------------------------------------
        # Forward Direction Quiver
        # --------------------------------------------------
        norm = np.hypot(u, v)
        qx = df.loc[sample_mask, "participant_pos_x"].to_numpy()
        qz = df.loc[sample_mask, "participant_pos_z_smooth"].to_numpy()
        u /= norm
        v /= norm
                
        # --------------------------------------------------
        # Column 0: Transition types
        # --------------------------------------------------
        ax = axes[i, 0]
        colors = []
        alphas = []
        for ttype in df["transition_type"]:
            if pd.isna(ttype):
                colors.append((0.8, 0.8, 0.8, 1.0))
                alphas.append(0.05)
            else:
                colors.append(transition_type_color_map[ttype])
                alphas.append(1.0)

        ax.scatter(x, z, c=colors, s=12, alpha=alphas, zorder=2)
        ax.plot(x[0], z[0], marker="s", color="green", zorder=3)
        ax.plot(x[-1], z[-1], marker="s", color="red", zorder=3)
        ax.quiver( qx, qz, u, v, angles="xy", scale_units="xy", scale=5.0, color="black", alpha=0.6, width=0.003, zorder=4 )

        ax.set_title(f"{trial_details['name']} - Transitions")
        ax.set_ylabel("Z")
        ax.grid(True)

        # --------------------------------------------------
        # Columns 1 & 2: Agent-based metrics
        # --------------------------------------------------
        for j, (key, cfg) in enumerate(highlight_styles.items(), start=1):
            ax = axes[i, j]
            mask = cfg["mask"](df)
            ax.scatter( x, z, c=[(0.8, 0.8, 0.8, 1.0)], s=12, alpha=0.05, zorder=1 )
            ax.scatter( x[mask], z[mask], c=[cfg["color"]], s=14, alpha=1.0, zorder=2 )
            ax.plot(x[0], z[0], marker="s", color="green", zorder=3)
            ax.plot(x[-1], z[-1], marker="s", color="red", zorder=3)
            ax.quiver( qx, qz, u, v, angles="xy", scale_units="xy", scale=5.0, color="black", alpha=0.6, width=0.003, zorder=4 )
            ax.set_title(cfg["title"])
            ax.grid(True)
        # Shared limits & labels
        for j in range(3):
            axes[i, j].set_xlim(-5.5, 5.5)
            axes[i, j].set_ylim(_GANTT_YLIMS[0], _GANTT_YLIMS[1])
            axes[i, j].set_xlabel("X")

    # --------------------------------------------------
    # Legends
    # --------------------------------------------------
    transition_legend = [
        Line2D([0], [0], marker='o', color='w',
            label='Speed change',
            markerfacecolor=transition_type_color_map['speed_change'], markersize=8),
        Line2D([0], [0], marker='o', color='w',
            label='Direction change',
            markerfacecolor=transition_type_color_map['direction_change'], markersize=8),
        Line2D([0], [0], marker='o', color='w',
            label='Mixed',
            markerfacecolor=transition_type_color_map['mixed'], markersize=8),
    ]

    common_legend = [
        Line2D([0], [0], marker='o', color='w',
            label='Background',
            markerfacecolor=(0.8, 0.8, 0.8, 1.0), markersize=8),
        Line2D([0], [0], marker='s', color='w',
            label='Start',
            markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='s', color='w',
            label='End',
            markerfacecolor='red', markersize=8),
    ]

    fig.legend(
        handles=transition_legend + common_legend,
        loc="upper right",
        fontsize=7,
    )

    fig.suptitle(suptitle, fontsize=12 )
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    # Outputs
    if outname is not None and len(outname) > 0:
        plt.savefig(outname, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

# Helper: generate the gaussian used for rolling smoothing
def plot_gaussian_window(sigma_samples, outname:str=None, show=True):
    window = int(6 * sigma_samples)
    # Generate Gaussian window
    g = gaussian(window, std=sigma_samples)
    # Normalize (optional, but helpful for interpretation)
    g /= g.sum()
    x = np.arange(-window // 2, window // 2)
    plt.figure(figsize=(6, 3))
    plt.plot(x, g, marker='o')
    plt.axvline(0, linestyle='--', alpha=0.5)
    plt.title(f"Gaussian rolling window (σ = {sigma_samples} samples)")
    plt.xlabel("Sample offset")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.tight_layout()

    if outname is not None and len(outname) > 0:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    if show: plt.show()
    else: plt.close()

def extract_events(
        src_dir:str, 
        trial_filename:str='trials.csv',

        eye_filename:str='eye.csv',
        ped_filename:str='pedestrians.csv',
        show:bool=True
):
    # Read the trial dataframe, and extract the relevant trials
    trial_src = os.path.join(src_dir, trial_filename)
    assert os.path.exists(trial_src), f"Trial file \"{trial_filename}\" doesn't appear to exist..."
    trials_df = pd.read_csv(trial_src)
    print(trials_df.to_string() + "\n")

    # Initialize the caches for each major dataframe
    details = []
    gaze_dfs = []
    view_dfs = []
    par_dfs = []
    peds_dfs = []

    # Generate plots directory
    plot_outdir = os.path.join(src_dir, 'plots')
    os.makedirs(plot_outdir, exist_ok=True)

    # Iterate through eaech trial
    for index, trial in trials_df.iterrows():
        print(f"=== \033[4mProcessing Trial #{trial.trial_id}: \"{trial.trial_name}\"\033[0m ===")
        
        # Identify relevant folders in this trial directory, ignore those that do not contain the necessary files
        trial_dir = os.path.join(src_dir, str(trial.trial_id))
        if not os.path.exists(trial_dir):
            print(f"{_PRINT_INDENT}Trial directory \"{trial.trial_id}\" doesn't appear to exist... SKIPPING")
            continue
        ped_filepath = os.path.join(trial_dir, ped_filename)
        eye_filepath = os.path.join(trial_dir, eye_filename)
        if not os.path.exists(ped_filepath):
            print(f"{_PRINT_INDENT}Pedestrian file \"{ped_filename}\" not found... SKIPPING")
            continue
        if not os.path.exists(eye_filepath):
            print(f"{_PRINT_INDENT}Gaze file \"eye_filename\" not found... SKIPPING")
            continue
        print(f"{_PRINT_INDENT}Trial #{trial.trial_id} valid!")
        
        # Read participant's gaze data and positional data
        print(f"{_PRINT_INDENT}Extracting gazes and participant movements...")
        gazes_df, par_df, eye_df, sigma_samples = read_eye_data(eye_filepath, trial.start_unix_ms)

        # Read the pedestrians data
        print(f"{_PRINT_INDENT}Extracting pedestrian movements...")
        peds_df, view_df = read_ped_data(ped_filepath, par_df, trial.start_unix_ms)
        
        # Perform additional operations on participant data to calculate the following:
        # - number of agents within some radius (otherwise, 0),
        # - the distance of the closest agent, if there are any agents (otherwise, -1)
        par_df2 = interpret_distances(par_df, peds_df)

        # Append the trial ID to each of these outputs, then cache them for later
        gazes_df['trial_id'] = trial.trial_id
        view_df['trial_id'] = trial.trial_id
        par_df2['trial_id'] = trial.trial_id
        peds_df['trial_id'] = trial.trial_id
        details.append({'id':trial.trial_id,'name':trial.trial_name, 'start_unix_ms':trial.start_unix_ms})
        gaze_dfs.append(gazes_df)
        view_dfs.append(view_df)
        par_dfs.append(par_df2)
        peds_dfs.append(peds_df)

        # Plot rolling gausssian sigma samples
        plot_gaussian_window(sigma_samples, outname=os.path.join(plot_outdir, f"rolling_gaussian_trial-{trial.trial_id}.png"), show=show)
    
    # After processing, check if we have any populating each. Each must have at least one
    assert len(gaze_dfs) > 0, "No gazes processed... Invalid participant!"
    assert len(view_dfs) > 0, "No view segments processed... Invalid participant!"
    assert len(par_dfs) > 0, "No participant rows processed... Invalid participant!"
    assert len(peds_dfs) > 0, "No pedestrian. rows processed... Invalid participant!"
    gaze_df = pd.concat(gaze_dfs, ignore_index=True)
    view_df = pd.concat(view_dfs, ignore_index=True)
    par_df = pd.concat(par_dfs, ignore_index=True)
    peds_df = pd.concat(peds_dfs, ignore_index=True)

    # Plot gantt chart and participant chart
    plot_gaze_gantt(gaze_dfs, details, os.path.basename(src_dir), outname=os.path.join(plot_outdir, 'gantt-gazes.png'), show=show)
    plot_view_gantt(view_dfs, details, os.path.basename(src_dir), outname=os.path.join(plot_outdir, 'gantt-views.png'), show=show)
    plot_participant_trajectories(par_df, details, f"Participant {os.path.basename(src_dir)} Metrics", outname=os.path.join(plot_outdir, "participant_metrics.png"), show=show)

    """
    transition_type_color_map = {
        'speed_change':     (0.12, 0.47, 0.71, 1.0),  # blue
        'direction_change': (0.89, 0.10, 0.11, 1.0),  # red
        'mixed':            (0.50, 0.50, 0.50, 1.0),  # gray
    }
    fig, axes = plt.subplots(ncols=1, nrows=len(details), figsize=(7.5,10), sharex=True, sharey=True)
    for i in range(len(details)):
        trial_details = details[i]
        ax = axes[i]

        df = par_df[par_df['trial_id'] == trial_details['id']]
        x = df['participant_pos_x'].to_numpy()
        z = df['participant_pos_z'].to_numpy()
        dt_sample_rate = df['unix_ms'].diff().median() / 1000.0
        fs = 1.0 / dt_sample_rate
        z = lowpass(z, fs)

        colors = []
        alphas = []
        for ttype in df['transition_type']:
            if pd.isna(ttype):
                colors.append((0.8, 0.8, 0.8, 1.0))  # background
                alphas.append(0.05)
            else:
                colors.append(transition_type_color_map[ttype])
                #colors.append(transition_type_color_map['mixed'])
                alphas.append(1.0)

        sc = ax.scatter( x, z, c=colors, s=12, alpha=alphas)
        ax.plot(x[0], z[0], marker='s', color="green", label="Start", zorder=3)
        ax.plot(x[-1], z[-1], marker='s', color="red", label="End", zorder=3)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-2.6, -1.6)
        ax.set_title(f"Trial #{trial_details['id']}: {trial_details['name']} - Speed Changes")
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                label='Speed change',
                markerfacecolor=transition_type_color_map['speed_change'], markersize=8),
            Line2D([0], [0], marker='o', color='w',
                label='Direction change',
                markerfacecolor=transition_type_color_map['direction_change'], markersize=8),
            Line2D([0], [0], marker='o', color='w',
                label='Mixed',
                markerfacecolor=transition_type_color_map['mixed'], markersize=8),
            Line2D([0], [0], marker='s', color='w',
                label='Start',
                markerfacecolor='green', markersize=8),
            Line2D([0], [0], marker='s', color='w',
                label='End',
                markerfacecolor='red', markersize=8),
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=6)
        ax.grid(True)
    plt.suptitle(f"Participant {os.path.basename(src_dir)}: Agent Trajectory")
    plt.tight_layout()
    plt.savefig(os.path.join(src_dir, 'participant_transitions.png'), bbox_inches='tight', dpi=300)
    if show: plt.show()
    else: plt.close()

    # Now, let's generate a plot similar to participant_transitions, but instead we plot the number of neighbors within some radius of the participant
    highlight_styles = {
        "n_agents": {
            "mask": lambda df: df["n_agents"] > 0,
            "color": (0.12, 0.47, 0.71, 1.0),  # blue
            "title": "Agents within R",
        },
        "min_distance": {
            "mask": lambda df: df["min_distance"] > -1,
            "color": (0.89, 0.10, 0.11, 1.0),  # red
            "title": "Closest agent detected",
        },
    }

    fig, axes = plt.subplots(nrows=len(details), ncols=2, figsize=(7.5, 7.5), sharex=True, sharey=True)
    for i, trial_details in enumerate(details):
        df = par_df[par_df["trial_id"] == trial_details["id"]]
        x = df["participant_pos_x"].to_numpy()
        z = df["participant_pos_z"].to_numpy()

        # Optional filtering (same as your original)
        dt_sample_rate = df["unix_ms"].diff().median() / 1000.0
        fs = 1.0 / dt_sample_rate
        z = lowpass(z, fs)

        for j, (key, cfg) in enumerate(highlight_styles.items()):
            ax = axes[i, j] if len(details) > 1 else axes[j]
            mask = cfg["mask"](df)
            # Background trajectory
            ax.scatter( x, z, c=[(0.8, 0.8, 0.8, 1.0)], s=12, alpha=0.05, zorder=1 )
            # Highlighted points
            ax.scatter( x[mask], z[mask], c=[cfg["color"]], s=14, alpha=1.0, zorder=2 )
            # Start / End
            ax.plot(x[0], z[0], marker="s", color="green", zorder=3)
            ax.plot(x[-1], z[-1], marker="s", color="red", zorder=3)
            # Limits
            ax.set_xlim(-5.5, 5.5)
            ax.set_ylim(-2.6, -1.6)
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            # Title
            ax.set_title(f"{cfg['title']}: {trial_details['name']}")
            ax.grid(True)

    # Legend (shared)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
            label="Condition met",
            markerfacecolor=(0.3, 0.3, 0.3, 1.0), markersize=8),
        Line2D([0], [0], marker="o", color="w",
            label="Background",
            markerfacecolor=(0.8, 0.8, 0.8, 1.0), markersize=8),
        Line2D([0], [0], marker="s", color="w",
            label="Start",
            markerfacecolor="green", markersize=8),
        Line2D([0], [0], marker="s", color="w",
            label="End",
            markerfacecolor="red", markersize=8),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8)
    fig.suptitle(f"Participant {os.path.basename(src_dir)}: Agent-relative trajectory")
    plt.tight_layout()
    plt.savefig(os.path.join(src_dir, "participant_pedestrian_metrics.png"), bbox_inches="tight", dpi=300)
    if show: plt.show()
    else: plt.close()
    """



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("src_dir", help="Directory path to the participant's folder", type=str)
    parser.add_argument("-tf", '--trial_filename', help="Filename relative to `src_dir` of the trial csv", type=str, default='trials.csv')
    
    parser.add_argument('-ef', '--eye_filename', help="The expected filename of the eye data within the trial directory", type=str, default='eye.csv')
    parser.add_argument('-pf', '--ped_filename', help="The expected filename of the pedestrian data within the trial directory", type=str, default='pedestrians.csv')
    parser.add_argument('-s', '--show', help="Should we show any output from plot generation?", action='store_true')
    
    args = parser.parse_args()
    
    extract_events(
        args.src_dir,
        trial_filename=args.trial_filename,

        eye_filename=args.eye_filename,
        ped_filename=args.ped_filename,
        show=args.show
    )

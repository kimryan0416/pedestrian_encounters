import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.cm import get_cmap

def segments_to_intervals(
    labels, # (N,) segment ids
    t       # (N,) timestamps
):
    intervals = []
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start_idx]:
            intervals.append((
                t[start_idx],
                t[i] - t[start_idx],
                labels[start_idx]
            ))
            start_idx = i
    # last segment
    intervals.append((
        t[start_idx],
        t[-1] - t[start_idx],
        labels[start_idx]
    ))
    return intervals

def plot_segmented_trajectory(
    title:str, 
    time,           # (N-2) -> [time]
    traj,           # (N-2,3) -> [x, y, time]
    speed,          # (N,2) -> [value, time]
    force,          # (N,2) -> [value, time]
    zero_crossings, # (K,1) -> K = # of zero crossings
    accel_segments, # K-1 clusters derived from zcrossings
    dir_segments,   # D clusters,
    confederate_gaze_segments,
    neighbor_segments,
    min_distances,
    visible_segments,
    color_map='tab20',
    add_segment_labels_to_legend:bool=True
):
    # Define the figure
    fig, axs = plt.subplots(4, 1, figsize=(8, 8))

    # Assign colors to segments
    n_accel_segments = accel_segments.max() + 1
    n_neighbor_segments = neighbor_segments.max() + 1
    accel_cmap = get_cmap(color_map, n_accel_segments)

    # ---- 1st PLOT: trajectory ----
    for seg_id in range(n_accel_segments):
        mask = accel_segments == seg_id
        label = f"Speed Segment {seg_id}" if add_segment_labels_to_legend else None
        axs[0].plot(
            traj[mask, 0],
            traj[mask, 1],
            color=accel_cmap(seg_id),
            linewidth=2,
            label=label
        )
    axs[0].scatter(traj[0, 0], traj[0, 1], c="green", marker='o', s=30, label="Start", zorder=3)
    axs[0].scatter(traj[-1, 0], traj[-1, 1], c="red", marker='x', s=30, label="End", zorder=3)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title("2D Trajectory Segmented by Zero-Crossings")
    axs[0].legend(ncol=2, fontsize=8)

    # ---- 2nd PLOT: Direction ----
    n_dir_segments = dir_segments.max() + 1
    dir_cmap = get_cmap(color_map, n_dir_segments)
    speed_norm = speed / (speed.max() + 1e-12)
    marker_sizes = 0.1 + 20 * np.sqrt(speed_norm)
    alpha_min, alpha_max = 0.25, 0.9
    marker_alpha = 1 - (alpha_min + (alpha_max - alpha_min) * np.sqrt(speed_norm))
    for seg in range(n_dir_segments):
        mask = dir_segments == seg
        label = f"Turn segment {seg}" if add_segment_labels_to_legend else None
        # base trajectory line
        axs[1].plot(
            traj[mask, 0],
            traj[mask, 1],
            color=dir_cmap(seg),
            linewidth=2,
            label=label,
            zorder=1
        )
        """
        axs[1].scatter(
            traj[mask, 0],
            traj[mask, 1],
            s=marker_sizes[mask],
            color=dir_cmap(seg),
            alpha=marker_alpha[mask],
            linewidths=0,
            zorder=2
        )
        """
    axs[1].scatter(traj[0, 0], traj[0, 1], c="green", marker='o', s=30, label="Start", zorder=3)
    axs[1].scatter(traj[-1, 0], traj[-1, 1], c="red", marker='x', s=30, label="End", zorder=3)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_title("Trajectory Segmented by Directional Change")
    axs[1].legend(ncol=2, fontsize=8)

    # ---- 3rd PLOT: speed + force ----
    #axs[2].plot(speed[:,1], speed[:,0], c='orange', label='speed', zorder=2)
    count = 0
    for seg_id in range(n_accel_segments):
        mask = accel_segments == seg_id
        if not np.any(mask): continue
        count += 1
        axs[2].plot(
            time[mask],
            speed[mask],
            color=accel_cmap(seg_id),
            linewidth=2,
            zorder=3,
            label=f"Speed" if count == 1 else None
        )
    axs[2].plot(time, force, c='blue', alpha=0.25, label='force', zorder=1)
    axs[2].axhline(0, c='black', alpha=0.7)
    for i, zc in enumerate(zero_crossings):
        axs[2].axvline(zc, color='gray', alpha=0.6)
    axs[2].set_ylabel("Magnitude")
    axs[2].set_xlabel("Time")
    axs[2].set_ylim(-1,2)
    axs[2].legend(fontsize=8)
    axs[2].set_title("Speed / Force with Zero-Crossings")

    # ---- 4th PLOT: compressed Gantt-style segmentation ----
    accel_intervals = segments_to_intervals(accel_segments, time)
    dir_intervals   = segments_to_intervals(dir_segments, time)
    # Vertical layout
    row_height = 0.4
    visible_y0 = 0.1
    visible_y1 = visible_y0 + row_height
    min_distance_y0 = 0.6
    min_distance_y1 = min_distance_y0 + row_height
    gaze_y = (1.1, row_height)
    dir_y   = (1.6, row_height)
    accel_y0  = 2.1
    accel_y1  = accel_y0 + row_height
    v_max = np.max(speed)
    v_norm = speed / v_max if v_max > 0 else speed
    d_max = np.max(min_distances)
    d_norm = min_distances / d_max if d_max > 0 else min_distances
    vis_max = np.max(visible_segments)
    vis_norm = visible_segments / vis_max if vis_max > 0 else visible_segments
    # Acceleration segments
    for seg_id in range(n_accel_segments):
        mask = accel_segments == seg_id
        if not np.any(mask): continue
        t_seg = time[mask]
        v_seg = v_norm[mask]
        axs[3].fill_between(
            t_seg,
            accel_y0,
            accel_y0 + v_seg * row_height,
            color=accel_cmap(seg_id),
            alpha=0.85,
            linewidth=0
        )
        """
        for t0, dt, seg_id in accel_intervals:
            axs[3].broken_barh(
                [(t0, dt)],
                accel_y,
                facecolors=accel_cmap(seg_id),
                alpha=0.9
            )
        """
    # Direction segments
    for t0, dt, seg_id in dir_intervals:
        axs[3].broken_barh(
            [(t0, dt)],
            dir_y,
            facecolors=dir_cmap(seg_id),
            alpha=0.9
        )
    # Gaze targets
    for t0, dt in confederate_gaze_segments:
        axs[3].broken_barh(
            [(t0, dt)],
            gaze_y,
            facecolors='red',
            alpha=0.9
        )
    # Min Distances
    for seg_id in np.unique(neighbor_segments):
        if seg_id == 0: continue
        mask = neighbor_segments == seg_id
        if not np.any(mask): continue
        t_seg = time[mask]
        d_seg = d_norm[mask]
        axs[3].fill_between(
            t_seg,
            min_distance_y0,
            min_distance_y0 + d_seg * row_height,
            color=accel_cmap(seg_id),
            alpha=0.85,
            linewidth=0
        )
    # Visible, same-side pedestrians
    for seg_id in np.unique(visible_segments):
        print(seg_id)
        if seg_id == 0: continue
        mask = visible_segments == seg_id
        if not np.any(mask): continue
        t_seg = time[mask]
        vis_seg = vis_norm[mask]
        axs[3].fill_between(
            t_seg,
            visible_y0,
            visible_y0 + vis_seg * row_height,
            color=accel_cmap(seg_id),
            alpha=0.85,
            linewidth=0
        )
    # Formatting
    axs[3].set_ylim(0, 3.2)
    axs[3].set_yticks([0.25, 0.75, 1.25, 1.75, 2.25])
    axs[3].set_yticklabels(["Visible, SS Peds", "Neighbors Within R=2", "Confederate Gazes", "Direction", "Acceleration"])
    axs[3].set_xlabel("Time")
    axs[3].set_title("Segment Timeline Overview")
    axs[3].grid(True, axis="x", alpha=0.3)

    # Display
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_speed_acceleration(
    speed, 
    force, 
    zero_crossings=None, 
    outname:str=None
):
    fig = plt.figure(figsize=(10,5))
    plt.plot(speed[:,1], speed[:,0], c='orange', zorder=2, label='speed')
    plt.plot(force[:,1], force[:,0], c='blue', zorder=1, label='acceleration')
    plt.axhline(y=0, c='black', alpha=0.75)
    if zero_crossings is not None:
        for zero_crossing in zero_crossings:
            plt.axvline(x=zero_crossing, c='gray')
    plt.legend()
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    plt.show()


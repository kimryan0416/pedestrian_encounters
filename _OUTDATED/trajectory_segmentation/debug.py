import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from collections import defaultdict

def compute_windowed_dbscan(v, window_size, eps, step=None, min_samples=5):
    if step is None:
        step = window_size
    results = []
    for start in range(0, len(v) - window_size + 1, step):
        end = start + window_size
        idx = np.arange(start, end)
        labels = DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit_predict(v[idx])
        results.append({
            "start": start,
            "end": end,
            "indices": idx,
            "labels": labels
        })
    return results

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0

def compute_associations(windows):
    associations = {}

    for i in range(len(windows) - 1):
        wA, wB = windows[i], windows[i + 1]

        for lblA in np.unique(wA["labels"]):
            if lblA == -1:
                continue
            idxA = wA["indices"][wA["labels"] == lblA]

            for lblB in np.unique(wB["labels"]):
                if lblB == -1:
                    continue
                idxB = wB["indices"][wB["labels"] == lblB]

                s = jaccard(idxA, idxB)
                if s > 0:
                    associations[(i, lblA, lblB)] = s

    return associations

def coassociation_strength(same, total, threshold=0.6):
    edges = []
    for key in total:
        if same[key] / total[key] >= threshold:
            edges.append(key)
    return edges

def extract_segments(edges, N):
    segments = []
    current = [0]

    edge_set = set(edges)

    for i in range(N - 1):
        if (i, i + 1) in edge_set or (i + 1, i) in edge_set:
            current.append(i + 1)
        else:
            segments.append(current)
            current = [i + 1]

    segments.append(current)
    return segments

def run_dbscan(v, eps, min_samples=5):
    if len(v) == 0:
        return np.array([])
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(v)

def cluster_strength(idx_a, idx_b):
    a = set(idx_a)
    b = set(idx_b)
    return len(a & b) / len(a | b) if a | b else 0.0

def compute_windowed_dbscan_time(v, t, window_ms, eps, min_samples=5):
    windows = []

    time_windows = compute_time_windows(t, window_ms)

    for (start, end) in time_windows:
        idx = np.arange(start, end)

        labels = DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit_predict(v[idx])

        windows.append({
            "indices": idx,
            "labels": labels,
            "t_start": t[start],
            "t_end": t[end - 1]
        })

    return windows

def compute_time_windows(t, window_ms, step_ms=None):
    """
    Returns a list of (start_idx, end_idx) pairs such that
    t[start_idx:end_idx] lies inside a time window.
    """
    if step_ms is None:
        step_ms = window_ms

    t0 = t[0]
    t_end = t[-1]

    windows = []
    current_start = t0

    while current_start + window_ms <= t_end:
        start_idx = np.searchsorted(t, current_start, side="left")
        end_idx = np.searchsorted(t, current_start + window_ms, side="right")

        if end_idx > start_idx:
            windows.append((start_idx, end_idx))

        current_start += step_ms

    return windows

# Calculate the signed turning angles across a velocity window
def calculate_turning_angles(v, eps=1e-6):
    speed = np.linalg.norm(v, axis=1)
    valid = speed > eps

    v_hat = np.zeros_like(v)
    v_hat[valid] = v[valid] / speed[valid, None]

    dot = np.sum(v_hat[1:] * v_hat[:-1], axis=1)
    dot = np.clip(dot, -1.0, 1.0)

    cross = (
        v_hat[:-1, 0] * v_hat[1:, 1]
        - v_hat[:-1, 1] * v_hat[1:, 0]
    )

    angle = np.arctan2(cross, dot)
    angle[~(valid[1:] & valid[:-1])] = 0.0
    return angle

# Calculate the curvature energy based on turn angles and windows of time
def calculate_curvature_energy(t, turn_angle, window_ms=500):
    energy = np.zeros(len(t))
    start = 0
    for i in range(1, len(t)):
        while t[start + 1] < t[i] - window_ms:
            start += 1
        energy[i] = np.sum(np.abs(turn_angle[start:i]))
    return energy

# Same as `calculate_curvature_energy`, but this time comparing both small and large time windows
def calculate_curvature_energy_multiscale(t, turn_angle, windows_ms=(300, 1200), weights=(0.3, 0.7)):
    """
    Multi-scale curvature energy to capture both sharp and gradual turns.

    Parameters
    ----------
    t : (M,) array
        Timestamps (ms), aligned with turn_angle
    turn_angle : (M,) or (M-1,) array
        Signed turning angles
    windows_ms : tuple
        Window sizes in milliseconds
    weights : tuple
        Weight for each window (same length as windows_ms)

    Returns
    -------
    energy : (M,) array
        Combined curvature energy
    """
    assert len(windows_ms) == len(weights)

    energies = []
    for w in windows_ms:
        e = calculate_curvature_energy(t, turn_angle, window_ms=w)
        energies.append(e)

    energy = np.zeros_like(energies[0])
    for w, e in zip(weights, energies):
        energy += w * e

    return energy

def segments_to_labels(n, segments):
    labels = -1 * np.ones(n, dtype=int)
    for i, idx in enumerate(segments):
        labels[np.asarray(idx)] = i
    return labels

def plot_velocity_over_windows(v, t, window_size=1000, segments=None, color_map='Dark2'):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    sc = ax.scatter([], [], s=20)
    ax.set_xlabel("v_x")
    ax.set_ylabel("v_y")
    ax.set_aspect("equal")
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    ax_time = plt.axes([0.15, 0.18, 0.7, 0.03])
    time_slider = Slider(
        ax_time, "Center time (s)", t[0], t[-1], valinit=t[0]
    )

    ax_window = plt.axes([0.15, 0.12, 0.7, 0.03])
    window_slider = Slider(
        ax_window, "Window size (ms)", 100, t[-1] - t[0], valinit=window_size
    )

    # ---- segment handling ----
    if segments is not None:
        labels = -1 * np.ones(len(t), dtype=int)
        for i, idx in enumerate(segments):
            labels[np.asarray(idx)] = i

        cmap = get_cmap(color_map)
        n_segments = len(segments)
    else:
        labels = None
        
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

        sigma = W / 6
        alpha = gaussian_alpha(t_win, t0, sigma)

        sc.set_offsets(v_win)

        colors = np.zeros((len(v_win), 4))

        if labels is None:
            # Default color
            colors[:, :3] = [0.2, 0.4, 0.8]
        else:
            lbl_win = labels[mask]
            for i, lbl in enumerate(lbl_win):
                if lbl >= 0:
                    colors[i, :3] = cmap(lbl % cmap.N)[:3]
                else:
                    colors[i, :3] = [0.6, 0.6, 0.6]  # noise / unassigned

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

# Calcualte segments from curvature energy
def segment_by_curvature_energy(t, curvature_energy, threshold="auto", min_duration=10):
    """
    Segment trajectory based on directional change.

    Parameters
    ----------
    t : (N,) array
        Time values
    curvature_energy : (N-1,) array
        Windowed curvature energy
    threshold : float or "auto"
        Energy threshold
    min_duration : int
        Minimum segment length in samples

    Returns
    -------
    segments : (N,) int
        Segment label per sample
    """
    energy = np.asarray(curvature_energy)

    if threshold == "auto":
        threshold = np.mean(energy) + 1.0 * np.std(energy)

    active = energy > threshold

    segments = np.zeros(len(t), dtype=int)
    seg_id = 0
    count = 0

    for i in range(1, len(t)):
        segments[i] = seg_id
        if active[i] and not active[i-1]:
            if count >= min_duration:
                seg_id += 1
                count = 0
        count += 1

    return segments

def plot_dbscan_windows_over_time(v, t):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.35, wspace=0.25)

    for ax, title in zip((axA, axB), ("Window A", "Window B")):
        ax.set_title(title)
        ax.set_xlabel("v_x")
        ax.set_ylabel("v_y")
        ax.set_aspect("equal")
        ax.axhline(0, lw=0.5)
        ax.axvline(0, lw=0.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    scA = axA.scatter([], [], s=30)
    scB = axB.scatter([], [], s=30)

    info = fig.text(
        0.5, 0.97, "",
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    # ---- sliders ----
    ax_idx = plt.axes([0.15, 0.25, 0.7, 0.03])
    idx_slider = Slider(ax_idx, "Window index", 0, len(t)-1, valinit=0, valstep=1)

    ax_win = plt.axes([0.15, 0.20, 0.7, 0.03])
    win_slider = Slider(ax_win, "Window size", 20, 500, valinit=100, valstep=10)

    ax_eps = plt.axes([0.15, 0.15, 0.7, 0.03])
    eps_slider = Slider(ax_eps, "DBSCAN eps", 0.05, 1.0, valinit=0.3)

    cmap = get_cmap("tab10")

    state = {}

    def colorize(labels):
        colors = np.zeros((len(labels), 4))
        for i, lbl in enumerate(labels):
            if lbl == -1:
                colors[i] = [0.6, 0.6, 0.6, 0.4]
            else:
                colors[i] = cmap(lbl % cmap.N)
                colors[i, 3] = 0.9
        return colors

    def update(_):
        i = int(idx_slider.val)
        W = int(win_slider.val)
        eps = eps_slider.val

        idxA = np.arange(i, min(i + W, len(v)))
        idxB = np.arange(i + W, min(i + 2 * W, len(v)))

        vA, vB = v[idxA], v[idxB]

        labelsA = run_dbscan(vA, eps)
        labelsB = run_dbscan(vB, eps)

        state["A"] = (idxA, labelsA)
        state["B"] = (idxB, labelsB)

        scA.set_offsets(vA)
        scA.set_facecolors(colorize(labelsA))

        scB.set_offsets(vB)
        scB.set_facecolors(colorize(labelsB))

        info.set_text(
            f"Window A: [{i}:{i+W})   "
            f"Window B: [{i+W}:{i+2*W})   "
            f"eps={eps:.2f}"
        )

        fig.canvas.draw_idle()

    # ---- hover logic ----
    def cluster_strength(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a | b else 0.0

    def on_move(event):
        for name, sc in zip(("A", "B"), (scA, scB)):
            if event.inaxes is sc.axes:
                cont, ind = sc.contains(event)
                if not cont:
                    continue

                idxs, labels = state[name]
                lbl = labels[ind["ind"][0]]
                if lbl == -1:
                    return

                other = "B" if name == "A" else "A"
                idxs2, labels2 = state[other]

                members1 = idxs[labels == lbl]

                msg = f"Cluster {lbl} ({name})\n"
                for lbl2 in np.unique(labels2):
                    if lbl2 == -1:
                        continue
                    members2 = idxs2[labels2 == lbl2]
                    s = cluster_strength(members1, members2)
                    if s > 0:
                        msg += f"↔ {lbl2} ({other}): {s:.2f}\n"

                info.set_text(msg)
                fig.canvas.draw_idle()
                return

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    idx_slider.on_changed(update)
    win_slider.on_changed(update)
    eps_slider.on_changed(update)

    update(None)
    plt.show()

def windowed_dbscan_inspector(v, t):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.35)

    scA = axA.scatter([], [], s=30)
    scB = axB.scatter([], [], s=30)

    info = fig.text(
        0.5, 0.95, "",
        ha="center",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    for ax in (axA, axB):
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(0, lw=0.5)
        ax.axvline(0, lw=0.5)

    # ---- sliders ----
    ax_idx = plt.axes([0.15, 0.25, 0.7, 0.03])
    idx_slider = Slider(ax_idx, "Window index", 0, 1, valinit=0, valstep=1)

    ax_win = plt.axes([0.15, 0.20, 0.7, 0.03])
    win_slider = Slider(
        ax_win,
        "Window size (ms)",
        100,
        t[-1] - t[0],
        valinit=1000,
        valstep=100
    )

    ax_eps = plt.axes([0.15, 0.15, 0.7, 0.03])
    eps_slider = Slider(ax_eps, "DBSCAN eps", 0.05, 1.0, valinit=0.3)

    cmap = plt.get_cmap("tab10")

    state = {
        "windows": None,
        "associations": None
    }

    def recompute(_):
        window_ms = win_slider.val
        eps = eps_slider.val

        windows = compute_windowed_dbscan_time(
            v, t, window_ms, eps
        )

        associations = compute_associations(windows)

        state["windows"] = windows
        state["associations"] = associations

        max_idx = max(0, len(windows) - 2)
        idx_slider.ax.set_xlim(0, max_idx)   # THIS is the key line
        idx_slider.valmin = 0
        idx_slider.valmax = max_idx
        idx_slider.valstep = 1
        idx_slider.set_val(0)

        draw(0)

    def colorize(labels):
        colors = np.zeros((len(labels), 4))
        for i, lbl in enumerate(labels):
            if lbl == -1:
                colors[i] = [0.6, 0.6, 0.6, 0.4]
            else:
                colors[i] = cmap(lbl % cmap.N)
                colors[i, 3] = 0.9
        return colors
    
    def draw(i):
        windows = state["windows"]
        if windows is None:
            return

        i = int(i)
        wA, wB = windows[i], windows[i + 1]

        scA.set_offsets(v[wA["indices"]])
        scA.set_facecolors(colorize(wA["labels"]))
        axA.set_title(f"Window {i}")

        scB.set_offsets(v[wB["indices"]])
        scB.set_facecolors(colorize(wB["labels"]))
        axB.set_title(f"Window {i+1}")

        info.set_text(
            f"Showing windows {i} → {i+1}"
        )

        fig.canvas.draw_idle()
    
    def on_move(event):
        windows = state["windows"]
        associations = state["associations"]

        if windows is None or associations is None:
            return

        # Current window index from slider
        i = int(idx_slider.val)

        for name, ax, sc in (
            ("A", axA, scA),
            ("B", axB, scB),
        ):
            if event.inaxes is not ax:
                continue

            contains, info_dict = sc.contains(event)
            if not contains:
                return

            ind = info_dict["ind"][0]

            w = windows[i] if name == "A" else windows[i + 1]
            lbl = w["labels"][ind]

            if lbl == -1:
                info.set_text(f"Noise point ({name})")
                fig.canvas.draw_idle()
                return

            msg = f"Window {i if name == 'A' else i+1}\n"
            msg += f"Cluster {lbl}\n"

            if name == "A":
                # look forward
                for lbl2 in np.unique(windows[i + 1]["labels"]):
                    if lbl2 == -1:
                        continue
                    s = associations.get((i, lbl, lbl2), 0.0)
                    if s > 0:
                        msg += f"→ Cluster {lbl2}: {s:.2f}\n"
            else:
                # look backward
                for lbl2 in np.unique(windows[i]["labels"]):
                    if lbl2 == -1:
                        continue
                    s = associations.get((i, lbl2, lbl), 0.0)
                    if s > 0:
                        msg += f"← Cluster {lbl2}: {s:.2f}\n"

            info.set_text(msg)
            fig.canvas.draw_idle()
            return

    # EXPENSIVE → recompute
    win_slider.on_changed(recompute)
    eps_slider.on_changed(recompute)

    # CHEAP → view only
    idx_slider.on_changed(draw)

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    recompute(None)
    plt.show()

def identify_accel_threshold(force, accel_segments):
    # `accel_segments` is just a an array of N-2 length. It maps directly to `force_time` in terms of indices.
    # Therefore, in order to describe these segments, we need to do the following:
    #   1. Determine the sign of each element in `accel_segment`. This can be done by majority vote of the sign of each accel row
    #   2. For each accel segment, determine the mean (or median?) acceleration value 
    force_min = np.min(force)
    force_max = np.max(force)
    force_normalized = (force - force_min) / (force_max - force_min)
    force_segments = np.column_stack([force_normalized, accel_segments])
    n_accel_segments = accel_segments.max() + 1   # number of groups
    sums = np.zeros(n_accel_segments)
    counts = np.zeros(n_accel_segments)
    np.add.at(sums, accel_segments, force)
    np.add.at(counts, accel_segments, 1)
    mean_accel_segments = sums / counts[:, None]

    thresholds = np.linspace(0, 1, 1000)
    accel_means = np.abs(mean_accel_segments[:,0])
    means = accel_means[:,None]
    ths = thresholds[:,None]
    above = means > thresholds
    below = means <= thresholds
    count_above = above.sum(axis=0)
    count_below = below.sum(axis=0)
    ratio = count_above / count_below

    fig, ax = plt.subplots(figsize=(7, 4))
    line, = ax.plot(thresholds, ratio, marker='o', markersize=1)
    plt.axhline(0, linestyle='--', alpha=0.5)
    plt.xlabel("Threshold")
    plt.ylabel("Above vs Below imbalance")
    plt.title("Threshold sweep")
    cursor = mplcursors.cursor(line, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        sel.annotation.set_text(f"Threshold = {x:.3f}\nImbalance = {y:.3f}")
    plt.tight_layout()
    plt.show()








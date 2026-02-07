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
import numpy as np
import ruptures as rpt
import pandas as pd
#from collections import defaultdict

def assign_segments_from_zero_crossings(t, zero_crossings):
    zero_crossings = np.asarray(zero_crossings)
    segments = np.zeros(len(t), dtype=int)
    for i, zc in enumerate(zero_crossings):
        segments[t >= zc] += 1
    return segments

# Segment trajectory based on changes in heading statistics.
# heading : (N,) array
# penalty : ?
def assign_segments_by_heading_changepoints( heading, penalty=10 ):
    signal = heading.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    cps = algo.predict(pen=penalty)
    segments = np.zeros(len(heading), dtype=int)
    seg_id = 0
    last = 0
    for cp in cps:
        segments[last:cp] = seg_id
        seg_id += 1
        last = cp
    return segments, cps

def assign_segments_from_values(t, values):
    segments = []
    start_idx = 0
    for i in range(1, len(values)):
        if values[i] != values[start_idx]:
            segments.append({
                "start_t": t[start_idx],
                "end_t": t[i - 1],
                "value": values[start_idx],
                "mask": slice(start_idx, i)
            })
            start_idx = i
    # last segment
    segments.append({
        "start_t": t[start_idx],
        "end_t": t[-1],
        "value": values[start_idx],
        "mask": slice(start_idx, len(values))
    })

    return segments

def assign_indices_from_timestamp_intervals(
    t, 
    interval_df:pd.DataFrame, 
    ts_index_mapper=None,
    start_ts_colname:str='start_unix_ms', 
    end_ts_colname:str='end_unix_ms'
):
    indices = np.zeros(len(t))
    for _, row in interval_df.iterrows():
        start_index = row[start_ts_colname]
        end_index = row[end_ts_colname]
        if ts_index_mapper is not None:
            start_index = ts_index_mapper(start_index)
            end_index = ts_index_mapper(end_index)
        for i in range(start_index, end_index+1):
            indices[i] += 1
    return indices

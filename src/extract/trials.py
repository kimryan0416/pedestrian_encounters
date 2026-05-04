""" 
=================================
Description
=================================
This file focuses on extracting trial data from participants' directories
"""

# =================================
# Imports
# =================================

import pandas as pd


# =================================
# Extracting trial data
# =================================

def data_from_df(
        trial_df:pd.DataFrame, 
        trial_id:int
):
    trial_row = trial_df[trial_df['trial_id'] == trial_id].iloc[0]
    trial_name = trial_row['trial_name']
    start_ms = trial_row['sim_unix_ms']
    end_ms = trial_row['end_unix_ms']
    return trial_name, start_ms, end_ms

def data(
        trial_src:str,
        trial_id:int
):
    trial_df = pd.read_csv(trial_src)
    return extract_trial_data_from_df(trial_df, trial_id)
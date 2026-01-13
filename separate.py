import os
import argparse
import pandas as pd

from RecordMuse.analysis import psd 

_DESCRIPTION =  "After calculating offsets, separate the EEG, eye, and pedestrian data into " + \
                "the individual trials. We also separate them based on whether they're part " + \
                "of the calibration stage or actual simulation."


def separate_files(
        src_dir:str, 
        offsets_src:str, 
        trials_src:str, 
        eeg_src:str, 
        eye_src:str, 
        ped_src:str,
        ts_col:str='unix_ms'    ):
    
    # Get the offsets and trials, validate that their folders and the calibration files inside are valid
    offsets_df = pd.read_csv(os.path.join(src_dir, offsets_src))
    
    # Read the other files
    eeg_df = pd.read_csv(os.path.join(src_dir, eeg_src))
    eye_df = pd.read_csv(os.path.join(src_dir, eye_src))
    ped_df = pd.read_csv(os.path.join(src_dir, ped_src))

    # Apply offset calculation
    # new timestamp calculation: 
    #   - `offfset_eeg-gaze` was calculated by subtracting the EEG fro mthe EYE timestamps
    #   - This means that these offset values represent how much ahead the EEG data is
    #       - E.G. If the offset value is +, then it means that the EEG timestamps are ahead of the EYE timestamps
    #       - E.G. if the offset value is -, then the EEG timestamps are earlier than the EYE timestamps
    #   - Therefore, to apply to the EEG data, we:
    #       1. Copy the original `ts_col` timestamp into a copied column
    #       2. Apply a simple 
    offset_median = offsets_df['offset_eeg-gaze'].median()
    eeg_df = eeg_df.copy()
    eeg_df[f"{ts_col}_original"] = eeg_df[ts_col]
    eeg_df[ts_col] = eeg_df[ts_col].apply(lambda x: x - offset_median)
    eeg_outpath = os.path.join(src_dir, 'eeg_corrected.csv')
    eeg_df.to_csv(eeg_outpath, index=False)
    print("Corrected EEG data saved to:", eeg_outpath)
    
    # Calculate bandpowers given the EEG data
    bandpowers_src = psd.calculate_psd(eeg_outpath)
    bandpowers_df = pd.read_csv(bandpowers_src)
    
    # With the bandpowers now saved, we can begin to slice each of the following per trial:
    # - EEG
    # - Bandpowers
    # - Pedestrians
    # - Eye
    trials_df = pd.read_csv(os.path.join(src_dir, trials_src))
    for _, trial in trials_df.iterrows():
        tdir = os.path.join(src_dir, str(trial['trial_id']))
        assert os.path.exists(tdir), f"Directory for trial {trial['trial_id']} doesn't exist..."
        assert os.path.exists(os.path.join(tdir, 'calibration.csv')), f"Calibration file for trial {trial['trial_id']} not found..."

        # Derive the timestamps for the entire trial.
        # Note that the trial data contains the timestamps starting from AFTER the calibration.
        # In other words, no need to seek out the calibration data again
        start_ts, end_ts = int(trial['start_unix_ms']), int(trial['end_unix_ms'])
        start_frame, end_frame = int(trial['start_frame']), int(trial['end_frame'])
        trial_eeg = eeg_df[eeg_df[ts_col].between(start_ts, end_ts)]
        trial_bandpowers = bandpowers_df[bandpowers_df[ts_col].between(start_ts, end_ts)]
        trial_eye = eye_df[eye_df[ts_col].between(start_ts, end_ts)]
        trial_ped = ped_df[ped_df['frame'].between(start_frame, end_frame)]
        
        # Save them insdide `tdir`
        trial_eeg.to_csv(os.path.join(tdir, 'eeg.csv'), index=False)
        trial_bandpowers.to_csv(os.path.join(tdir, 'bandpowers.csv'), index=False)
        trial_eye.to_csv(os.path.join(tdir, 'eye.csv'), index=False)
        trial_ped.to_csv(os.path.join(tdir, 'pedestrians.csv'), index=False)
        print(f"Trial {trial['trial_id']} extracted!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument('src_dir', help="The directory to the participant's data", type=str)
    parser.add_argument('-o', '--offsets', help="The filename relative to `src_dir` defining the offsets csv", type=str, default='offsets.csv')
    parser.add_argument('-t', '--trials', help="The filename relative to `src_dir` defining the trials csv", type=str, default='trials.csv')
    parser.add_argument('-ee', '--eeg', help="The filename relative to `src_dir` defining the EEG csv", type=str, default='eeg-vr/EEG_filtered_normalized.csv')
    parser.add_argument('-ey', '--eye', help="The filename relative to `src_dir` defining the eye csv", type=str, default='eye.csv')
    parser.add_argument('-p', '--ped', help="The filename relative to `src_dir` defining the pedestrians csv", type=str, default='pedestrians.csv')
    parser.add_argument('-tc', '--timestamp_column', help="The timestamp column across all data", type=str, default='unix_ms')
    args = parser.parse_args()
    separate_files(
        args.src_dir,
        args.offsets,
        args.trials,
        args.eeg,
        args.eye,
        args.ped,
        args.timestamp_column
    )
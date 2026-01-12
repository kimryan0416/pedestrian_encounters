import os
import shutil
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt

_TRIAL_NAMES = [
    "Trial-ApproachAudio Start", 
    "Trial-BehindAudio Start", 
    "Trial-Behind Start", 
    "Trial-AlleyRunnerAudio Start", 
    "Trial-AlleyRunner Start", 
    "Trial-Approach Start"
]
_CALIBRATION_COLUMNS = [
    'unix_ms', 
    'frame', 
    'rel_timestamp',
    'event', 
    'overlap_counter'
]
_CALIBRATION_FILES = [
    'calibration_test_1.csv',
    'calibration_test_2.csv',
    'calibration_test_3.csv',
    'calibration_test_4.csv',
    'calibration_test_5.csv',
    'calibration_test_6.csv'
]

def find_files_by_pattern(src_dir:str, pattern:str="*.csv"):
    return glob.glob(os.path.join(src_dir, pattern))

def read_calibration_file(_F:str, correction:bool=True):
    df = pd.read_csv(_F)                            # Read the file
    df = df.iloc[:, :len(_CALIBRATION_COLUMNS)]     # Correction
    df.columns = _CALIBRATION_COLUMNS
    return df               

def mkdirs(_DIR:str, delete_existing:bool=True):    
    if delete_existing and os.path.exists(_DIR):    # If the folder already exists, delete it
        shutil.rmtree(_DIR)
    os.makedirs(_DIR, exist_ok=True)                # Create a new empty directory
    return _DIR                                     # Return the directory

def plot_calibrations(x, start_time, end_time, xlabel:str, cdfs):
    # --- Collect first timestamps from calibration files ---
    markers = []
    for f, df in cdfs:
        if xlabel not in df.columns:    continue  # skip if missing
        first_time = df[xlabel].iloc[0]
        markers.append((f"{os.path.basename(f)}\n{first_time}", first_time))

    # --- Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(x, [0]*len(x), alpha=0.2, label="Eye dataset timeline")

    # Add vertical markers for each smaller file
    for name, t in markers:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.7)
        plt.text(t, 0.1, name, rotation=90, verticalalignment='bottom', fontsize=8)

    # Add a vertical marker for the start and ends too.
    plt.axvline(x=start_time, color='blue', linestyle='--', alpha=0.7)
    plt.text(start_time, 0.1, f"Eye Start\n{start_time}", rotation=90, verticalalignment='bottom', fontsize=7.5)

    # Render other stuff
    plt.xlim(start_time, end_time)
    plt.xlabel(xlabel)
    plt.yticks([])
    plt.title(f"Timeline of Eye Data to Calibrations:")
    plt.legend()
    plt.tight_layout()
    plt.show()

def identify_trials(
        src_dir, 
        eye_src:str, 
        #ped_src:str, 
        #eeg_src:str, 
        #bp_src:str, 
        ts_col:str='unix_ms'):

    # Read the eye, pedestrian, eeg, and bandpower data
    eye = pd.read_csv(os.path.join(src_dir, eye_src))
    #pedestrians = pd.read_csv(os.path.join(vr_dir, ped_src))
    #eeg = pd.read_csv(os.path.join(eeg_dr, eeg_src))
    #bandpowers = pd.read_csv(os.path.join(eeg_dr, bp_src))
    assert ts_col in eye.columns, f"The expected timestamp column \"{ts_col}\" is not present in \"{eye_src}\""
    #assert ts_col in pedestrians.columns, f"The expected timestamp column \"{ts_col}\" is not present in \"{ped_src}\""
    #assert ts_col in eeg.columns, f"The expected timestamp column \"{ts_col}\" is not present in \"{eeg_src}\""
    #assert ts_col in bandpowers.columns, f"The expected timestamp \"{ts_col}\" is not present in \"{bp_src}\""

    # Get the start and end times of this trial
    start_time = eye[ts_col].min()
    end_time = eye[ts_col].max()

    # Find and plot calibration files
    calibrations = [(f, read_calibration_file(f)) for f in find_files_by_pattern(src_dir, "calibration_*.csv")]
    plot_calibrations(eye[ts_col], start_time, end_time, ts_col, calibrations)

    # Get trials, derive end unix ms and frame from calibration_test_7.csv
    trials = eye[eye['event'].isin(_TRIAL_NAMES)]
    trials = trials[['event', 'unix_ms', 'frame']].rename(columns={'event':'trial_name', 'unix_ms':'start_unix_ms', 'frame':'start_frame'})
    trials = trials.sort_values('start_unix_ms')
    trials['end_unix_ms'] = trials['start_unix_ms'].shift(-1)
    trials['end_frame'] = trials['start_frame'].shift(-1)
    # To get the last entries of `end_unix_ms` and `end_frame`, we have to look at `calibration_7.csv`
    last_cal_df = read_calibration_file(os.path.join(src_dir, 'calibration_test_7.csv'))
    last_cal_first = last_cal_df.iloc[0].to_dict()
    last_cal_start = last_cal_first['unix_ms']
    last_cal_frame = last_cal_first['frame']
    # then append it to our trials
    trials.at[trials.index[-1], 'end_unix_ms'] = last_cal_start
    trials.at[trials.index[-1], 'end_frame'] = last_cal_frame
    # typecast start and end frames as integer
    trials = trials.astype({"start_frame": int, "end_frame": int})
    # Cleanup: 
    # 1. Create "trial #" column that numbers each trial, 
    # 2. Rename trials to remove extraneous info, and 
    # 3. Mark if a trial is one that involves audio or not
    trials['trial_id'] = range(1, len(trials) + 1)
    trials['trial_name'] = trials['trial_name'].apply(lambda x: (x.split(" ")[0]).split('-')[1])
    trials['trial_audio'] = trials['trial_name'].apply(lambda x: "Audio" in x)
    trials = trials[['trial_id','trial_name','trial_audio','start_unix_ms','start_frame','end_unix_ms','end_frame']]

    # Create a directory for each trial, save the calibration file associated with that trial, and save it inside of a subdirectory in each trial called `calibration/`
    for index, row in trials.iterrows():
        # Create the directory
        tdir = mkdirs(os.path.join(src_dir, f"{row['trial_id']}/"))
        #p['trials'].append(f"{row['trial_id']}")
        # Create a `calibration` subdirectory inside the trial directory
        #cdir = os.path.join(tdir, 'calibration/')
        #mkdirs(cdir)
        # Save calibration file (cleaned up one)
        cdf = read_calibration_file(os.path.join(src_dir, f"calibration_test_{row['trial_id']}.csv"))
        cdf.to_csv(os.path.join(tdir, "calibration.csv"), index=False) 
        # Splice the eye, ped, eeg, and bandpowers based on timing
        #trial_eye = eye[eye['unix_ms'].between(start_time, end_time)]
        #trial_eye.to_csv(os.path.join(tdir, 'eye.csv'), index=False)
        #trial_peds = pedestrians[pedestrians['unix_ms'].between(start_time, end_time)]
        #trial_peds.to_csv(os.path.join(tdir, 'pedestrians.csv'), index=False)

    # Save trials as `trials.csv`, then return output path of `trials.csv`
    outpath = os.path.join(src_dir, 'trials.csv')
    trials.to_csv(outpath, index=False)
    return outpath

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Looking at a participant's data, separates and splits data into their constituent trials.")
    parser.add_argument('src_dir', help="The root dir where the participant's VR data is stored. Should contain a bunch of calibration CSV files and an eye csv file, among other things.")
    parser.add_argument('-es', '--eye_src', help="The filepath relative to `src_dir` that references the participant's eye data", type=str, default='eye.csv')
    #parser.add_argument('-ps', '--ped_src', help="The filepath relative to `<src_dr>/vr/` that references the participant's pedestrian data", type=str, default='pedestrians.csv')
    #parser.add_argument('-bs', '--bandpower_src', help="The filepath relative to `<src_dir>/eeg/` that references the participant's bandpower data.", type=str, default="bandpowers.csv")
    parser.add_argument('-tc', '--timestamp_column', help='The timestamp column of choice', default='unix_ms')
    args = parser.parse_args()
    identify_trials(
        args.src_dir, 
        args.eye_src, 
        #args.ped_src, 
        #args.eeg_src,
        #args.bandpower_src,
        ts_col=args.timestamp_column)

import os
import pandas as pd
import argparse

from RecordMuse.processing import convert, filter, normalize
import trials
import offsets

def process_participant(
        pdir:str, 
        pid:str,
        convert_eeg:bool, 
        convert_groupby_choice:str,
        filter_eeg:bool, 
        calculate_offsets:bool,
        rest_eeg_filename:str='eeg-rest.csv', 
        vr_eeg_filename:str='eeg-vr.csv',
        eye_filename:str='eye.csv',
        blinks_filename:str=None,
):
    # Understand our paths to our EEG files
    eeg_rest_src = os.path.join(pdir, rest_eeg_filename)
    eeg_vr_src = os.path.join(pdir, vr_eeg_filename)
    blinks_vr_src = os.path.join(pdir, blinks_filename) if blinks_filename is not None else None
    assert os.path.exists(eeg_rest_src), f'Rest EEG path \"{eeg_rest_src}\" does not exist'
    assert os.path.exists(eeg_vr_src), f'VR EEG path \"{eeg_vr_src}\" does not exist'
    
    # Optional step: convert if toggled
    if convert_eeg:
        print("\n\033[4mConverting EEG files:\033[0m\n")
        _, eeg_rest_src, _, _, _ = convert.mm_to_bluemuse(eeg_rest_src, output_dir='eeg-rest', groupby_choice=convert_groupby_choice)
        print("- \033[36mNew Rest EEG:\033[0m", eeg_rest_src)
        _, eeg_vr_src, _, _, blinks_vr_src = convert.mm_to_bluemuse(eeg_vr_src, output_dir='eeg-vr', groupby_choice=convert_groupby_choice)
        print("- \033[31mNew VR EEG:\033[0m", eeg_vr_src)
    
    # Optional step: filter if toggled
    if filter_eeg:
        print("\n\033[4mFiltering EEG files:\033[0m\n")
        eeg_rest_src = filter.filter_eeg(eeg_rest_src, apply_bandpass=True, verbose=False)
        print("- \033[36mFiltered Rest EEG:\033[0m", eeg_rest_src)
        eeg_vr_src = filter.filter_eeg(eeg_vr_src, apply_bandpass=True, verbose=False)
        print("- \033[31mFiltered VR EEG:\033[0m", eeg_vr_src)

    # Normalize the VR EEG data using rest EEG data
    print("\n\033[4mNormalizing EEG data:\033[0m\n")
    eeg_src = normalize.normalize(eeg_rest_src, eeg_vr_src, ts_col='unix_ms', start_buffer=2500, end_buffer=5000, validate=True)
    print("- \033[31mNormalized EEG:\033[0m", eeg_src)

    # Identify Trials
    print("\n\033[4mIdentifying Trials:\033[0m\n")
    trials_src = trials.identify_trials(pdir, eye_filename, ts_col='unix_ms')
    print("- \033[31mTrials:\033[0m", trials_src)

    print("\n\033[4mCalculating Offsets:\033[0m\n")
    # Offset Calculation, but ONLY if we need to (or if we are forced to)
    offsets_src = os.path.join(pdir, 'offsets.csv')
    if calculate_offsets or not os.path.exists(offsets_src):
        # Calculate if `offsets.csv` doesn't exist or if we are forcing
        offsets_src = offsets.calculate_offsets(
            pdir, 
            os.path.relpath(trials_src, start=pdir),
            eye_filename, 
            os.path.relpath(eeg_src, start=pdir), 
            blinks_src=blinks_vr_src,
            ts_col='unix_ms', 
            start_buffer=5000, 
            end_buffer=500
        )
    # Reading offset src, checking if it has PID as a column, then modifying if needed
    assert offsets_src is not None, 'Invalid participant!'
    offsets_df = pd.read_csv(offsets_src)
    if 'pid' not in offsets_df.columns:
        print("\033[36mModifying offsets with PID\033[0m")
        offsets_df['pid'] = pid
        offsets_df.to_csv(offsets_src, index=False)
    print("- \033[31mOffsets:\033[0m", offsets_src)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Given a participant directory, automate their operation to reduce the need to run commands in the terminal window.")
    parser.add_argument('participant_dir', help="Relative path to the directory of the participant data.", type=str)
    parser.add_argument('participant_name', help="The participant's unique ID", type=str)
    parser.add_argument('-c', '--convert', help="Convert the EEG data into a BlueMuse-compatible format? Please use if using EEG data directly read through other LSL streaming apps like Mind Monitor", action="store_true")
    parser.add_argument('-gbc', '--groupby_choice', help="For conversion, groupby then use either last or first", type=str, choices=['last','first'], default='first')
    parser.add_argument('-f', '--filter', help="Filter the EEG data to remove 60Hz and bandpass", action="store_true")
    parser.add_argument('-o', '--offsets', help="FORCE the system to calculate offsets.", action='store_true')
    parser.add_argument('-re', '--rest_eeg', help="The filename of the rest EEG. If converted, the path will be adjusted to accomodate the outputted conversion folder path.", type=str, default="eeg-rest.csv")
    parser.add_argument('-ve', '--vr_eeg', help="The filename of the vr EEG. If converted, the path will be adjusted to accomodate the outputted conversion folder path.", type=str, default="eeg-vr.csv")
    parser.add_argument('-e', '--eye', help="The filename of the eye data file", type=str, default='eye.csv')
    parser.add_argument('-b', '--blinks', help="The flename of the blinks data file", type=str, default="")
    args = parser.parse_args()
    process_participant(
        args.participant_dir, 
        args.participant_name,
        args.convert, 
        args.groupby_choice,
        args.filter,
        args.offsets,
        rest_eeg_filename = args.rest_eeg, 
        vr_eeg_filename = args.vr_eeg,
        eye_filename = args.eye,
        blinks_filename=args.blinks if len(args.blinks) > 0 else None
    )

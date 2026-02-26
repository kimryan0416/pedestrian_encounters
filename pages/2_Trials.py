import streamlit as st
from pathlib import Path
import subprocess
import sys
from utils.core import \
    find_recording_dirs, \
    check_multiple_dirs_with_files, \
    check_valid_dir_with_file
from utils.trials import identify_trials
from utils.offsets import calculate_offsets
from utils.separate import separate_files

# ================================
# Page Logistics
# ================================
_TITLE = "Trial Management"
_DESCRIPTION =  "_The raw data from the VR analysis needs to be sorted, isolated into their own trial-specific folders, and more. " + \
                "We also need to find _valid_ trials - those that have the necessary files. This part of the application focuses " + \
                "on this task specifically."

st.set_page_config(page_title=_TITLE, page_icon="📁")
st.title(_TITLE)
st.markdown(_DESCRIPTION)

# ================================
# State Initialization
#   We store references to existing trials that match certain qualifications
# ================================
if "trials" not in st.session_state:    st.session_state.trials = []
if "selected" not in st.session_state:  st.session_state.selected = set()


# ================================
# State Initialization
#   We store references to existing trials that match certain qualifications
# ================================
st.divider()
st.header("Identifying and Selecting Trials to Analyze")
st.markdown("The first step is to actually identify our trials. You must provide a root directory to search, and provide the expected filepaths to the requested files.")
# --------------------------
# Interactive fields.
# --------------------------
root_dir = st.text_input("Root directory to scan", value="./samples/pedestrian_encounters/")
col1, col2 = st.columns(2)
with col1:
    eeg_src = st.text_input(
        "Simulation EEG filepath", 
        value='_recording_vr/EEG_filtered_normalized.csv', 
        help="_Filepath (relative to the directory of each **trial**) to the EEG file representing the simulation EEG_"
    ) 
col1, col2 = st.columns(2)
with col1:
    eye_src = st.text_input(
        "Eye-tracking filepath", 
        value='eye.csv', 
        help="_Filepath (relative to the directory of each **trial**) to the eye-tracking data_"
    )
with col2:
    ped_src = st.text_input(
        "Pedestrian filepath", 
        value='pedestrians.csv', 
        help="_Filepath (relative to the directory of each **trial**) to the pedestrian data_"
    )
# --------------------------
# Interactive Buttons
# --------------------------
if st.button("Scan for valid trials", width="stretch"):
    root = Path(root_dir).resolve()
    if not root.exists():
        st.error("Directory does not exist")
    else:
        st.session_state.trials = find_recording_dirs(root, prefix=None, recursive=False)
        st.session_state.selected.clear()


# ================================
# Interaction #1: Selecting valid trial directories, then doing stuff to them.
# ================================
trials = st.session_state.trials
if not trials:  
    st.info("No trial directories found.")
else:
    # Initially check - how many are valid, and how many are invalid?
    valid, invalid = check_multiple_dirs_with_files(trials, [eeg_src, eye_src, ped_src])
    # If we don't have any valids, then it's bust
    if len(valid) == 0:
        st.info(f"Found {len(trials)} trial(s) [{len(valid)}/{len(trials)} successful matches]")
    # We found at least one valid trial
    else:
        st.success(f"Found {len(trials)} recording(s)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("_Trials that contain both rest and simulation EEG:_")
            if len(valid) == 0:
                st.info("No valid trials! Double-check your data!")
            else:
                for trial in valid:
                    name = trial['name']
                    key = trial["relative_path"]
                    created = trial['created'].strftime('%Y-%m-%d %H:%M:%S')
                    checked = st.checkbox(
                        f"{name}  (`{key}` - {created})",
                        key=f"chk_{key}",
                        value=key in st.session_state.selected
                    )
                    if checked: st.session_state.selected.add(key)
                    else:       st.session_state.selected.discard(key)
        with col2:
            st.markdown("_Trials that are missing one or more necessary files:_")
            if len(invalid) == 0:
                st.info("No invalid trials! Congratulations!")
            else:
                for trial in invalid:
                    name = trial['name']
                    key = trial["relative_path"]
                    created = trial['created'].strftime('%Y-%m-%d %H:%M:%S')
                    st.checkbox(
                        f"📁 {name}  (`{key}` - {created})",
                        key=f"chk_{key}",
                        value=key in st.session_state.selected,
                        disabled=True
                    )

# --------------------------
# Actual operations for handling trials
# --------------------------
if len(st.session_state.selected) > 0:
    selected = [
        t for t in st.session_state.trials 
        if t['relative_path'] in st.session_state.selected
    ]
    # --------------------------
    # Explanation Text
    # --------------------------
    st.divider()
    st.markdown(
        "After selecting the trials above, folow these steps below:\n" + \
        "1. _Generating trial directories,_\n" + \
        "2. _Calculating offsets between VR and EEG timestamps, and_\n" + \
        "3. _Separate all files into their respective trials, given the calculated offsets._"
    )


    # ================================
    # Interaction #2: Perform the following:
    #   1. Generating trial directories"
    #   2. Calculating offsets between VR and EEG timestamps"
    #   3. Separate all files into their respective trials, given the calculated offsets."
    # ================================
    # Interaction: what kind of tiemstamp column should be used?
    ts_col = st.text_input("Timestamp column name", value='unix_ms', help="The column to use as the representative timestamp column in your data.")
    # Division into columns
    col1, col2, col3 = st.columns([0.35, 0.3, 0.35])
    # Operation #1: Generating directories
    with col1:
        # Operation #1: Generate separate directories for each trial
        if st.button("Generate directories", help="This operation should generate folders named \"1\" to \"6\", each with a `calibrations.csv`. A `trials.csv` is also generated too.", width="stretch"):
            # We only use ones that are verified recordings
            # Generate a progress bar
            progress_bar = st.progress(0, text="Analyzing trials...")
            for index in range(len(selected)):
                # Get the current recording and relative path
                trial = selected[index]
                name = trial['name']
                path = trial["absolute_path"]
                # Update the progress bar
                progress_bar.progress(int((index)/len(selected) * 100), text=name)
                # Validate the files initially by plotting them
                identify_trials(str(path), eye_src=eye_src, ts_col=ts_col)
                # Update the progress bar
                progress_bar.progress(int((index+1)/len(selected) * 100), text=name)
            progress_bar.progress(100, "All selected trials processed!")
    # Operation #2: Generate offsets for trials
    with col2:
        with st.popover("Calc. Offsets", width="stretch"):
            st.markdown("### Disclaimer: Has to be done manually")
            st.markdown("After generating trial directories, you are expected to have generated an `offsets.py`. This `offset.py` would have been used when segmenting the data itself into each trial. However, this operation cannot be done via Streamlit (the package running this interface). The only alternative is to run the operation via command line manually. Here's the general command you will want to run:")
            st.code("python offsets.py [src_dir] [-ts] [-es] [-ees] [-tc] [-sb] [-eb] [-s]", language="bash")
            st.markdown(
                "Where:\n" + \
                "- `-src_dir`: The directory of the participant\n" + \
                "- `-ts`, `--trial_src`: The filepath relative to `src_dr` that references the participant's trial listings\n" + \
                "- `-es`, `--eye_src`: The filepath relative to `src_dir` that references the participant's eye data\n" + \
                "- `-ees`, `--eeg_src`: The filepath relative to `src_dir` that references the participant's EEG data, usually after filtering and normalization\n" + \
                "- `-tc`, `--timestamp_column`: The timestamp column of choice\n" + \
                "- `-sb`, `--start_buffer`: The amount of time removed from the start of each calibration stage\n" + \
                "- `-eb`, `--end_buffer`: The amount of time removed from the end of each calibration stage\n" + \
                "- `-s`, `--smooth`: If toggled, will attempt to smooth based on in-built parameters"
            )
    # Operation #3: Separating Data
    with col3:
        # Button interaction
        if st.button("Separating Data", width="stretch"):
            # Generate a progress bar
            progress_bar = st.progress(0, text="Separating data...")
            errors = []
            for index in range(len(selected)):
                # Get the current recording and relative path
                trial = selected[index]
                name = trial['name']
                path = trial["absolute_path"]
                # Update the progress bar
                progress_bar.progress(int((index)/len(selected) * 100), text=name)
                # Check: does this trial have an `offsets.py`?
                if check_valid_dir_with_file(path, ['offsets.csv']):
                    # Conduct the operation
                    separate_files(str(path), 'offsets.csv', 'trials.csv', eeg_src, eye_src, ped_src, ts_col=ts_col)
                else:
                    errors.append(name)
                # Update the progress bar
                progress_bar.progress(int((index+1)/len(selected) * 100), text=name)
            # Update progress bar
            if len(errors) == 0:
                progress_bar.progress(100, "All trials successfully separated!")
            elif len(errors) != len(selected):
                progress_bar.progress(int((len(selected)-len(errors))/len(selected) * 100), "Errors in one or more trial directories!")
                st.error(
                    "The following trials had issues:\n" + \
                    '\n'.join([f"- {n}" for n in errors])
                )


# ================================
# Command Lines: For running the operations via command line
# ================================
st.divider()
st.header("Command Line Equivalents")
st.markdown(
    "If you want to run these commands via the command line, " + \
    "then here are the list of commands."
)
# --------------------------
# Interactions: 
#   1. Trial Directory Generation
#   2. Offset Calculations
#   3. Data Separation
# --------------------------
with st.expander("### Command Lines"):
    st.code(
        "# Trial Directory Generation:\n" + \
        "python utils/trials.py [root data directory of participant] [-es] [-tc]\n" + \
        "# Offset Calculations:\n" + \
        "python utils/offsets.py [src_dir] [-ts] [-es] [-ees] [-tc] [-sb] [-eb] [-s]\n" + \
        "# Data Separation:\n" + \
        "python utils/separate.py [src directory to participant] [-o] [-t] [-eeg] [-ey] [-p] [-tc]", 
        language="bash"
    )
# --------------------------
# Interaction #2: Pre-process EEG
# --------------------------
with st.expander("### Pre-processing EEG"):
    st.code(
        "# Filter EEG:\n" + \
        "python RecordMuse/processing/filter.py [path to your rest or vr eeg `.csv`] -b -v\n" + \
        "# Normalize EEG:\n" + \
        "python RecordMuse/processing/normalize.py [path to rest EEG] [path to VR eeg] [-tc] [-sb] [-eb] [-v]\n" + \
        "# Calculate PSD and bandpowers:\n" + \
        "python RecordMuse/analysis/psd.py [path to filtered, normalized EEG]\n" + \
        "# Validate via Plotting:\n" + \
        "python RecordMuse/analysis/validate.py <path/to/directory> [-tc <timestamp/column/name>] [-p]", 
        language="bash"
    )

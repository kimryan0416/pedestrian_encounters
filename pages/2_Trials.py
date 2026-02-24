import streamlit as st
from pathlib import Path
import subprocess
import sys
from utils.core import find_recording_dirs, check_valid_dir_with_file
from utils.trials import identify_trials
from utils.offsets import calculate_offsets

_TITLE = "Trial Management"
_DESCRIPTION =  "_The raw data from the VR analysis needs to be sorted, isolated into their own trial-specific folders, and more. " + \
                "We also need to find _valid_ trials - those that have the necessary files. This part of the application focuses " + \
                "on this task specifically. This page therefore is dedicated with:_\n" + \
                "1. _Generating trial directories,_\n" + \
                "2. _Calculating offsets between VR and EEG timestamps, and_\n" + \
                "3. _Separete all files into their respective trials, given the calculated offsets._"

st.set_page_config(page_title=_TITLE, page_icon="📁")
st.title(_TITLE)
st.markdown(_DESCRIPTION)

# Initialize states, which store references to existing participant directories and those that are selected
if "trials" not in st.session_state:            st.session_state.trials = []
if "selected_trials" not in st.session_state:   st.session_state.selected_trials = set()

# This is where we have the input system for finding files with the prefix `_recording_`.
st.header("Identifying and Selecting Trials to Analyze")
st.markdown("The first step is to actually identify our trials. You must provide a root directory to search, and provide the expected filepaths to the requested files.")
# This input text allows someone to check the root directory to scan
root_dir = st.text_input("Root directory to scan", value="./samples/pedestrian_encounters/")
# We split between two columns
col1, col2 = st.columns(2)
with col1:
    rest_eeg_src = st.text_input(
        "Rest EEG filepath", 
        value='_recording_rest/EEG.csv', 
        help="_Filepath (relative to the directory of each **trial**) to the EEG file representing the rest-state EEG_"
    )
with col2:
    sim_eeg_src = st.text_input(
        "Simulation EEG filepath", 
        value='_recording_vr/EEG.csv', 
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

# This button performs the finding of such recordings
if st.button("Scan for valid trials"):
    root = Path(root_dir).resolve()
    if not root.exists():
        st.error("Directory does not exist")
    else:
        st.session_state.trials = find_recording_dirs(root, prefix=None, recursive=False)
        st.session_state.selected_trials.clear()

# This interface now lists out all identified recordings, and allows someone to check which ones they want to analyze
trials = st.session_state.trials
if not trials:  
    st.info("No trial directories found.")
else:
    st.success(f"Found {len(trials)} recording(s)")
    col1, col2 = st.columns(2)
    with col1:
        for rec in trials:
            key = rec["relative_path"]
            valid_dir = check_valid_dir_with_file(rec['absolute_path'], [rest_eeg_src, sim_eeg_src, eye_src, ped_src])
            checked = st.checkbox(
                f"{rec['name']}  (`{key}` - {rec['created'].strftime('%Y-%m-%d %H:%M:%S')})",
                key=f"chk_{key}",
                value=key in st.session_state.selected_trials,
                disabled=not valid_dir
            )
            if checked: st.session_state.selected_trials.add(key)
            else:       st.session_state.selected_trials.discard(key)
    with col2:
        st.subheader("Trial Operations")
        # Operation #1: Generate separate directories for each trial
        if st.button("Generate directories", help="This operation should generate folders named \"1\" to \"6\", each with a `calibrations.csv`. A `trials.csv` is also generated too."):
            # We only use ones that are verified recordings
            selected = [
                rec for rec in trials
                if rec["relative_path"] in st.session_state.selected_trials
            ]
            # If no recordings, then we return an error message
            if not selected:
                st.warning("No trials selected.")
            # If at least one recording, then we operate.
            else:
                # Generate a progress bar
                progress_bar = st.progress(0, text="Analyzing trials...")
                for index in range(len(selected)):
                    # Get the current recording and relative path
                    rec = selected[index]
                    path = rec["absolute_path"]
                    # Validate the files initially by plotting them
                    identify_trials(str(path), eye_src="eye.csv", ts_col='unix_ms')
                    # Update the progress bar
                    percentage = int((index+1)/len(selected) * 100)
                    progress_bar.progress(percentage, text=str(path))
        # Operation #2: Generate offsets for trials
        if st.button("Calculate Offsets", help="This option requires manual inputs to calculate offsets between peaks/valleys across EEG and VR eye data; generates an `offsets.csv` file in the trial directory. **Can be time-consuming!**"):
            # We only use ones that are verified recordings
            selected = [
                rec for rec in trials
                if rec["relative_path"] in st.session_state.selected_trials
            ]
            # If no recordings, then we return an error message
            if not selected:    
                st.warning("No trials selected.")
            # If at least one recording, then we operate.
            else:
                # Generate a progress bar
                progress_bar = st.progress(0, text="Analyzing trials...")
                for index in range(len(selected)):
                    # Get the current recording and relative path
                    rec = selected[index]
                    path = rec["absolute_path"]
                    # Validate the files initially by plotting them
                    subprocess.Popen([
                        sys.executable, "utils/offsets.py", str(path),
                        '-ts', 'trials.csv',
                        '-es', eye_src,
                        '-ees', sim_eeg_src,
                    ])
                    #offset_success = calculate_offsets(str(path), 'trials.csv', eye_src, sim_eeg_src) is not None
                    #if not offset_success:
                    #    st.info(f"Trial {rec['name']} could not generate offsets...")
                    # Update the progress bar
                    percentage = int((index+1)/len(selected) * 100)
                    progress_bar.progress(percentage, text=str(path))
        



# This is the input for all selected directories from above.
st.divider()

col3, col4, col5 = st.columns(3)
with col3:
    st.header("Generate individual Trial Directories")
    st.divider()
    st.subheader("Generate Trial Directories")

    if st.button("Generate directories"):
        # We only use ones that are verified recordings
        selected = [
            rec for rec in recordings
            if rec["relative_path"] in st.session_state.selected_trials
        ]
        # If no recordings, then we return an error message
        if not selected:
            st.warning("No trials selected.")
        # If at least one recording, then we operate.
        else:
            # Generate a progress bar
            progress_bar = st.progress(0, text="Analyzing trials...")
            for index in range(len(selected)):
                # Get the current recording and relative path
                rec = selected[index]
                path = rec["absolute_path"]

                # Validate the files initially by plotting them
                identify_trials(str(path), eye_src="eye.csv", ts_col='unix_ms')

                # Update the progress bar
                percentage = int((index+1)/len(selected) * 100)
                progress_bar.progress(percentage, text=str(path))

with col4:
    st.header("Calculate Offests")
    st.markdown("This is the **second** step. This is a _time-consuming_ process, but it requires **manual** input. Your task is to estimate timing offsets between EEG and eye-tracking peaks/valleys.")
    if st.button("Calculate Offsets"):
        # We only use ones that are verified recordings
        selected = [
            rec for rec in recordings
            if rec["relative_path"] in st.session_state.selected_trials
        ]
        # If no recordings, then we return an error message
        if not selected:
            st.warning("No trials selected.")
        # If at least one recording, then we operate.
        else:
            # Generate a progress bar
            progress_bar = st.progress(0, text="Analyzing trials...")
            for index in range(len(selected)):
                # Get the current recording and relative path
                rec = selected[index]
                path = rec["absolute_path"]

                # Validate the files initially by plotting them
                calculate_offsets(
                    str(path), eye_src="eye.csv", ts_col='unix_ms')

                # Update the progress bar
                percentage = int((index+1)/len(selected) * 100)
                progress_bar.progress(percentage, text=str(path))
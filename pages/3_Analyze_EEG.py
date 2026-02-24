import os
import streamlit as st
from pathlib import Path
from utils.core import find_recording_dirs, check_valid_dir_with_file
from RecordMuse.analysis.psd import calculate_psd
from RecordMuse.analysis.validate import validate

st.title("Analyzing EEG")
st.markdown("After you've recorded your EEG data, you must process it into a form that makes sense for future correlation analysis. This involves calculating bandpowers and such.")

# Initialize states, which store references to existing participant directories and those that are selected
if "recordings" not in st.session_state:
    st.session_state.recordings = []
if "selected_recordings" not in st.session_state:
    st.session_state.selected_recordings = set()

# This is where we have the input system for finding files with the prefix `_recording_`.
st.header("Identifying and Selecting EEG Data to Analyze")
col1, col2 = st.columns(2)
with col1:
    # This input text allows someone to check the root directory to scan
    root_dir = st.text_input("Root directory to scan", value="./samples/pedestrian_encounters/")
with col2:
    # This button performs the finding of such recordings
    if st.button("Scan for recordings"):
        root = Path(root_dir).resolve()
        if not root.exists():
            st.error("Directory does not exist")
        else:
            st.session_state.recordings = find_recording_dirs(root)
            st.session_state.selected_recordings.clear()

# This interface now lists out all identified recordings, and allows someone to check which ones they want to analyze
recordings = st.session_state.recordings
if not recordings:
    st.info("No recording directories found.")
else:
    st.success(f"Found {len(recordings)} recording(s)")
    for rec in recordings:
        key = rec["relative_path"]
        valid_dir = check_valid_dir_with_file(rec['absolute_path'], ['EEG.csv','Accelerometer.csv','Gyroscope.csv'])
        checked = st.checkbox(
            f"{rec['name']}  (`{key}` - {rec['created'].strftime('%Y-%m-%d %H:%M:%S')})",
            key=f"chk_{key}",
            value=key in st.session_state.selected_recordings,
            disabled=not valid_dir
        )
        if checked: st.session_state.selected_recordings.add(key)
        else:       st.session_state.selected_recordings.discard(key)

# This is the input for all selected directories from above.
st.divider()
if st.button("Validate and calcualte PSDs for these EEG recordings"):
    # We only use ones that are verified recordings
    selected = [
        rec for rec in recordings
        if rec["relative_path"] in st.session_state.selected_recordings
    ]
    # If no recordings, then we return an error message
    if not selected:
        st.warning("No recordings selected.")
    # If at least one recording, then we operate.
    else:
        # Generate a progress bar
        progress_bar = st.progress(0, text="Analyzing selected recordings...")
        for index in range(len(selected)):
            # Get the current recording and relative path
            rec = selected[index]
            path = rec["absolute_path"]

            # Validate the files initially by plotting them
            validate(str(path)+'/', with_lines=True, show=False)

            # Generate the PSD, based on the eeg data
            calculate_psd(os.path.join(path, 'EEG.csv'))

            # Update the progress bar
            percentage = int((index+1)/len(selected) * 100)
            progress_bar.progress(percentage, text=str(path))
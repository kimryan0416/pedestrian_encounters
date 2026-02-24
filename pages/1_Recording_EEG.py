import streamlit as st
import subprocess
import sys
from pathlib import Path
from utils.core import \
    find_recording_dirs, \
    check_valid_dir_with_file, \
    check_multiple_dirs_with_files
from RecordMuse.processing.convert import mm_to_bluemuse
from RecordMuse.processing.filter import filter_eeg
from RecordMuse.processing.normalize import normalize
from RecordMuse.analysis.psd import calculate_psd


# ================================
# Page Logistics
# ================================
_TITLE = "Recording EEG"
_DESCRIPTION = "_Here, you can easily visualize or record EEG data. " + \
                "This involves using our custom package in combination with " + \
                "\"BlueMuse\" ([https://github.com/kowalej/BlueMuse](https://github.com/kowalej/BlueMuse)), " + \
                "a Windows-based LSL wrapper for recording EEG data from Interaxon's \"Muse\" line of BCIs " + \
                "([https://choosemuse.com/](https://choosemuse.com/)). The link to our own custom " + \
                "package is located here: [https://github.com/SimpleDevs-Tools/RecordMuse](https://github.com/SimpleDevs-Tools/RecordMuse)._"

st.set_page_config(page_title=_TITLE, page_icon="📈")
st.title(_TITLE)
st.markdown(_DESCRIPTION)


# ================================
# State Initialization
#   We store references to existing trials that match certain qualifications
#   Unique to this page, we also store a `proc` or "process" state that is referred to when demo-ing or recording EEG data.
# ================================
if "trials" not in st.session_state:    st.session_state.trials = []
if "selected" not in st.session_state:  st.session_state.selected = set()
if "proc" not in st.session_state:      st.session_state.proc = None
# --------------------------
# This is a unique helper function. It tells us if a process is running
# --------------------------
def is_running():
    p = st.session_state.proc
    return p is not None and p.poll() is None


# ================================
# Interface #1: Demo-ing, Recording, and Converting
#   In the interface, we have 3 columns, each with unique functions.
#   1. DEMO the recording procedure. This invokes the `proc` state to check if the subprocess for it is running or not.
#   2. RECORD the EEG. This also uses the `proc` state and actually saves the recordings with a `_recording_` prefix
#   3. CONVERT recordings from Mind Monitor into the correct formatting, compatible wiht BlueMuse recordings.
#  ================================
col1, col2, col3 = st.columns(3)

# --------------------------
# Column 1: Demoing
# --------------------------
with col1:
    # Header and text description
    st.header("Demo")
    st.markdown("See if your BlueMuse system is working. This is **purely** for visualization and debugging; it doesn't actually record.")
    # Interactable buttons
    if st.button("Run Demo Visualization", disabled=is_running()):
        st.session_state.proc = subprocess.Popen([sys.executable, "RecordMuse/record/demo.py"])
        st.rerun()
    if st.button("Stop Demo Visualization", disabled=not is_running()):
        st.session_state.proc.terminate()
        st.session_state.proc = None
        st.rerun()
        
# --------------------------
# Column 2: Recording
# --------------------------
with col2:
    # Header and text description
    st.header("Record")
    st.markdown("Record data using BlueMuse and your Muse device. For best results, follow the default settings provided.")
    # Command line designations abd interactable fields
    arg1 = st.text_input("Output directory", value=None, placeholder="Blank = datetime-stamped dir.")
    arg2 = st.number_input("Recording duration (seconds)", value=600, placeholder="How long should the recording last (in seconds)?")
    arg3 = st.checkbox("Visualize streams", value=False)
    # Interactable buttons
    if st.button("Start Recording", disabled=is_running()):
        args = [ sys.executable, "RecordMuse/record/record.py"]
        if arg1 is not None: args.extend(['-d', arg1])
        if arg2 is not None: args.extend(['-rd', str(arg2)])
        if arg3: args.append('-v')
        st.session_state.proc = subprocess.Popen(args)
        st.rerun()
    if st.button("Stop Recording", disabled=not is_running()):
        st.session_state.proc.terminate()  # SIGTERM
        st.session_state.proc = None
        st.rerun()

# --------------------------
# Column 3: Converting
# --------------------------
with col3:
    # Header and text description
    st.header("Convert")
    st.markdown("Convert from Mind Monitor's formating to BlueMuse's format. _Timestamps may not properly convert._", help="Mind Monitor condenses all recordings into a single row, including EEG, IMU, and Heart Rate. Consequently, many rows may share the same timestamp. This system will try its best to account for this, but you'll likely only get the last recording per timestamp group.")
    # Command line designations and interactable fields
    arg1 = st.text_input("EEG file from Mind Monitor.", value=None)
    arg2 = st.text_input("Output directory name (optional)", value=None, placeholder="Blank = automatically generated")
    arg3 = st.selectbox("Timestamp Group Candidate Selection", options=['Last','First'])
    # Interactable buttons
    if st.button("Convert"):
        output_dir, eeg_outpath, accel_outpath, gyro_outpath, blinks_outpath = mm_to_bluemuse(arg1, output_dir=arg2, groupby_choice=arg3.lower())
        st.markdown(f"Converted files saved within `{output_dir}`")


# ================================
# Interface #2: Pre-Processing EEG
#   Assuming that you use the tools above to record and/or convert your EEG 
#   data into a usable format, we can now pre-process the EEG. To do this, 
#   we need an interface (just a form) to identify viable trials with.
# ================================
st.divider()
st.header("Pre-process EEG")
st.markdown("All recordings are identified with a prefix `_recording_` to them. You can use this tool to check how many have been identified. You can also control under which root directory you want to check under, if needed.")
st.markdown("Regardless, the current task is to, for each trial with rest EEG and simulation EEG, normalize the data for PSD analysis.")
# --------------------------
# Interactive fields.
# --------------------------
root_dir = st.text_input("Root directory to scan", value="./samples/pedestrian_encounters/")
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
# --------------------------
# Interactive button
# --------------------------  
if st.button("Scan for Trials", width="stretch"):
    root = Path(root_dir).resolve()
    if not root.exists():
        st.error("Directory does not exist")
    else:
        st.session_state.trials = find_recording_dirs(root, prefix=None, recursive=False)
        st.session_state.selected.clear()

# ================================
# Interaction #3: Selecting files, then post-processing them.
#   We need an interface that does several things. Namely:
#   1. Checking for the trials that actually contain the designated rest and simulation EEG files
#   2. Filtering the signal data for both the rest and EEG data
#   3. Normalizing the simulation data based on the EEG data
#   4. Calculating the PSDs based on the normalized data
# ================================
trials = st.session_state.trials
if not trials: 
    st.info("No trial directories found.")
else:
    # Initially check - how many are valid, and how many are invalid?
    valid, invalid = check_multiple_dirs_with_files(trials, [rest_eeg_src, sim_eeg_src])
    # If we don't have any valids, then it's bust
    if len(valid) == 0:
        st.info(f"Found {len(trials)} trial(s) [{len(valid)}/{len(trials)} successful matches]")
    # We found at least one valid trial
    else:
        st.success(f"Found {len(trials)} trial(s) [{len(valid)}/{len(trials)} successful matches]")
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
                        f"📁 {name}  (`{key}` - {created})",
                        key=f"chk_{key}",
                        value=key in st.session_state.selected,
                    )
                    if checked: st.session_state.selected.add(key)
                    else:       st.session_state.selected.discard(key)
        with col2:
            st.markdown("_Trials that are missing one or both EEG recordings:_")
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
# Actual operations for pre-processing EEG data
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
        "When you click the following button, these operations will occur in order:\n" + \
        "1. **Filter** the signal data for both the rest and EEG data\n" + \
        "2. **Normalize** the simulation data based on the EEG data\n" + \
        "3. **Calculate** the PSDs based on the normalized data"
    )
    # --------------------------
    # Interactive fields
    # --------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Filtering Params")
        st.markdown("This operation generates a new `EEG_filtered.csv` for each of the rest and simulation eeg files.")
        apply_bandpass = st.checkbox("Apply bandpass filtering to smooth EEG signals")
    with col2:
        st.subheader("Normalization Params")
        timestamp_colname = st.text_input("Timestamp column name", value='lsl_unix_ts')
        col2a, col2b = st.columns(2)
        with col2a:     start_buffer = st.number_input("Start buffer", value=5.0, help="The amount of time at the start that is ignored; the time scale is dependent on if your timestamp column is in seconds, milliseconds, or other.")
        with col2b:     end_buffer = st.number_input("End buffer", value=5.0, help="The amount of time at the end that is ignored; the time scale is dependent on your timestamp column's time scale.")
        validate_normalization = st.checkbox("Validate via command line", help="Make sure you have access to the command line if you select this.")
    # --------------------------
    # Interactive button
    # --------------------------
    if st.button(f"Pre-process EEG for selected trials ({len(selected)} trials)", width="stretch"):
        # We begin operations. We add a progress bar for better visualization of the process
        progress_bar = st.progress(0, text="Pre-processing EEG...")
        # Loop through each selected trial
        for index in range(len(selected)):
            trial = selected[index]
            name = trial['name']
            path = trial['absolute_path']
            progress_bar.progress(
                int(index/len(selected) * 100),
                text=f"Pre-processing EEG... | {str(path)}"
            )
            # First op: filtering
            progress_bar.progress(
                int(index/len(selected) * 100),
                text=f"Pre-processing EEG... | {str(path)} | Rest EEG Filtering"
            )
            eeg_rest_filtered_src = filter_eeg(str(path / rest_eeg_src), apply_bandpass=apply_bandpass)
            progress_bar.progress(
                int(index/len(selected) * 100),
                text=f"Pre-processing EEG... | {str(path)} | Simulation EEG Filtering"
            )
            eeg_sim_filtered_src = filter_eeg(str(path / sim_eeg_src), apply_bandpass=apply_bandpass)
            # Second op: normalization
            progress_bar.progress(
                int(index/len(selected) * 100),
                text=f"Pre-processing EEG... | {str(path)} | Normalization"
            )
            eeg_normalized_src = normalize(
                str(path / eeg_rest_filtered_src), 
                str(path / eeg_sim_filtered_src), 
                ts_col=timestamp_colname, 
                start_buffer=start_buffer, 
                end_buffer=end_buffer,
                validate=validate_normalization
            )
            progress_bar.progress(
                int(index/len(selected) * 100),
                text=f"Pre-processing EEG... | {str(path)} | PSD Calculation"
            )
            # Third op: PSD Calculation
            calculate_psd(str(path / eeg_normalized_src))
            # Close off with progresss report
            progress_bar.progress(
                int((index+1)/len(selected) * 100),
                text=f"Pre-processing EEG... | {str(path)} Complete!"
            )
        # Complete the progress bar
        progress_bar.progress(100, text="Completed Pre-processing EEG!")


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
# Interaction #1: Demo, Record, and Convert
# --------------------------
with st.expander("### Demo, Record, or Convert EEG"):
    st.code(
        "# Demo:\n" + \
        "python RecordMuse/record/demo.py\n" + \
        "# Record EEG via BlueMuse:\n" + \
        "python RecordMuse/record/record.py [-d <OUTPUT_DIR>]\n" + \
        "# Convert:\n" + \
        "python RecordMuse/processing/convert.py [path to eeg file]", 
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
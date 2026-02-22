import streamlit as st
import subprocess
import sys
from pathlib import Path
from utils.core import find_recording_dirs
from RecordMuse.processing.convert import mm_to_bluemuse

st.set_page_config(page_title="Recording EEG", page_icon="📈")

_TITLE = "Recording EEG"
_DESCRIPTION = "_Here, you can easily visualize or record EEG data. " + \
                "This involves using our custom package in combination with " + \
                "\"BlueMuse\" ([https://github.com/kowalej/BlueMuse](https://github.com/kowalej/BlueMuse)), " + \
                "a Windows-based LSL wrapper for recording EEG data from Interaxon's \"Muse\" line of BCIs " + \
                "([https://choosemuse.com/](https://choosemuse.com/)). The link to our own custom " + \
                "package is located here: [https://github.com/SimpleDevs-Tools/RecordMuse](https://github.com/SimpleDevs-Tools/RecordMuse)._"

st.title(_TITLE)
st.markdown(_DESCRIPTION)

# Initialize a session state to control recording via subprocesses
if "proc" not in st.session_state:
    st.session_state.proc = None
def is_running():
    p = st.session_state.proc
    return p is not None and p.poll() is None

# Create two columns
recording_cols = st.columns(3)

with recording_cols[0]:
    st.header("Demo")
    st.markdown("See if your BlueMuse system is working. This is **purely** for visualization and debugging; it doesn't actually record.")
   
    if st.button("Run Demo Visualization", disabled=is_running()):
        st.session_state.proc = subprocess.Popen([
            sys.executable, 
            "RecordMuse/record/demo.py"
        ])
        st.rerun()
    
    if st.button("Stop Demo Visualization", disabled=not is_running()):
        st.session_state.proc.terminate()
        st.session_state.proc = None
        st.rerun()
        

with recording_cols[1]:
    st.header("Record")
    st.markdown("Record data using BlueMuse and your Muse device. For best results, follow the default settings provided.")

    arg1 = st.text_input("Output directory", value=None, placeholder="Blank = datetime-stamped dir.")
    arg2 = st.number_input("Recording duration (seconds)", value=600, placeholder="How long should the recording last (in seconds)?")
    arg3 = st.checkbox("Visualize streams", value=False)

    if st.button("Start Recording", disabled=is_running()):
        args = [ sys.executable, "RecordMuse/record/record.py"]
        if arg1 is not None: args.extend(['-d', arg1])
        if arg2 is not None: args.extend(['-rd', str(arg2)])
        if arg3: args.append('-v')
        print(args)
        st.session_state.proc = subprocess.Popen(args)
        st.rerun()

    if st.button("Stop Recording", disabled=not is_running()):
        st.session_state.proc.terminate()  # SIGTERM
        st.session_state.proc = None
        st.rerun()

with recording_cols[2]:
    st.header("Convert")
    st.markdown("Convert from Mind Monitor's formating to BlueMuse's format. _Timestamps may not properly convert._", help="Mind Monitor condenses all recordings into a single row, including EEG, IMU, and Heart Rate. Consequently, many rows may share the same timestamp. This system will try its best to account for this, but you'll likely only get the last recording per timestamp group.")

    arg1 = st.text_input("EEG file from Mind Monitor.", value=None)
    arg2 = st.text_input("Output directory name (optional)", value=None, placeholder="Blank = automatically generated")
    arg3 = st.selectbox("Timestamp Group Candidate Selection", options=['Last','First'])

    if st.button("Convert"):
        output_dir, eeg_outpath, accel_outpath, gyro_outpath, blinks_outpath = mm_to_bluemuse(arg1, output_dir=arg2, groupby_choice=arg3.lower())
        st.markdown(f"Converted files saved within `{output_dir}`")

st.divider()

st.header("Search recordings")
st.markdown("All recordings are identified with a prefix `[recording]-` to them. You can use this tool to check how many have been identified. You can also control under which root directory you want to check under, if needed.")

recordings = find_recording_dirs(Path('.').resolve())
col4, col5 = st.columns(2)

with col4:
    root_dir = st.text_input("Root directory to scan", value=".")
with col5:
    if st.button("Scan for recordings"):
        root = Path(root_dir).resolve()
        if not root.exists():
            st.error("Directory does not exist")
        else:
            recordings = find_recording_dirs(root)

if not recordings: 
    st.info("No recording directories found.")
else:
    st.success(f"Found {len(recordings)} recording(s)")
    for rec in recordings:
        st.markdown(f"""
            **📁 {rec['name']}**  
            • Path: `{rec['relative_path']}`  
            • Created: {rec['created'].strftime('%Y-%m-%d %H:%M:%S')}
            """
        )
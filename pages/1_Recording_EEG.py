import streamlit as st
import subprocess
import sys

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
st.session_state.proc = None
def is_running():
    return st.session_state.proc is not None and st.session_state.proc.poll() is None
start_disabled = is_running()
stop_disabled = not is_running()

# Create two columns
col1, col2 = st.columns(2)

with col2:
    st.header("Demo")
    st.markdown("You can invoke this operation to just see if your BlueMuse system is working. It produces a set of dynamic graphs that visualize the various data channels recorded by BlueMuse. This is **purely** for visualization and debugging; it doesn't actually record.")
    st.markdown("_To close this demo, simply close all windows that pop up. Don't try to `Ctrl+C` as this will turn off this Streamlit application too._")
   
    if st.button("Run Demo Visualization", disabled=start_disabled):
        st.session_state.proc = subprocess.Popen([
            sys.executable, 
            "RecordMuse/record/demo.py"
        ])
    
    if st.button("Stop Demo Visualization", disabled=stop_disabled):
        st.session_state.proc.terminate()
        st.session_state.proc = None
    

with col1:
    st.header("Recording")
    st.markdown("You can invoke this operation to actually record data using BlueMuse and your Muse device. You can toggle two separate parameters to control aspects of the recording session. For best results, follow the default settings provided.")
    st.markdown("_To close this demo, simply close all windows that pop up. Don't try to `Ctrl+C` as this will turn off this Streamlit application too._")

    arg1 = st.text_input("Output directory", value="", placeholder="Leave this blank to create a datetime-stamped output directory")
    arg2 = st.number_input("Recording duration", value=600, placeholder="How long should the recording last (in seconds)?")
    arg3 = st.checkbox("Visualize streams (debugging only)", value=False)

    if st.button("Start Recording", disabled=start_disabled):
        st.session_state.proc = subprocess.Popen(
            [sys.executable, "record.py", '-d', arg1, '-rd', arg2, '-v', arg3]
        )
    if st.button("Stop Recording", disabled=stop_disabled):
        st.session_state.proc.terminate()  # SIGTERM
        st.session_state.proc = None

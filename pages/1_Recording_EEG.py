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

col1, col2 = st.columns(2)

with col1:
    st.header("Demo")
    st.markdown("You can invoke this operation to just see if your BlueMuse system is working. It produces a set of dynamic graphs that visualize the various data channels recorded by BlueMuse. This is **purely** for visualization and debugging; it doesn't actually record.")
    if st.button("Run Demo Visualization"):
        subprocess.Popen([
            sys.executable, 
            "RecordMuse/record/demo.py"
        ])

with col2:
    st.header("Recording")

import streamlit as st

_TITLE = "Pedestrian Encounters: Post-Processing"
_DESCRIPTION = "_This interface allows you to process participant debug data " + \
                "from \"Pedestrian Encounters\"._"

def main(a:str, b:int):
    return f"{a} - {b}"

st.title(_TITLE)
st.markdown(_DESCRIPTION)

st.markdown("# Hello World")
arg1 = st.text_input("Argument 1")
arg2 = st.number_input("Argument 2", value=0)

if st.button("Run Command"):
    result = main(arg1, arg2)
    st.write("Result:", result)
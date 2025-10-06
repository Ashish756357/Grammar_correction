import streamlit as st
from grammar_model import GrammarCorrector

st.set_page_config(page_title="Grammar Correction AI", page_icon="✍️", layout="centered")

st.title("✍️ Grammar Correction AI")
st.write("Fix your grammar instantly using AI!")

# Initialize model
@st.cache_resource
def load_model():
    return GrammarCorrector()

model = load_model()

# User input
user_input = st.text_area("Enter your sentence:", placeholder="Type something like 'He go to school everyday.'")

if st.button("Correct Grammar"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Correcting..."):
            corrected_text = model.correct(user_input)
        st.success("✅ Corrected Sentence:")
        st.text_area("Output", corrected_text, height=100)

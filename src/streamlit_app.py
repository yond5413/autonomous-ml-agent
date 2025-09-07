import streamlit as st
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestrator import Orchestrator
import json

st.title("Autonomous Tabular ML Agent")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview:", df.head())

    # Target selection
    target = st.selectbox("Select target variable", df.columns)

    if st.button("Run Preprocessing Agent"):
        st.info("Sending to orchestrator...")
        orchestrator = Orchestrator()
        result = orchestrator.run_pipeline(df, target)
        st.info("Orchestrator finished.")
        if result:
            st.json(json.loads(result))

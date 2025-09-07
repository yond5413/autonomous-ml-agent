import streamlit as st
import pandas as pd
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestrator import Orchestrator
import json

st.title("Autonomous Tabular ML Agent")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = "/tmp/autonomous-ml-agent"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df_preview = pd.read_csv(file_path, nrows=5)
    st.write("Preview:", df_preview.head())

    # Target selection
    target = st.selectbox("Select target variable", df_preview.columns)

    if st.button("Run Preprocessing Agent"):
        st.info("Sending to orchestrator...")
        orchestrator = Orchestrator()
        try:
            result = orchestrator.run_pipeline(file_path, target)
            st.info("Orchestrator finished.")

            if result:
                st.success("Pipeline completed successfully!")

                if result.get("model_selection"):
                    with st.expander("Model Selection Details", expanded=True):
                        try:
                            model_selection_json = json.loads(result["model_selection"])
                            st.json(model_selection_json)
                        except (json.JSONDecodeError, TypeError):
                            st.text(result["model_selection"])
                
                if result.get("preprocessing"):
                    with st.expander("Preprocessing Result"):
                        try:
                            preprocessing_json = json.loads(result["preprocessing"])
                            st.json(preprocessing_json)
                        except (json.JSONDecodeError, TypeError):
                            st.text(result["preprocessing"])

                if result.get("data_exploration"):
                    with st.expander("Data Exploration Summary"):
                        st.text(result["data_exploration"])

        except Exception as e:
            st.error(f"An error occurred in the pipeline:")
            st.exception(e)
            

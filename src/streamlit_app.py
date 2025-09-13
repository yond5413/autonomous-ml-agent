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

                # Display Leaderboard
                if "leaderboard" in result:
                    st.header("Leaderboard")
                    leaderboard_df = pd.DataFrame([
                        {
                            "Model": r["model_selection"]["recommended_model"],
                            "F1 Score": r["metrics"].get("f1_score"),
                            "R2 Score": r["metrics"].get("r2_score"),
                        }
                        for r in result["leaderboard"]
                    ])
                    st.dataframe(leaderboard_df)

                    for i, model_result in enumerate(result["leaderboard"]):
                        with st.expander(f"Details for {model_result["model_selection"]["recommended_model"]}", expanded=i==0):
                            st.subheader("Metrics")
                            st.json(model_result["metrics"])

                            if "feature_importances" in model_result and model_result["feature_importances"]:
                                st.subheader("Feature Importances")
                                feat_imp_df = pd.DataFrame(list(model_result["feature_importances"].items()), columns=["Feature", "Importance"])
                                st.bar_chart(feat_imp_df.set_index("Feature"))

                            if "summary" in model_result:
                                st.subheader("Summary")
                                st.markdown(model_result["summary"])
                
                # Display Deployment Information
                if "deployment" in result:
                    st.header("Deployment Information")
                    st.info(f"The best model has been deployed. You can find the deployment files in the `{result['deployment']['deployment_path']}` directory.")
                    with st.expander("Deployment README"):
                        with open(os.path.join(result['deployment']['deployment_path'], 'README.md'), 'r') as f:
                            st.markdown(f.read())

                if result.get("model_selection"):
                    with st.expander("Model Selection Details"):
                        st.json(result["model_selection"])
                
                if result.get("preprocessing"):
                    with st.expander("Preprocessing Result"):
                        st.json(result["preprocessing"])

                if result.get("data_exploration"):
                    with st.expander("Data Exploration Summary"):
                        st.text(result["data_exploration"])

        except Exception as e:
            st.error(f"An error occurred in the pipeline:")
            st.exception(e)
            


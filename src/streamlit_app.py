import streamlit as st
import pandas as pd
import requests, json   # or an internal RPC to orchestrator

st.title("AutoTabular — Preprocessing Explorer")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview", df.head())
    target = st.selectbox("Select target column", df.columns)
    if st.button("Run Preprocessing Agent"):
        # send dataframe+target to orchestrator (POST /orchestrate or via library)
        # For Phase 1 you can base64/csv->post to your local orchestrator
        st.info("Running preprocessing agent...")
        # show returned JSON/summary/plots in same UI

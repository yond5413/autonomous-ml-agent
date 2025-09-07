# Autonomous Tabular ML Agent

This project is an implementation of the Autonomous Tabular ML Agent as described in the PRD.

## Phase 1: Streamlit UI + Orchestrator Agent

The first phase of this project is to create a Streamlit front-end that allows a user to upload a CSV file, select a target column, and then kick off an orchestration loop. The orchestrator agent will then use a sandboxed preprocessing agent to explore the data and return a summary.

## Getting Started

1. **Install Poetry:**

   If you don't have Poetry installed, you can install it by following the official instructions:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install Dependencies:**

   Install the project dependencies using Poetry:
   ```bash
   poetry install
   ```

3. **Set E2B API Key:**

   This project uses the E2B code interpreter sandbox. You will need an E2B API key to run the application. You can get one from the [E2B website](https://e2b.dev/).

   Set the API key as an environment variable:
   ```bash
   export E2B_API_KEY="your_api_key"
   ```

4. **Run the Application:**

   Run the Streamlit application using Poetry:
   ```bash
   poetry run streamlit run src/streamlit_app.py
   ```

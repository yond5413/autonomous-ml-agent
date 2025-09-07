Got it ✅ — let’s narrow scope to Step 1: orchestrator + Streamlit front-end + preprocessing agent before we even think about training/ensembling. Here’s how I’d break it down:

Phase 1: Streamlit UI + Orchestrator Agent

Streamlit (entrypoint)

User uploads .csv

User selects target column from dropdown

Kick off orchestration loop

Flow:

Streamlit sends dataset → Orchestrator Agent.

Orchestrator Agent decides: “First, sandbox-preprocessing agent should explore data.”

Preprocessing Agent (low-reasoning LLM in sandbox) runs code like:

Load CSV, inspect shape, dtypes, missing %

Show descriptive stats, class distribution, correlations

Suggest preprocessing needs (encoding, imputation, scaling, leakage risk)

Orchestrator Agent receives summary → decides next: “Pass to model selection agent later.”

Agent Roles (initial scope)
1. Orchestrator Agent (high reasoning, runs outside sandbox)

Lives on OpenRouter (could use GPT-4.1/Claude Opus/Gemini 1.5 Pro depending on your budget).

Responsibilities:

Calls sandbox agents with clear instructions + code generation.

Interprets outputs/logs and decides next steps.

Keeps conversation-like state across pipeline (like a conductor).

2. Preprocessing Explorer Agent (low reasoning, runs inside sandbox)

Task: Write & run Python in sandbox → return structured JSON insights.

Example outputs:

{
  "n_rows": 32561,
  "n_cols": 15,
  "missing_values": {"age": 0, "workclass": 1836, "occupation": 1843},
  "categorical": ["workclass", "education", "marital_status"],
  "numerical": ["age", "hours_per_week"],
  "target_balance": {"<=50K": 24720, ">50K": 7841},
  "notes": "Target is imbalanced (3:1). Workclass/occupation missing ~6%."
}

3. (Later) Model Selection Agent (high reasoning)

Reads insights → proposes model candidates + HPO strategy.

Not needed for v0, but the Orchestrator should leave a “slot” to call it.

4. (Later) Preprocessing Transformer Agent (low reasoning)

Implements transformations (encoders, imputers, scalers).

Will follow after model selection in pipeline.

Data flow for v0 demo

User → Streamlit → Orchestrator → Sandbox → Orchestrator → Streamlit

Example:

User uploads adult.csv, selects income.

Orchestrator → Sandbox agent: “Run EDA and return JSON summary.”

Sandbox executes Python in container → returns JSON summary.

Orchestrator converts to readable summary + sends back to Streamlit UI.

Example: Streamlit Skeleton (upload + target selection)
import streamlit as st
import pandas as pd

st.title("Autonomous Tabular ML Agent")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview:", df.head())

    # Target selection
    target = st.selectbox("Select target variable", df.columns)

    if st.button("Run Preprocessing Agent"):
        # This would call your orchestrator agent
        st.write("Sending to orchestrator... (mock output)")
        st.json({
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "missing": df.isnull().sum().to_dict()
        })


For now this just fakes preprocessing, but you’ll wire it into orchestrator + sandbox next.

Why this is a good first step

Keeps things modular: Orchestrator just needs to forward dataset + target → sandbox agent.

Sandbox agent runs safe, limited EDA code (pandas, numpy, matplotlib if needed).

You immediately have a working demo in Streamlit: upload → target select → structured EDA report.

Leaves a clean extension point for model selection + preprocessing.

👉 Question for you:
Do you want me to draft the sandbox agent’s first EDA code template (the code it would run for exploration and return JSON), so you can plug it into your orchestrator right after the Streamlit step?

You said:
Please provide initial prd to get initial phase working
Streamlit app first phase, preprocessing agent steps broken into parts and potential feedback loop
ChatGPT said:
Thought for 7s
PRD — Phase 1: Streamlit front-end + Preprocessing (Exploration) Agent

Nice — we’ll keep this tightly scoped so you have a working demo fast. This PRD covers the Streamlit upload UI, the orchestrator → sandbox preprocessing explorer agent, the structured outputs, and the feedback/repair loop between the low-reasoning sandbox agent and the high-reasoning orchestrator.

1 — Goal (one-liner)

Provide a Streamlit-based entrypoint where a user uploads a CSV and selects a target column; a sandboxed preprocessing explorer agent runs automated EDA (like a data scientist), returns a structured JSON summary + human-friendly report, and the orchestrator decides whether to request more probing or move to model selection.

2 — Success criteria (what “done” looks like)

Streamlit accepts CSV, shows preview, lets user choose a target.

Clicking “Run Preprocessing” triggers the orchestrator which calls the sandbox explorer agent.

Sandbox returns a well-formed JSON EDA report plus optional visualization artifacts (PNG/mini-plots).

Streamlit shows both the JSON and a readable natural-language summary (LLM-generated).

System supports at least one feedback iteration: orchestrator can request deeper probing (e.g., sample imputations, correlation checks, leakage checks).

No external network calls inside sandbox; resource limits enforced.

3 — Scope (in) / Out-of-scope (out)

In-scope (Phase 1)

Streamlit file upload + target select UI

Orchestrator (controller) that calls sandbox agent

Sandbox explorer agent that runs EDA and returns JSON and artifacts

Basic feedback loop (orchestrator can request additional exploration / clarifications)

Deterministic logging of each run

Out-of-scope (for Phase 1)

Model training / HPO / ensembling

Full production deployment of exported models

Complex meta-learning warm starts (just record minimal run metadata)

4 — High-level architecture & dataflow

User (Streamlit) → Orchestrator (OpenRouter/high-reasoning LLM + controller) → Sandbox Explorer Agent (low-reasoning LLM + code execution environment e2b/Daytona) → returns JSON + artifacts → Orchestrator → Streamlit.

Key contract: the sandbox returns a JSON object (strict schema) and zero-or-more artifact files (PNG, CSV, small NPZ) saved in a known artifacts folder. Orchestrator interprets JSON and decides next step: accept, request more probing, or flag for repair.

5 — Streamlit UI: minimal wireframe & UX

Components:

Title & description

FileUploader (CSV)

Preview table (first N rows)

Target column selector (dropdown from df.columns)

Metric selector (optional for later; e.g., "accuracy/roc_auc/regression_mse")

Button: Run Preprocessing Agent

Results area:

Raw JSON (collapsible)

Human readable summary (LLM-generated)

Quick visualization gallery (histograms, missingness heatmap, class balance bar)

Action buttons: “Request Deeper EDA”, “Proceed to Model Selection” (disabled for Phase 1)

Sample Streamlit skeleton (plug-and-play):

# streamlit_app.py (starter)
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

6 — Sandbox Explorer Agent: responsibilities (broken into parts)

Part A — Ingestion & Sanity checks

Validate CSV parseability (encoding, delimiter), show row/col counts, sample schema.

Sanity flags: constant columns, duplicate rows, suspicious column names (email/id), extremely large file.

Part B — Column typing & profiling

Infer column types: numeric, categorical, datetime, text/freeform.

For each column: missing %; unique count (cardinality); example values; dtype suggested.

Part C — Target analysis

Determine task type: classification vs regression vs multilabel → based on target dtype/unique count.

For classification: class counts, class imbalance ratio, suggested class mapping.

For regression: distribution stats, skewness, outlier counts (IQR rule).

Part D — Univariate summaries

For numeric: mean, median, std, min/max, 5-number summary, skew/kurtosis.

For categorical: top-k values + frequencies; rare category %.

For datetime: range, frequency suggestions (e.g., daily/monthly), suggest new features.

Part E — Bivariate & correlation

Pearson/Spearman correlation between numeric features and target.

Cramér’s V or similar for categorical-cat relationships (optional).

Flag high-correlation pairs (> 0.9) for possible multicollinearity.

Part F — Missingness & patterns

Missing % per column, pattern analysis (is missing correlated with target?), list columns with >X% missing (threshold configurable).

Suggest imputation strategies (median/mean/mode/flag + predictive impute).

Part G — Leakage & data quality heuristics

Detect columns that may be target-leaks (near-perfect correlation with target, ID-like columns with many unique values but aligned with target).

Flag columns with suspicious naming (e.g., contains 'label', 'outcome', 'y', 'target' besides true target).

Part H — Quick visualizations (small, optional)

Histogram for numeric columns (sample top N).

Bar chart for categorical counts.

Missingness heatmap (small image).

Part I — Actionable suggestions / next steps

Recommended preprocessing steps per feature (impute/encode/scale/drop).

Recommended CV strategy baseline (stratified K-fold if classification, KFold if regression).

Suggested candidate model families (coarse; e.g., linear/tree/nn) — this is a small hint, not a final selection.

7 — Sandbox output schema (strict JSON)

Orchestrator expects exactly this JSON keys (example):

{
  "dataset": {
    "n_rows": 10000,
    "n_cols": 12,
    "columns": ["age","workclass","education","..."]
  },
  "columns_profile": {
    "age": {
      "dtype": "numeric",
      "n_missing": 12,
      "pct_missing": 0.12,
      "unique": 63,
      "summary": {"min": 17, "q1": 25, "median": 35, "q3": 45, "max": 90, "mean": 38.4, "std": 12.1},
      "suggested_action": "impute_median|scale_optional"
    },
    "...": {}
  },
  "target": {
    "name": "income",
    "task_type": "classification",
    "n_classes": 2,
    "class_counts": {"<=50K": 24720, ">50K": 7841},
    "imbalance_ratio": 3.15
  },
  "correlations": {
    "numeric_to_target": [{"feature": "hours_per_week", "pearson": 0.12}],
    "feature_feature_highcorr": [{"f1": "age", "f2": "years_worked", "pearson": 0.92}]
  },
  "flags": {
    "high_cardinality": ["occupation"],
    "high_missing": ["workclass"],
    "possible_leakage": ["final_settlement_amount"]
  },
  "visuals": {
    "histograms": ["artifacts/hist_age.png"],
    "missing_heatmap": "artifacts/missing_heatmap.png"
  },
  "notes": "target imbalanced, moderate missingness in workclass (~7%), recommend stratified CV and median imputation for numeric."
}


Also include a small meta object with run_id, script_version, runtime_ms, errors (if any).

8 — Orchestrator → Sandbox API contract (minimal)

Orchestrator calls sandbox with:

{
  "action": "explore",
  "inputs": {
    "data_path": "/sandbox/input/upload.csv",
    "target": "income",
    "max_columns_to_explore": 100,
    "plot_limit": 10,
    "missing_threshold_pct": 30
  },
  "constraints": {
    "cpu": 1,
    "memory_mb": 2048,
    "timeout_seconds": 120
  }
}


Sandbox must return one JSON described above and write listed artifacts to /sandbox/artifacts/<run_id>/.... All stdout must be structured JSON for easy parsing; only logs go to stderr or a structured log file.

9 — Feedback / repair loop (how iterations work)

Initial run: Orchestrator requests explore → sandbox returns report.

Orchestrator decision:

If flags.possible_leakage not empty OR high_missing columns > threshold OR target.task_type ambiguous → orchestrator issues follow-up tasks:

drill_missing (predictive imputation test)

leakage_check (train quick proxy model on suspected leak columns)

type_disambiguation (ask for sampling + user confirmation)

If output looks healthy: present summary to user with “Proceed” option.

Follow-up runs: Orchestrator crafts a targeted explore subtask (e.g., run predictive imputation experiment) with a tighter timeout and returns structured results.

Repair: If sandbox execution fails (exceptions, memory blowup), orchestrator:

Captures error JSON (traceback + captured stderr)

Calls LLM-based repair policy to generate a patched script or reduce workload (sample fewer columns)

Retries up to N attempts (configurable). If still failing → escalate to user with clear error + fallback suggestions.

Iteration policies (Phase 1):

Max follow-up iterations: configurable, default 3.

Max total runtime budget for exploration per dataset: enforced by orchestrator (sum of timeouts).

10 — Example sandbox EDA script (starter)

This is the code the sandbox agent should run (or generate and run). It prints a single JSON to stdout.

# sandbox_explore.py (simple)
import json, sys, os, pandas as pd, numpy as np
from pathlib import Path
run_id = os.environ.get("RUN_ID","run_123")

def profile(df, target):
    cols = df.columns.tolist()
    n_rows,n_cols = df.shape
    columns_profile = {}
    for c in cols:
        ser = df[c]
        n_missing = int(ser.isnull().sum())
        pct_missing = float(n_missing / n_rows * 100)
        unique = int(ser.nunique(dropna=True))
        dtype = str(ser.dtype)
        suggested = ""
        if pd.api.types.is_numeric_dtype(ser):
            s = ser.dropna()
            if s.empty:
                summary = {}
            else:
                summary = {"min": float(s.min()), "q1": float(s.quantile(0.25)),
                           "median": float(s.median()), "q3": float(s.quantile(0.75)),
                           "max": float(s.max()), "mean": float(s.mean()), "std": float(s.std())}
            suggested = "impute_median|scale_optional"
        elif pd.api.types.is_datetime64_any_dtype(ser):
            summary = {"min": str(ser.min()), "max": str(ser.max())}
            suggested = "datetime_expand"
        else:
            summary = {"top_values": ser.dropna().value_counts().head(5).to_dict()}
            if unique > 100:
                suggested = "high_cardinality|target_encoder"
            else:
                suggested = "impute_mode|onehot"
        columns_profile[c] = {
            "dtype": dtype, "n_missing": n_missing, "pct_missing": pct_missing,
            "unique": unique, "summary": summary, "suggested_action": suggested
        }
    return {"dataset": {"n_rows": n_rows, "n_cols": n_cols, "columns": cols},
            "columns_profile": columns_profile}

def main(path, target):
    df = pd.read_csv(path)
    report = profile(df, target)
    # target analysis
    t = df[target]
    if pd.api.types.is_numeric_dtype(t):
        report["target"] = {"name":target,"task_type":"regression",
                            "summary": {"mean":float(t.mean()), "std": float(t.std())}}
    else:
        counts = t.value_counts().to_dict()
        report["target"] = {"name":target,"task_type":"classification",
                            "n_classes": len(counts), "class_counts": counts,
                            "imbalance_ratio": max(counts.values())/min(counts.values()) if len(counts)>1 else None}
    run_meta = {"run_id": run_id, "script_version": "v0.1"}
    out = {"meta": run_meta, **report, "notes": "basic profiling complete"}
    print(json.dumps(out))
if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])


(Phase 1: keep scripts small, deterministic, and safe — no heavy ML libraries.)

11 — Prompts & orchestrator policies (starter)

Orchestrator → Sandbox prompt template

Task: explore tabular CSV and return JSON summary.
Inputs: /sandbox/input/upload.csv ; target: {target}
Constraints: No network. Max memory 2GB. Timeout 120s. Return single valid JSON to stdout matching schema: dataset, columns_profile, target, correlations, flags, visuals, notes, meta.
Do not print extra logs.


Orchestrator repair policy (short)

If JSON invalid: respond with "parse_error" + stderr; automatically reduce workload (limit columns to 20) and retry.

If columns > 200: ask for sampling (downsample rows or limit columns) — orchestrator will propose sample and ask user.

12 — Observability & logging

All runs log to an append-only runs_log.jsonl containing run_id, timestamp, user_id (if available), dataset_hash, runtime_ms, exit_code, errors.

Artifacts stored under artifacts/<run_id>/.

Streamlit shows a short run_id and a link to artifacts for debugging.

13 — Security & sandbox constraints

Sandbox must have no outbound network access.

File system access restricted to a working dir. No root-level operations.

Execution limits: CPU (1–2 core), RAM (2–4GB), timeout (configurable).

Orchestrator must sanitize user-provided file names.

All user CSVs should be scanned for extremely large size; if > threshold, orchestrator asks to subsample.

14 — Testing & QA (Phase 1)

Unit tests for JSON schema validation (sample good/bad payloads).

Integration test: sample datasets (UCI Adult, Titanic) run end-to-end: upload → orchestrator → sandbox → UI display.

Fault injection tests: corrupt CSV, huge cardinality, intentional code exception in sandbox to verify repair loop.

Security test: verify sandbox lacks outbound network (attempt curl should fail).

15 — Acceptance criteria & demo script (what to show)

Acceptance:

Upload known CSV; select target; click run; see:

Raw JSON that matches schema,

Human-readable summary (LLM-curated),

At least 2 visual artifacts (histogram + missingness).

Trigger a follow-up exploration (e.g., leakage check) and show updated report.

Observe logs with run_id in runs_log.jsonl.

Demo script (sequence of actions you run during demo):

Open Streamlit app, upload adult.csv.

Select income as target → Run Preprocessing.

Streamlit displays summary + visuals.

Click “Request deeper EDA” → orchestrator issues drill_missing → sandbox returns predictive-imputation test results → Streamlit updates.

Show runs_log.jsonl with run entries.

16 — MVP checklist (deliverables for Phase 1)

 Streamlit app with upload + target selector

 Orchestrator skeleton that receives upload and calls sandbox

 Sandbox execute environment (script runner w/ resource constraints)

 Basic explorer agent script that returns the JSON schema

 UI rendering of JSON + LLM-generated natural language summary

 Feedback loop: orchestrator can request 1–3 follow-up probes

 Logging (runs_log.jsonl) and artifact storage

17 — Risks & mitigations

Risk: sandbox runs out of memory/time on large datasets → Mitigation: orchestrator prechecks file size and forces sampling.

Risk: false leakage flags → Mitigation: include human-in-the-loop confirm step before dropping columns.

Risk: inconsistent JSON from sandbox → Mitigation: strict JSON schema validation and auto-repair by retrying with reduced scope.

Risk: LLM hallucination in human summary → Mitigation: summaries must be grounded using only keys in JSON (or show “confidence: low” if derived).

18 — Next steps (sequence — no time estimates)

Implement Streamlit upload/preview/target UI (skeleton).

Implement orchestrator endpoint that accepts CSV and target, writes to sandbox input dir, creates run_id.

Implement sandbox runner that executes the example sandbox_explore.py with provided constraints.

Render returned JSON + visuals in Streamlit (human summary generated by orchestrator LLM).

Add the simple feedback loop (orchestrator issues drill_missing if > X% missing).

Add tests and run sample datasets.

19 — Example quick prompts (copy/paste)

Orchestrator→LLM (generate summary for UI)

You are a summarizer. Input: {json_report}. Write a short (3–5 sentence) human-readable summary focusing on task type, class balance (if classification), top 3 flags (missingness, high_cardinality, possible leakage), and recommended next step. Use only facts contained in the JSON.


Orchestrator→Sandbox (execute)

Action: EXPLORE CSV
Data path: /sandbox/input/{file}
Target: {target}
Return: single JSON matching schema to stdout.
Constraints: no network, max 120s, memory 2GB.
If any error occurs, print { "meta": {"error": "<traceback>"} } and exit code non-zero.
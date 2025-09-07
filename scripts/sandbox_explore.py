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

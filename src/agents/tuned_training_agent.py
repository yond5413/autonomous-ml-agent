import os
import re
import json
from e2b_code_interpreter import Sandbox
from openai import OpenAI
from .base import Agent

class TunedTrainingAgent(Agent):
    def _snapshot_files(self, sandbox: Sandbox) -> list[str]:
        try:
            snap_exec = sandbox.run_code(
                "import os, json\n"
                "roots = [os.getcwd(), '/home/user', '/tmp']\n"
                "seen = set()\n"
                "files = []\n"
                "for root in roots:\n"
                "    try:\n"
                "        for base, _, fns in os.walk(root):\n"
                "            if base.startswith('/proc') or base.startswith('/sys'):\n"
                "                continue\n"
                "            for fn in fns:\n"
                "                p = os.path.join(base, fn)\n"
                "                if p not in seen:\n"
                "                    seen.add(p)\n"
                "                    files.append(p)\n"
                "    except Exception:\n"
                "        pass\n"
                "print(json.dumps(files))\n",
                timeout=120
            )
            stdout = getattr(snap_exec, "stdout", "") or getattr(snap_exec, "logs", "") or ""
            return json.loads(stdout) if stdout else []
        except Exception:
            return []

    def _get_execution_output(self, execution):
        """Extract stdout, stderr and structured error from an execution object safely."""
        stdout = getattr(execution, "stdout", "") or ""
        stderr = getattr(execution, "stderr", "") or ""
        err = getattr(execution, "error", None)

        # If stdout/stderr are not directly available, try parsing from logs
        if not stdout and not stderr:
            logs = getattr(execution, "logs", None)
            if isinstance(logs, (list, tuple)):
                stdout_parts = []
                stderr_parts = []
                for log_entry in logs:
                    if hasattr(log_entry, "stdout") and log_entry.stdout:
                        stdout_parts.append(str(log_entry.stdout))
                    if hasattr(log_entry, "stderr") and log_entry.stderr:
                        stderr_parts.append(str(log_entry.stderr))
                    elif isinstance(log_entry, str):
                        stdout_parts.append(log_entry)
                    elif isinstance(log_entry, dict):
                        for key in ("stdout", "text", "output", "message"):
                            val = log_entry.get(key)
                            if isinstance(val, str) and val.strip():
                                stdout_parts.append(val)
                        for key in ("stderr", "error"):
                            val = log_entry.get(key)
                            if isinstance(val, str) and val.strip():
                                stderr_parts.append(val)

                stdout = "\n".join(stdout_parts).strip()
                stderr = "\n".join(stderr_parts).strip()
            elif isinstance(logs, str):
                stdout = logs.strip()
        
        # Try additional attributes that might contain output
        if not stdout:
            for attr in ['results', 'output', 'text']:
                val = getattr(execution, attr, None)
                if isinstance(val, str) and val.strip():
                    stdout = val.strip()
                    break
                elif isinstance(val, list) and val:
                    # Join list items if it's a list of strings
                    try:
                        stdout = "\n".join(str(item) for item in val).strip()
                        if stdout:
                            break
                    except:
                        continue

        return stdout, stderr, err

    def _extract_json_from_text(self, text: str):
        if not text:
            return None
        
        # First try: extract JSON from the last line (most likely location)
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue
        
        # Second try: find candidate JSON objects in the entire text
        candidates = re.findall(r'\{[\s\S]*?\}', text, re.DOTALL)
        for candidate in reversed(candidates):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None
    def execute(self, model_selection: dict, preprocessing_output: dict, tuning_results: dict, **kwargs):
        # 1. Extract necessary inputs
        preprocessing_sandbox = preprocessing_output["sandbox"]
        remote_paths = preprocessing_output.get("remote_paths", {})
        download_urls = preprocessing_output.get("download_urls", {})
        local_paths = preprocessing_output.get("local_dataset_paths", {})
        model_name = model_selection.get("recommended_model")
        task_type = model_selection.get("task_type")
        best_params = tuning_results.get("best_params")

        # Create a new sandbox for tuned training to avoid file path conflicts
        sandbox = Sandbox.create()
        
        # Download and upload required files to the new sandbox
        def download_and_upload_file(key: str) -> str | None:
            # Try local paths first (already downloaded)
            local_path = local_paths.get(key)
            if local_path and os.path.exists(local_path):
                try:
                    with open(local_path, "rb") as f:
                        remote_file = sandbox.files.write(f"{key}.csv", f)
                    return remote_file.path
                except Exception as e:
                    print(f"Failed to upload local file {local_path}: {e}")
            
            # Try downloading from preprocessing sandbox
            remote_path = remote_paths.get(key)
            if remote_path and isinstance(remote_path, str):
                try:
                    file_bytes = preprocessing_sandbox.files.read(remote_path)
                    if isinstance(file_bytes, str):
                        file_bytes = file_bytes.encode('utf-8')
                    
                    filename = f"{key}.csv" if key != "preprocessor" else "preprocessor.joblib"
                    remote_file = sandbox.files.write(filename, file_bytes)
                    return remote_file.path
                except Exception as e:
                    print(f"Failed to download and upload file {key} from {remote_path}: {e}")
            
            # Try download URLs as last resort
            download_url = download_urls.get(key)
            if download_url and isinstance(download_url, str) and download_url.startswith("http"):
                try:
                    import requests
                    response = requests.get(download_url, timeout=300)
                    response.raise_for_status()
                    
                    filename = f"{key}.csv" if key != "preprocessor" else "preprocessor.joblib"  
                    remote_file = sandbox.files.write(filename, response.content)
                    return remote_file.path
                except Exception as e:
                    print(f"Failed to download file {key} from URL {download_url}: {e}")
            
            return None

        x_train_path = download_and_upload_file("X_train")
        y_train_path = download_and_upload_file("y_train")
        x_test_path = download_and_upload_file("X_test")
        y_test_path = download_and_upload_file("y_test")
        x_val_path = download_and_upload_file("X_val")
        y_val_path = download_and_upload_file("y_val")

        # Validate that required files were uploaded successfully
        if not x_train_path or not y_train_path or not x_test_path or not y_test_path:
            sandbox.kill()
            missing = []
            if not x_train_path: missing.append("X_train")
            if not y_train_path: missing.append("y_train") 
            if not x_test_path: missing.append("X_test")
            if not y_test_path: missing.append("y_test")
            raise Exception(f"Failed to upload required training files: {missing}")

        val_lines = ""
        if x_val_path and y_val_path:
            val_lines = f"\n            - X_val_path: {x_val_path}\n            - y_val_path: {y_val_path}"

        prompt = f"""
        You are an expert data scientist. Write a complete Python script to train the final optimized model using the provided best hyperparameters, then evaluate it.

        CONTEXT
        - Model to be trained: {model_name}
        - Task Type: {task_type}
        - Best Hyperparameters (must be used exactly when instantiating the model):
          ```
          {best_params}
          ```
        - Datasets (absolute paths inside the environment):
          - X_train_path: {x_train_path}
          - y_train_path: {y_train_path}
          - X_test_path: {x_test_path}
          - y_test_path: {y_test_path}{val_lines}

        CRITICAL INSTRUCTIONS
        1) Data loading: At the top, assign file path variables exactly as provided. Load with pandas. Ensure aligned indices (reset_index(drop=True)) and 1D target via .squeeze().
        2) Validation handling: If X_val and y_val are provided, combine with training (concatenate rows) before fitting the final model; otherwise train only on training data.
        3) Metrics:
           - Classification: accuracy, precision, recall, f1_score.
           - Regression: mse, mae, r2.
        4) Feature importances: If available (e.g., tree-based models), compute and include as a dict of feature_name -> importance. If feature names are unknown, return indices as strings.
        5) Saving: Save to tuned_model.pickle via pickle; compute absolute path dynamically with os.path.abspath.
        6) Output format: Print a single JSON object to stdout with this exact schema and nothing else after it:
           {{
             "metrics": {{ ... }},
             "feature_importances": {{ "<name>": <float>, ... }} | null,
             "model_path": "/abs/path/to/tuned_model.pickle"
           }}

        **CRITICAL FORMATTING RULES:**
        7) ABSOLUTELY NO F-STRINGS: Do not use f"..." syntax anywhere in your code.
        8) NO STRING FORMATTING: Do not use .format() method or % formatting.
        9) NO CURLY BRACES IN STRINGS: Avoid {{ }} in any string literals except in the final JSON structure.
        10) For metrics calculation: First compute each metric value, assign to variables, then build the dict:
            ```
            accuracy_val = accuracy_score(y_test, y_pred)  
            metrics = {{"accuracy": accuracy_val}}
            ```
        11) SILENCE ALL WARNINGS: Add these lines at the top after imports:
            import warnings
            warnings.filterwarnings('ignore')
            import os
            os.environ['PYTHONWARNINGS'] = 'ignore'
        12) JSON SERIALIZATION: Convert numpy types to Python native types before JSON serialization:
            def convert_numpy_types(obj):
                import numpy as np
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
        13) No other prints than the final JSON.
        14) Reply ONLY with the full Python code inside a single ```python code block.

        EXAMPLE STRUCTURE (illustrative)
        ```python
        import os, json, pickle, pandas as pd
        import warnings, numpy as np
        from sklearn.metrics import accuracy_score
        # from sklearn... import Model
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        os.environ['PYTHONWARNINGS'] = 'ignore'
        
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # best_params = {...}
        # instantiate model = Model(**best_params)
        # fit, predict, compute metrics
        out_path = os.path.abspath('tuned_model.pickle')
        with open(out_path, 'wb') as f:
            pickle.dump(model, f)
        result = {"metrics": convert_numpy_types(metrics), "feature_importances": convert_numpy_types(feat_imp_or_null), "model_path": out_path}
        print(json.dumps(result))
        ```

        Now produce the complete runnable script.
        """

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        
        messages = [
            {"role": "system", "content": "You are a data engineering expert who writes and debugs Python scripts for model training."},
            {"role": "user", "content": prompt}
        ]
        
        max_retries = 3
        models = ["qwen/qwen-2.5-coder-32b-instruct:free", "agentica-org/deepcoder-14b-preview:free", "openai/gpt-oss-20b:free"]
        for attempt in range(max_retries):
            # Use different models for different attempts to increase success rate
            model_to_use = models[attempt % len(models)]
            
            response = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=2500
            )
            
            generated_text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": generated_text})

            code_match = re.search(r'```(?:python)?\s*(.*?)```', generated_text, re.DOTALL)
            
            if not code_match:
                print("\n[" + self.__class__.__name__ + "] Attempt " + str(attempt + 1) + "/" + str(max_retries) + " failed: LLM did not return a Python code block.")
                feedback = "Your response did not contain a Python code block. Please generate only the Python code, enclosed in a ```python ... ``` block."
                messages.append({"role": "user", "content": feedback})
                if attempt == max_retries - 1:
                    raise Exception("No code found in response after multiple attempts.")
                continue

            code = code_match.group(1).strip()

            try:
                # Prevent sandbox expiration during long final training
                try:
                    sandbox.set_timeout(3600)
                except Exception:
                    pass

                before_files = self._snapshot_files(sandbox)
                execution = sandbox.run_code(code, timeout=1800) # 30 minute timeout for training

                stdout, stderr, err = self._get_execution_output(execution)

                if err:
                    error_details = (
                        "Error Name: " + str(getattr(err, 'name', 'ExecutionError')) + "\n" +
                        "Error Value: " + str(getattr(err, 'value', str(err))) + "\n" +
                        "Traceback:\n" + str(getattr(err, 'traceback', ''))
                    )
                    if stdout:
                        error_details += "\n--- stdout ---\n" + str(stdout)
                    if stderr:
                        error_details += "\n--- stderr ---\n" + str(stderr)
                    print("\n[" + self.__class__.__name__ + "] Attempt " + str(attempt + 1) + "/" + str(max_retries) + " failed: Code execution error.")
                    print(error_details)

                    feedback = (
                        "Your code failed to execute. Here is the error:\n\n" +
                        str(error_details) + "\n\n" +
                        "Please fix the code and provide the corrected full script."
                    )
                    messages.append({"role": "user", "content": feedback})
                    if attempt == max_retries - 1:
                        # Clean up sandbox before re-raising
                        try:
                            sandbox.kill()
                        except Exception:
                            pass
                        raise Exception("Code execution failed after " + str(max_retries) + " attempts. Last error: " + str(getattr(err, 'value', str(err))))
                    continue

                # Success
                combined = (stdout or "") + ("\n" + stderr if stderr else "")
                training_results = None
                if stdout:
                    try:
                        training_results = json.loads(stdout)
                    except Exception:
                        training_results = self._extract_json_from_text(combined)
                else:
                    training_results = self._extract_json_from_text(combined)

                if not training_results:
                    after_files = self._snapshot_files(sandbox)
                    created = [p for p in after_files if p not in set(before_files)] if after_files else []
                    model_candidates = [p for p in created if any(p.lower().endswith(ext) for ext in [".pickle", ".pkl", ".joblib"]) and ("model" in p.lower() or "tuned" in p.lower())]
                    model_path = model_candidates[0] if model_candidates else None
                    if not model_path:
                        common_path = "/home/user/tuned_model.pickle"
                        if common_path in (after_files or []):
                            model_path = common_path
                    if model_path:
                        training_results = {"metrics": {}, "model_path": model_path}
                    else:
                        raise Exception("The script ran but produced no output.")
                
                model_path = training_results.get("model_path")
                if model_path:
                    training_results["model_download_url"] = sandbox.download_url(model_path, timeout=600)

                # Generate summary
                summary_prompt = self._create_summary_prompt(model_selection, tuning_results, training_results)
                summary_response = client.chat.completions.create(
                    model="qwen/qwen3-coder:free",
                    messages=[
                        {"role": "system", "content": "You are a machine learning expert who explains model results in an easy-to-understand way."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_tokens=1024,
                )
                summary = summary_response.choices[0].message.content
                training_results["summary"] = summary

                # Clean up sandbox before returning
                try:
                    sandbox.kill()
                except Exception:
                    pass
                    
                return training_results

            except Exception as e:
                print("\n[" + self.__class__.__name__ + "] Attempt " + str(attempt + 1) + "/" + str(max_retries) + " failed: An unexpected exception occurred.")
                print("Exception: " + str(e))
                feedback = "An unexpected error occurred during execution: " + str(e)
                messages.append({"role": "user", "content": feedback})
                if attempt == max_retries - 1:
                    # Clean up sandbox before re-raising
                    try:
                        sandbox.kill()
                    except Exception:
                        pass
                    raise e
        
        # Clean up sandbox on failure
        try:
            sandbox.kill()
        except Exception:
            pass
        raise Exception("Failed to generate and execute valid code after multiple attempts.")

    def _create_summary_prompt(self, model_selection, tuning_results, training_results):
        model_name = model_selection.get("recommended_model")
        task_type = model_selection.get("task_type")
        best_params = tuning_results.get("best_params")
        metrics = training_results.get("metrics")
        feature_importances = training_results.get("feature_importances")

        prompt = f"""
        Please provide a summary of the machine learning model that was trained.

        Here is the information about the model:
        - **Model:** {model_name}
        - **Task Type:** {task_type}
        - **Best Hyperparameters:** {best_params}
        - **Evaluation Metrics:** {metrics}
        - **Feature Importances:** {feature_importances}

        Based on this information, please provide a concise summary that covers:
        1. An overview of the model and its performance.
        2. An interpretation of the evaluation metrics.
        3. An explanation of the most important features (if available).
        
        The summary should be easy to understand for a non-technical audience.
        """
        return prompt

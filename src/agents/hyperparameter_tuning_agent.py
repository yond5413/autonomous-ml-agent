import os
import re
import json
from e2b_code_interpreter import Sandbox
from openai import OpenAI
from .base import Agent

class HyperparameterTuningAgent(Agent):
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
    def execute(self, model_selection: dict, preprocessing_output: dict, **kwargs):
        # 1. Extract necessary inputs
        preprocessing_sandbox = preprocessing_output["sandbox"]
        remote_paths = preprocessing_output.get("remote_paths", {})
        download_urls = preprocessing_output.get("download_urls", {})
        local_paths = preprocessing_output.get("local_dataset_paths", {})
        model_name = model_selection.get("recommended_model")
        task_type = model_selection.get("task_type")

        # Create a new sandbox for hyperparameter tuning to avoid file path conflicts
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
                    
                    filename = key + ".csv" if key != "preprocessor" else "preprocessor.joblib"
                    remote_file = sandbox.files.write(filename, file_bytes)
                    return remote_file.path
                except Exception as e:
                    print("Failed to download and upload file " + str(key) + " from " + str(remote_path) + ": " + str(e))
            
            # Try download URLs as last resort
            download_url = download_urls.get(key)
            if download_url and isinstance(download_url, str) and download_url.startswith("http"):
                try:
                    import requests
                    response = requests.get(download_url, timeout=300)
                    response.raise_for_status()
                    
                    filename = key + ".csv" if key != "preprocessor" else "preprocessor.joblib"  
                    remote_file = sandbox.files.write(filename, response.content)
                    return remote_file.path
                except Exception as e:
                    print("Failed to download file " + str(key) + " from URL " + str(download_url) + ": " + str(e))
            
            return None

        x_train_path = download_and_upload_file("X_train")
        y_train_path = download_and_upload_file("y_train")

        # Validate that required files were uploaded successfully
        if not x_train_path or not y_train_path:
            sandbox.kill()
            missing = []
            if not x_train_path: missing.append("X_train")
            if not y_train_path: missing.append("y_train") 
            raise Exception("Failed to upload required training files: " + str(missing))

        prompt = ("You are an expert data scientist. Write a complete Python script to tune hyperparameters for the specified model using scikit-learn's RandomizedSearchCV.\n\n" +
                 "CONTEXT\n" +
                 "- Model to be tuned: " + str(model_name) + "\n" +
                 "- Task Type: " + str(task_type) + "\n" +
                 "- Data Exploration Summary:\n```\n" +
                 str(preprocessing_output["exploration_summary"]) + "\n```\n" +
                 "- Datasets (absolute paths inside the environment):\n" +
                 "  - X_train_path: " + str(x_train_path) + "\n" +
                 "  - y_train_path: " + str(y_train_path) + "\n\n" +
                 "CRITICAL INSTRUCTIONS\n" +
                 "1) Library: Use ONLY scikit-learn's RandomizedSearchCV (no Optuna or external tuners).\n" +
                 "2) Reproducibility: Set random_state=42 wherever applicable.\n" +
                 "3) Performance: Use n_jobs=-1 where supported. Keep n_iter reasonable (e.g., 25-50) based on model complexity.\n" +
                 "4) Data loading: At the top, assign file path variables exactly as provided (e.g., X_train_path = r\"" + str(x_train_path) + "\"). Load with pandas. Ensure X and y have aligned indices (reset_index(drop=True)) and y is 1D (use .squeeze()).\n" +
                 "5) Parameter space: Define a sensible search space for " + str(model_name) + ", informed by task type and typical ranges.\n" +
                 "6) Output format: Print a single JSON object to stdout with this exact schema and nothing else after it:\n" +
                 "   {\n" +
                 "     \"best_params\": { \"<param>\": <value>, ... }\n" +
                 "   }\n" +
                 "   CRITICAL: Convert numpy types to Python native types before JSON serialization to avoid 'Object of type int64 is not JSON serializable' errors.\n" +
                 "7) CRITICAL FORMATTING RULES - ABSOLUTELY NO F-STRINGS:\n" +
                 "   - Do not use f\"...\" syntax anywhere in your code\n" +
                 "   - Do not use .format() method or % formatting\n" +
                 "   - Do not use curly braces in string literals except for JSON output\n" +
                 "   - If you need to print debug information, use regular print statements with string concatenation only\n" +
                 "8) CRITICAL OUTPUT REQUIREMENT: The script MUST end with exactly one print statement that outputs the JSON. No other print statements should come after it.\n" +
                 "9) Do not print logs, warnings, or any other text after the final JSON output.\n" +
                 "10) SILENCE ALL WARNINGS: Add these lines at the top after imports to suppress warnings:\n" +
                 "    import warnings\n" +
                 "    warnings.filterwarnings('ignore')\n" +
                 "    import os\n" +
                 "    os.environ['PYTHONWARNINGS'] = 'ignore'\n" +
                 "11) Reply ONLY with the full Python code inside a single ```python code block.\n\n" +
                 "EXAMPLE STRUCTURE (illustrative)\n" +
                 "```python\n" +
                 "import json, pandas as pd\n" +
                 "import numpy as np\n" +
                 "import warnings\n" +
                 "import os\n" +
                 "from sklearn.model_selection import RandomizedSearchCV\n" +
                 "# import model class ...\n\n" +
                 "# Suppress all warnings to ensure clean JSON output\n" +
                 "warnings.filterwarnings('ignore')\n" +
                 "os.environ['PYTHONWARNINGS'] = 'ignore'\n\n" +
                 "def convert_numpy_types(obj):\n" +
                 "    if isinstance(obj, dict):\n" +
                 "        return {k: convert_numpy_types(v) for k, v in obj.items()}\n" +
                 "    elif isinstance(obj, list):\n" +
                 "        return [convert_numpy_types(v) for v in obj]\n" +
                 "    elif isinstance(obj, (np.integer, np.int64, np.int32)):\n" +
                 "        return int(obj)\n" +
                 "    elif isinstance(obj, (np.floating, np.float64, np.float32)):\n" +
                 "        return float(obj)\n" +
                 "    elif isinstance(obj, np.ndarray):\n" +
                 "        return obj.tolist()\n" +
                 "    else:\n" +
                 "        return obj\n\n" +
                 "X_train_path = r\"" + str(x_train_path) + "\"\n" +
                 "y_train_path = r\"" + str(y_train_path) + "\"\n" +
                 "X = pd.read_csv(X_train_path).reset_index(drop=True)\n" +
                 "y = pd.read_csv(y_train_path).squeeze().reset_index(drop=True)\n\n" +
                 "# define model = ...\n" +
                 "# define param_distributions = { ... }\n" +
                 "search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=30, cv=5, n_jobs=-1, random_state=42, scoring=None)\n" +
                 "search.fit(X, y)\n" +
                 "best_params_converted = convert_numpy_types(search.best_params_)\n" +
                 "print(json.dumps({\"best_params\": best_params_converted}))\n" +
                 "```\n\n" +
                 "Now produce the complete runnable script.")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        
        messages = [
            {"role": "system", "content": "You are a data engineering expert who writes and debugs Python scripts for hyperparameter tuning."},
            {"role": "user", "content": prompt}
        ]
        
        # Ensure the sandbox does not expire during tuning (extend TTL)
        try:
            sandbox.set_timeout(3600)
        except Exception:
            pass

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
                execution = sandbox.run_code(code, timeout=1200)  # 20 minute timeout for tuning

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
                        try:
                            sandbox.kill()
                        except Exception:
                            pass
                        raise Exception(
                            "Code execution failed after " + str(max_retries) + " attempts. Last error: " + str(getattr(err, 'value', str(err)))
                        )
                    continue

                # Success: parse JSON from any available stream/logs
                combined = (stdout or "") + ("\n" + stderr if stderr else "")
                tuning_results = None
                if stdout:
                    try:
                        tuning_results = json.loads(stdout)
                    except Exception:
                        tuning_results = self._extract_json_from_text(combined)
                else:
                    tuning_results = self._extract_json_from_text(combined)

                if not tuning_results or not isinstance(tuning_results, dict):
                    # Add debug information to help understand what was captured
                    debug_info = "Script output debugging:\n"
                    debug_info += "- stdout length: " + str(len(stdout or "")) + "\n"
                    debug_info += "- stderr length: " + str(len(stderr or "")) + "\n" 
                    debug_info += "- stdout content: " + repr(stdout) + "\n"
                    debug_info += "- stderr content: " + repr(stderr) + "\n"
                    debug_info += "- combined: " + repr(combined) + "\n"
                    debug_info += "- tuning_results: " + repr(tuning_results) + "\n"
                    debug_info += "- execution object type: " + str(type(execution)) + "\n"
                    debug_info += "- execution attributes: " + str(dir(execution))
                    print(debug_info)
                    
                    feedback = ("Your script did not produce parseable JSON output. The output must be a single JSON object on stdout.\n\n" +
                              "Debug information:\n" +
                              "- stdout length: " + str(len(stdout or "")) + "\n" +
                              "- stderr length: " + str(len(stderr or "")) + "\n" +
                              "Output received:\n" + str(combined) + "\n\n" +
                              "REQUIREMENTS:\n" +
                              "1. Your script MUST end with exactly this pattern:\n" +
                              "   best_params_converted = convert_numpy_types(search.best_params_)\n" +
                              "   print(json.dumps({\"best_params\": best_params_converted}))\n\n" +
                              "2. Add warnings.filterwarnings('ignore') at the top to suppress all output\n" +
                              "3. Do NOT print anything else after the JSON output\n" +
                              "4. Make sure the script runs without errors\n\n" +
                              "Expected output format: {\"best_params\": {\"param_name\": value, ...}}")
                    messages.append({"role": "user", "content": feedback})
                    if attempt == max_retries - 1:
                        raise Exception("The script ran but produced no parseable JSON output.")
                    continue

                # Clean up sandbox before returning
                try:
                    sandbox.kill()
                except Exception:
                    pass

                return tuning_results

            except Exception as e:
                print("\n[" + self.__class__.__name__ + "] Attempt " + str(attempt + 1) + "/" + str(max_retries) + " failed: An unexpected exception occurred.")
                print("Exception: " + str(e))
                feedback = "An unexpected error occurred during execution: " + str(e)
                messages.append({"role": "user", "content": feedback})
                if attempt == max_retries - 1:
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

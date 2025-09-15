import os
import re
import json
from e2b_code_interpreter import Sandbox
from openai import OpenAI
from .base import Agent

class SimpleTrainingAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_error_context = None
    def _build_error_feedback_prompt(self, error_context: dict, original_prompt: str) -> str:
        error_type = error_context.get("error_type", "UnknownError")
        error_message = error_context.get("error_message", "")
        line_number = error_context.get("line_number", "Unknown")
        problematic_code = error_context.get("problematic_code", "")
        suggested_fixes = error_context.get("suggested_fixes", [])
        stdout = error_context.get("stdout", "")
        stderr = error_context.get("stderr", "")

        fixes_block = "\n".join("- " + str(fix) for fix in suggested_fixes) if suggested_fixes else "- Review recent changes around the reported line\n- Ensure balanced quotes, brackets, parentheses\n- Check indentation and block structure"
        stdout_block = ("\n**Stdout (context):**\n```\n" + str(stdout) + "\n```") if stdout else ""
        stderr_block = ("\n**Stderr (context):**\n```\n" + str(stderr) + "\n```") if stderr else ""

        return ("**PREVIOUS TRAINING ATTEMPT FAILED**\n\n" +
                "**Error Details:**\n" +
                "- Type: " + str(error_type) + "\n" +
                "- Message: " + str(error_message) + "\n" +
                "- Line: " + str(line_number) + "\n\n" +
                "**Problematic Code Section:**\n" +
                "```\n" + str(problematic_code) + "\n```\n\n" +
                "**Suggested Fixes:**\n" + str(fixes_block) + 
                str(stdout_block) + str(stderr_block) + "\n\n" +
                "**Original Requirements:**\n" + str(original_prompt) + "\n\n" +
                "**TASK:**\n" +
                "Provide a corrected, complete Python script that adheres to the original requirements and addresses the error above. Reply ONLY with the full Python code inside a single markdown code block.")

    def _generate_suggested_fixes(self, error_type: str, error_message: str) -> list:
        fixes = []
        msg = (error_message or "").lower()
        if "indentationerror" in error_type.lower():
            fixes += [
                "Fix indentation levels consistently (spaces only)",
                "Ensure each control statement has a properly indented block",
            ]
        if "invalid format specifier" in msg or "format specifier" in msg:
            fixes += [
                "COMPLETELY REMOVE all f-strings (f'...') from your code",
                "Replace f-strings with simple string concatenation using + operator",
                "For metrics: accuracy_val = accuracy_score(y_test, y_pred); metrics = {'accuracy': accuracy_val}",
                "Never use curly braces {} inside strings except in final JSON structure",
                "Avoid .format() method and % formatting completely",
            ]
        if "valueerror" in error_type.lower() and "feature names" in msg:
            fixes += [
                "Use the same preprocessor used during fit to transform X_test",
                "Avoid manual column engineering mismatch; load preprocessed CSVs",
                "Ensure X_train and X_test have identical columns and order",
            ]
        if "keyerror" in error_type.lower() or "not found in axis" in msg:
            fixes.append("Verify all column names exist before selection")
        if not fixes:
            fixes += [
                "Verify variable names and imports",
                "Check function arguments and return values",
                "Add guards for missing files/paths",
            ]
        return fixes

    def _create_detailed_error_context(self, code: str, error: Exception, stdout: str = "", stderr: str = "") -> dict:
        error_type = type(error).__name__
        error_message = getattr(error, "value", None) or str(error)
        line_number = None
        if hasattr(error, 'lineno') and getattr(error, 'lineno'):
            line_number = getattr(error, 'lineno')
        elif 'line' in str(error).lower():
            try:
                m = re.search(r'line (\d+)', str(error))
                if m:
                    line_number = int(m.group(1))
            except Exception:
                line_number = None

        problematic_code = ""
        if code and line_number:
            lines = code.split('\n')
            start = max(0, line_number - 3)
            end = min(len(lines), line_number + 3)
            snippet = []
            for i in range(start, end):
                prefix = str(i+1) + ": "
                if i + 1 == line_number:
                    snippet.append(prefix + ">>> " + lines[i] + " <<< ERROR HERE")
                else:
                    snippet.append(prefix + lines[i])
            problematic_code = "\n".join(snippet)

        suggested_fixes = self._generate_suggested_fixes(error_type, error_message)
        return {
            "error_type": error_type,
            "error_message": error_message,
            "line_number": line_number,
            "problematic_code": problematic_code,
            "full_code": code,
            "stdout": stdout,
            "stderr": stderr,
            "suggested_fixes": suggested_fixes,
        }
    def _snapshot_files(self, sandbox: Sandbox) -> list[str]:
        try:
            snap_exec = sandbox.run_code(
                "import os, json\n"
                "roots = [os.getcwd(), '/home/user', '/tmp', '/tmp/autonomous-ml-agent']\n"
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

    def _validate_code_for_fstring_issues(self, code: str) -> tuple[bool, str]:
        """Check for potential f-string and formatting issues in generated code."""
        issues = []
        
        # Check for f-strings
        if re.search(r'f["\']', code):
            issues.append("Code contains f-strings (f'...' or f\"...\") which are prohibited")
        
        # Check for .format() calls
        if '.format(' in code:
            issues.append("Code contains .format() calls which should be avoided")
        
        # Check for % formatting
        if re.search(r'%\s*[sdif]', code):
            issues.append("Code contains % string formatting which should be avoided")
        
        # Check for suspicious curly braces in strings that aren't JSON
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Skip lines that are clearly JSON construction
            if 'json.dumps' in line or '"metrics"' in line or '"model_path"' in line:
                continue
            # Look for curly braces in string literals
            if re.search(r'["\'][^"\']*\{[^}]*\}[^"\']*["\']', line):
                issues.append("Line " + str(i+1) + " contains suspicious curly braces in string: " + line.strip())
        
        if issues:
            return False, "; ".join(issues)
        return True, ""
    def execute(self, model_selection: dict, preprocessing_output: dict, **kwargs):
        # 1. Extract necessary inputs
        preprocessing_sandbox = preprocessing_output["sandbox"]
        remote_paths = preprocessing_output.get("remote_paths", {})
        download_urls = preprocessing_output.get("download_urls", {})
        local_paths = preprocessing_output.get("local_dataset_paths", {})
        model_name = model_selection.get("recommended_model")
        task_type = model_selection.get("task_type")

        # Create a new sandbox for training to avoid file path conflicts
        sandbox = Sandbox.create()
        
        # Download and upload required files to the new sandbox
        def download_and_upload_file(key: str) -> str | None:
            # Try local paths first (already downloaded)
            local_path = local_paths.get(key)
            if local_path and os.path.exists(local_path):
                try:
                    with open(local_path, "rb") as f:
                        remote_file = sandbox.files.write(key + ".csv", f)
                    return remote_file.path
                except Exception as e:
                    print("Failed to upload local file " + str(local_path) + ": " + str(e))
            
            # Try downloading from preprocessing sandbox
            remote_path = remote_paths.get(key)
            if remote_path and isinstance(remote_path, str):
                try:
                    file_bytes = preprocessing_sandbox.files.read(remote_path)
                    if isinstance(file_bytes, str):
                        file_bytes = file_bytes.encode('utf-8')
                    
                    filename = (key + ".csv") if key != "preprocessor" else "preprocessor.joblib"
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
                    
                    filename = (key + ".csv") if key != "preprocessor" else "preprocessor.joblib"  
                    remote_file = sandbox.files.write(filename, response.content)
                    return remote_file.path
                except Exception as e:
                    print("Failed to download file " + str(key) + " from URL " + str(download_url) + ": " + str(e))
            
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
            raise Exception("Failed to upload required training files: " + str(missing))

        val_lines = ""
        if x_val_path and y_val_path:
            val_lines = "\n    - X_val_path: " + str(x_val_path) + "\n    - y_val_path: " + str(y_val_path)

        prompt = ("You are an expert data scientist. Write a complete Python script to train and evaluate the specified model.\n\n" +
                 "CONTEXT\n" +
                 "- Model to be trained: " + str(model_name) + "\n" +
                 "- Task Type: " + str(task_type) + "\n" +
                 "- Datasets (absolute paths):\n" +
                 "  - X_train_path: " + str(x_train_path) + "\n" +
                 "  - y_train_path: " + str(y_train_path) + "\n" +
                 "  - X_test_path: " + str(x_test_path) + "\n" +
                 "  - y_test_path: " + str(y_test_path) + str(val_lines) + "\n\n" +
                 "CRITICAL INSTRUCTIONS\n" +
                 "1) Data loading: At the top, assign file path variables exactly as provided (e.g., X_train_path = r\"" + str(x_train_path) + "\"). Load with pandas. Ensure X and y indices are aligned with reset_index(drop=True) and y is 1D with .squeeze().\n" +
                 "2) Validation handling: If both X_val_path and y_val_path are provided, load them and optionally report validation metrics, but the primary evaluation must be on test data. If validation is missing, proceed without it.\n" +
                 "3) Metrics:\n" +
                 "   - Classification: accuracy, precision, recall, f1_score.\n" +
                 "   - Regression: mse, mae, r2.\n" +
                 "4) Saving: Save the trained model as trained_model.pickle using pickle. Compute the absolute path dynamically with os.path.abspath.\n" +
                 "5) Output format: Print a single JSON object to stdout with this exact schema and nothing else after it:\n" +
                 "   {\n" +
                 "     \"metrics\": { ... },\n" +
                 "     \"model_path\": \"/abs/path/to/trained_model.pickle\"\n" +
                 "   }\n\n" +
                 "**CRITICAL FORMATTING RULES:**\n" +
                 "6) ABSOLUTELY NO F-STRINGS: Do not use f\"...\" syntax anywhere in your code.\n" +
                 "7) NO STRING FORMATTING: Do not use .format() method or % formatting.\n" +
                 "8) NO CURLY BRACES IN STRINGS: Avoid { } in any string literals except in the final JSON structure.\n" +
                 "9) If you need to print debug information, use regular print statements with string concatenation only.\n" +
                 "10) For metrics calculation: First compute each metric value, assign to variables, then build the dict:\n" +
                 "   ```\n" +
                 "   accuracy_val = accuracy_score(y_test, y_pred)\n" +
                 "   precision_val = precision_score(y_test, y_pred, average='weighted')\n" +
                 "   metrics = {\"accuracy\": accuracy_val, \"precision\": precision_val}\n" +
                 "   ```\n" +
                 "11) SILENCE ALL WARNINGS: Add these lines at the top after imports:\n" +
                 "    import warnings\n" +
                 "    warnings.filterwarnings('ignore')\n" +
                 "    import os\n" +
                 "    os.environ['PYTHONWARNINGS'] = 'ignore'\n" +
                 "12) JSON SERIALIZATION: Convert numpy types to Python native types before JSON serialization using convert_numpy_types function.\n" +
                 "13) ENSURE OUTPUT: The script MUST print the JSON output to stdout as the last action. Do not suppress or capture this output.\n" +
                 "14) Reply ONLY with the full Python code inside a single ```python code block.\n\n" +
                 "EXAMPLE STRUCTURE (illustrative)\n" +
                 "```python\n" +
                 "import os, json, pandas as pd\n" +
                 "import warnings, numpy as np\n" +
                 "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score\n" +
                 "# import model class ...\n\n" +
                 "# Suppress warnings\n" +
                 "warnings.filterwarnings('ignore')\n" +
                 "os.environ['PYTHONWARNINGS'] = 'ignore'\n\n" +
                 "def convert_numpy_types(obj):\n" +
                 "    if isinstance(obj, dict):\n" +
                 "        return {k: convert_numpy_types(v) for k, v in obj.items()}\n" +
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
                 "X_test_path = r\"" + str(x_test_path) + "\"\n" +
                 "y_test_path = r\"" + str(y_test_path) + "\"\n\n" +
                 "X_train = pd.read_csv(X_train_path).reset_index(drop=True)\n" +
                 "y_train = pd.read_csv(y_train_path).squeeze().reset_index(drop=True)\n" +
                 "X_test = pd.read_csv(X_test_path).reset_index(drop=True)\n" +
                 "y_test = pd.read_csv(y_test_path).squeeze().reset_index(drop=True)\n\n" +
                 "# model = ...\n" +
                 "# model.fit(X_train, y_train)\n" +
                 "# y_pred = model.predict(X_test)\n" +
                 "# metrics = {\"accuracy\": accuracy_score(y_test, y_pred)}  # choose by task type\n\n" +
                 "import pickle\n" +
                 "out_path = os.path.abspath('trained_model.pickle')\n" +
                 "with open(out_path, 'wb') as f:\n" +
                 "    pickle.dump(model, f)\n" +
                 "result = {\"metrics\": convert_numpy_types(metrics), \"model_path\": out_path}\n" +
                 "print(json.dumps(result))\n" +
                 "```\n\n" +
                 "Now produce the complete runnable script.")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        
        messages = [
            {"role": "system", "content": "You are a data engineering expert who writes and debugs Python scripts for model training."},
            {"role": "user", "content": prompt}
        ]

        # If an upstream controller provided error context, feed it to the LLM
        upstream_error_ctx = kwargs.get("error_context")
        if upstream_error_ctx and isinstance(upstream_error_ctx, dict):
            messages.append({
                "role": "user",
                "content": self._build_error_feedback_prompt(upstream_error_ctx, prompt)
            })
        
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

            # Validate code for f-string issues before execution
            is_valid, validation_error = self._validate_code_for_fstring_issues(code)
            if not is_valid:
                print("\n[" + self.__class__.__name__ + "] Attempt " + str(attempt + 1) + "/" + str(max_retries) + " failed: Code validation error.")
                print("Validation error: " + str(validation_error))
                
                feedback = ("Your code contains formatting issues that will cause execution failures:\n\n" +
                           str(validation_error) + "\n\n" +
                           "Please rewrite the code following these strict rules:\n" +
                           "- Use NO f-strings anywhere (no f\"...\" or f'...')\n" +
                           "- Use NO .format() method calls\n" +
                           "- Use NO % string formatting\n" +
                           "- Use simple string concatenation with + operator if needed\n" +
                           "- For metrics, calculate values first, then build dict: accuracy_val = accuracy_score(y_test, y_pred); metrics = {\"accuracy\": accuracy_val}\n\n" +
                           "Please provide the corrected code.")

                messages.append({"role": "user", "content": feedback})
                if attempt == max_retries - 1:
                    raise Exception("Code validation failed after multiple attempts: " + str(validation_error))
                continue

            try:
                before_files = self._snapshot_files(sandbox)

                # Streamed execution with robust stdout/stderr capture
                stdout_buffer = []
                stderr_buffer = []
                execution_error = None

                def on_stdout(data):
                    stdout_buffer.append(data.get('line', str(data)) if isinstance(data, dict) else str(data))

                def on_stderr(data):
                    stderr_buffer.append(data.get('line', str(data)) if isinstance(data, dict) else str(data))

                def on_error(error):
                    nonlocal execution_error
                    execution_error = error

                # Extend sandbox TTL to avoid premature shutdowns during training
                try:
                    sandbox.set_timeout(3600)
                except Exception:
                    pass

                try:
                    execution = sandbox.run_code(
                        code,
                        timeout=1800,
                        on_stdout=on_stdout,
                        on_stderr=on_stderr,
                        on_error=on_error,
                    )
                except TypeError:
                    # Fallback if this Sandbox version does not support streaming callbacks
                    execution = sandbox.run_code(code, timeout=1800)

                # Consolidate outputs from multiple sources
                stdout = ''.join(stdout_buffer).strip()
                stderr = ''.join(stderr_buffer).strip()

                if not stdout and not stderr:
                    # Try direct attributes
                    if hasattr(execution, 'stdout') and execution.stdout:
                        stdout = str(execution.stdout).strip()
                    if hasattr(execution, 'stderr') and execution.stderr:
                        stderr = str(execution.stderr).strip()

                    # Try logs
                    if not stdout:
                        logs = getattr(execution, 'logs', None)
                        if logs:
                            if isinstance(logs, str):
                                stdout = logs.strip()
                            elif isinstance(logs, (list, tuple)):
                                log_parts = []
                                for log in logs:
                                    if hasattr(log, 'stdout') and log.stdout:
                                        log_parts.append(str(log.stdout))
                                    elif isinstance(log, str):
                                        log_parts.append(log)
                                    elif hasattr(log, 'text') and log.text:
                                        log_parts.append(str(log.text))
                                if log_parts:
                                    stdout = '\n'.join(log_parts).strip()

                    # Try results
                    if not stdout:
                        results = getattr(execution, 'results', None)
                        if results:
                            result_parts = []
                            for result in results:
                                if hasattr(result, 'text') and result.text:
                                    result_parts.append(str(result.text))
                                elif isinstance(result, dict) and 'text' in result:
                                    result_parts.append(str(result['text']))
                                elif isinstance(result, str):
                                    result_parts.append(result)
                            if result_parts:
                                stdout = '\n'.join(result_parts).strip()

                err = execution_error or getattr(execution, 'error', None)

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

                    # Build structured error context for the next attempt
                    structured_ctx = self._create_detailed_error_context(code, err, stdout, stderr) if 'code' in locals() else {
                        "error_type": getattr(err, 'name', 'ExecutionError'),
                        "error_message": getattr(err, 'value', str(err)),
                        "stdout": stdout,
                        "stderr": stderr,
                        "suggested_fixes": ["Check traceback and fix indicated line", "Ensure consistent preprocessing between train and test"],
                    }
                    # store for orchestrator/tooling visibility
                    self.last_error_context = structured_ctx
                    messages.append({
                        "role": "user",
                        "content": self._build_error_feedback_prompt(structured_ctx, prompt)
                    })
                    if attempt == max_retries - 1:
                        raise Exception("Code execution failed after " + str(max_retries) + " attempts. Last error: " + str(getattr(err, 'value', str(err))))
                    continue

                # Success
                combined = (stdout or "") + ("\n" + stderr if stderr else "")
                training_results = None
                if stdout:
                    try:
                        training_results = json.loads(stdout)
                    except Exception:
                        # Try extracting JSON anywhere in combined output
                        training_results = self._extract_json_from_text(combined)
                else:
                    # Try extracting JSON from combined output (some envs stream to stderr)
                    training_results = self._extract_json_from_text(combined)

                if not training_results:
                    # Last resort: discover created files and synthesize minimal result
                    after_files = self._snapshot_files(sandbox)
                    created = [p for p in after_files if p not in set(before_files)] if after_files else []
                    model_candidates = [p for p in created if any(p.lower().endswith(ext) for ext in [".pickle", ".pkl", ".joblib"]) and ("model" in p.lower() or "trained" in p.lower())]
                    model_path = model_candidates[0] if model_candidates else None
                    if not model_path:
                        # Also check common default path
                        common_path = "/home/user/trained_model.pickle"
                        if common_path in (after_files or []):
                            model_path = common_path
                    if not model_path and after_files:
                        # Broaden search: any model artifact under known roots
                        all_model_files = [p for p in after_files if any(p.lower().endswith(ext) for ext in [".pickle", ".pkl", ".joblib"])]
                        # Prefer filenames containing 'trained' or 'model'
                        preferred = [p for p in all_model_files if ("trained" in os.path.basename(p).lower() or "model" in os.path.basename(p).lower())]
                        model_path = (preferred[0] if preferred else (all_model_files[0] if all_model_files else None))
                    if model_path:
                        training_results = {"metrics": {}, "model_path": model_path}
                    else:
                        raise Exception("The script ran but produced no output.")
                
                model_path = training_results.get("model_path")
                if model_path:
                    try:
                        # Download the model directly before killing sandbox
                        model_bytes = sandbox.files.read(model_path)
                        if isinstance(model_bytes, str):
                            try:
                                model_bytes = model_bytes.encode('latin-1')
                            except Exception:
                                model_bytes = model_bytes.encode('utf-8', 'ignore')
                        training_results["model_bytes"] = model_bytes
                        print("Model downloaded directly from sandbox before cleanup")
                    except Exception as e:
                        print("Failed to download model directly from sandbox: " + str(e))
                        # Fallback to download URL if direct download fails
                        try:
                            training_results["model_download_url"] = sandbox.download_url(model_path)
                        except Exception:
                            pass

                # Clean up sandbox before returning
                try:
                    sandbox.kill()
                except Exception:
                    pass
                    
                return training_results

            except Exception as e:
                print("\n[" + self.__class__.__name__ + "] Attempt " + str(attempt + 1) + "/" + str(max_retries) + " failed: An unexpected exception occurred.")
                print("Exception: " + str(e))
                structured_ctx = self._create_detailed_error_context(locals().get('code', ''), e, '', '')
                self.last_error_context = structured_ctx
                messages.append({
                    "role": "user",
                    "content": self._build_error_feedback_prompt(structured_ctx, prompt)
                })
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

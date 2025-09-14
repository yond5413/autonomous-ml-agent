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

        fixes_block = "\n".join(f"- {fix}" for fix in suggested_fixes) if suggested_fixes else "- Review recent changes around the reported line\n- Ensure balanced quotes, brackets, parentheses\n- Check indentation and block structure"
        stdout_block = f"\n**Stdout (context):**\n```\n{stdout}\n```" if stdout else ""
        stderr_block = f"\n**Stderr (context):**\n```\n{stderr}\n```" if stderr else ""

        return f'''
        **PREVIOUS TRAINING ATTEMPT FAILED**

        **Error Details:**
        - Type: {error_type}
        - Message: {error_message}
        - Line: {line_number}

        **Problematic Code Section:**
        ```
        {problematic_code}
        ```

        **Suggested Fixes:**
        {fixes_block}
        {stdout_block}
        {stderr_block}

        **Original Requirements:**
        {original_prompt}

        **TASK:**
        Provide a corrected, complete Python script that adheres to the original requirements and addresses the error above. Reply ONLY with the full Python code inside a single markdown code block.
        '''

    def _generate_suggested_fixes(self, error_type: str, error_message: str) -> list:
        fixes = []
        msg = (error_message or "").lower()
        if "indentationerror" in error_type.lower():
            fixes += [
                "Fix indentation levels consistently (spaces only)",
                "Ensure each control statement has a properly indented block",
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
                prefix = f"{i+1}: "
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

        return stdout, stderr, err

    def _extract_json_from_text(self, text: str):
        if not text:
            return None
        # Find candidate JSON objects
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
        sandbox = preprocessing_output["sandbox"]
        remote_paths = preprocessing_output.get("remote_paths", {})
        download_urls = preprocessing_output.get("download_urls", {})
        local_paths = preprocessing_output.get("local_dataset_paths", {})
        model_name = model_selection.get("recommended_model")
        task_type = model_selection.get("task_type")

        # Prefer absolute file paths inside sandbox over URLs
        def pick_path(key: str) -> str | None:
            # remote_paths are absolute paths inside sandbox
            path = remote_paths.get(key)
            if isinstance(path, str) and path:
                return path
            # fallback to download_urls if they are absolute sandbox paths (not http)
            url = download_urls.get(key)
            if isinstance(url, str) and url and not url.lower().startswith("http"):
                return url
            # last resort: local host paths (not ideal inside sandbox, but kept for completeness)
            lp = local_paths.get(key)
            return lp if isinstance(lp, str) and lp else None

        x_train_path = pick_path("X_train")
        y_train_path = pick_path("y_train")
        x_test_path = pick_path("X_test")
        y_test_path = pick_path("y_test")
        x_val_path = pick_path("X_val")
        y_val_path = pick_path("y_val")

        val_lines = ""
        if x_val_path and y_val_path:
            val_lines = f"\n    - X_val_path: {x_val_path}\n    - y_val_path: {y_val_path}"

        prompt = f"""
You are an expert data scientist tasked with writing a Python script to train a machine learning model and evaluate it.

Here is the context:
1.  **Model to be trained**: {model_name}
2.  **Task Type**: {task_type}
3.  **Datasets**: The datasets are available at the following absolute file paths inside the environment. Your script must load directly from these file paths (not URLs).
    - X_train_path: {x_train_path}
    - y_train_path: {y_train_path}
    - X_test_path: {x_test_path}
    - y_test_path: {y_test_path}{val_lines}

Important: Validation files may be missing. Your script MUST NOT fail if validation files are absent; simply proceed without using a validation set.

**Your task is to generate a Python script that performs the following steps:**
1.  **Load Data**: At the top of the script, assign variables for the file paths exactly as provided above (e.g., `X_train_path = r"{x_train_path}"`). Then load the datasets using `pd.read_csv(X_train_path)` etc. Always load training and test data. Load validation data only if both validation paths are provided (handle missing validation gracefully).
2.  **Instantiate Model**: Import the specified model (`{model_name}`) from scikit-learn and instantiate it.
3.  **Train Model**: Train the model on the training data (X_train, y_train).
4.  **Evaluate Model**: Evaluate the trained model on the test data (X_test, y_test).
    - For **classification** tasks, calculate accuracy, precision, recall, and F1-score.
    - For **regression** tasks, calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared.
5.  **Save Model**: Save the trained model to a file named `trained_model.pickle` using the `pickle` library.
6.  **Output**: Print a JSON object to standard output containing the calculated evaluation metrics and the path to the saved model file (`/home/user/trained_model.pickle`).

Please generate only the Python script itself, without any explanations.
"""

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
        for attempt in range(max_retries):
            response = client.chat.completions.create(
                model="moonshotai/kimi-dev-72b:free",#"qwen/qwen3-coder:free", # prefer this but rate limits
                messages=messages,
                max_tokens=2500
            )
            
            generated_text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": generated_text})

            code_match = re.search(r'```(?:python)?\s*(.*?)```', generated_text, re.DOTALL)
            
            if not code_match:
                print(f"\n[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries} failed: LLM did not return a Python code block.")
                feedback = "Your response did not contain a Python code block. Please generate only the Python code, enclosed in a ```python ... ``` block."
                messages.append({"role": "user", "content": feedback})
                if attempt == max_retries - 1:
                    raise Exception("No code found in response after multiple attempts.")
                continue

            code = code_match.group(1).strip()

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

                try:
                    execution = sandbox.run_code(
                        code,
                        timeout=1200,
                        on_stdout=on_stdout,
                        on_stderr=on_stderr,
                        on_error=on_error,
                    )
                except TypeError:
                    # Fallback if this Sandbox version does not support streaming callbacks
                    execution = sandbox.run_code(code, timeout=1200)

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
                        f"Error Name: {getattr(err, 'name', 'ExecutionError')}\n"
                        f"Error Value: {getattr(err, 'value', str(err))}\n"
                        f"Traceback:\n{getattr(err, 'traceback', '')}"
                    )
                    if stdout:
                        error_details += f"\n--- stdout ---\n{stdout}"
                    if stderr:
                        error_details += f"\n--- stderr ---\n{stderr}"
                    print(f"\n[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries} failed: Code execution error.")
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
                        raise Exception(f"Code execution failed after {max_retries} attempts. Last error: {getattr(err, 'value', str(err))}")
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
                        training_results["model_download_url"] = sandbox.download_url(model_path)
                    except Exception:
                        # If the sandbox API doesn't support download_url, continue without it
                        pass

                return training_results

            except Exception as e:
                print(f"\n[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries} failed: An unexpected exception occurred.")
                print(f"Exception: {e}")
                structured_ctx = self._create_detailed_error_context(locals().get('code', ''), e, '', '')
                self.last_error_context = structured_ctx
                messages.append({
                    "role": "user",
                    "content": self._build_error_feedback_prompt(structured_ctx, prompt)
                })
                if attempt == max_retries - 1:
                    raise e
        
        raise Exception("Failed to generate and execute valid code after multiple attempts.")

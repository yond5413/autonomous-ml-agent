import os
import re
import json
from pathlib import Path
from e2b_code_interpreter import Sandbox
from .base import Agent
from openai import OpenAI

class PreprocessingAgent(Agent):
    MAX_RETRIES = 3

    def execute(self, model_selection_result: str, exploration_summary: str, file_path: str, target_column: str, **kwargs):
        try:
            model_selection = json.loads(model_selection_result)
        except json.JSONDecodeError:
            raise ValueError("model_selection_result is not a valid JSON.")

        original_prompt = self._build_prompt(exploration_summary, model_selection, file_path, target_column)
        messages = [
            {"role": "system", "content": "You are a data engineering expert who writes and debugs Python scripts for data preprocessing. Your goal is to produce a single, complete, and runnable Python script that prints a JSON object to stdout."}, 
            {"role": "user", "content": original_prompt}
        ]

        # Add error context from previous attempt if available
        if "error_context" in kwargs:
            error_context = kwargs["error_context"]
            error_feedback = self._build_error_feedback_prompt(error_context, original_prompt)
            messages.append({"role": "user", "content": error_feedback})

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        for attempt in range(self.MAX_RETRIES):
            print(f"[{self.__class__.__name__}] Attempt {attempt + 1}/{self.MAX_RETRIES} to generate and execute script.")
            
            try:
                response = client.chat.completions.create(
                    model="deepseek/deepseek-chat-v3.1:free",#"anthropic/claude-3.5-sonnet",
                    messages=messages,
                    max_tokens=4096
                )

                if not response.choices:
                    raise Exception("LLM response had no choices.")
                
                generated_text = response.choices[0].message.content
                if not generated_text:
                    raise Exception("LLM response was empty.")

                messages.append({"role": "assistant", "content": generated_text})
                code = self._extract_code(generated_text)

                if not code:
                    feedback = "Your response did not contain a Python code block. Please provide the complete, corrected script inside a single markdown code block."
                    messages.append({"role": "user", "content": feedback})
                    continue

                return self._run_in_sandbox(code, file_path)

            except Exception as e:
                print(f"An error occurred on attempt {attempt + 1}: {e}")
                
                # Create detailed error context for next attempt
                if attempt < self.MAX_RETRIES - 1:
                    # Extract error details from exception
                    error_context = self._create_detailed_error_context(
                        code if 'code' in locals() else "",
                        e,
                        "",
                        str(e)
                    )
                    
                    # Add detailed error feedback to messages
                    error_feedback = self._build_error_feedback_prompt(
                        error_context, 
                        original_prompt
                    )
                    messages.append({"role": "user", "content": error_feedback})
                
                if attempt == self.MAX_RETRIES - 1:
                    raise e
        
        raise Exception(f"Failed to generate and execute valid code after {self.MAX_RETRIES} attempts.")

    def _get_execution_output(self, execution) -> tuple[str, str, dict | None]:
        """Extract stdout, stderr and structured error from an execution object."""
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


    def _extract_code(self, text: str) -> str | None:
        match = re.search(r'```(?:python)?\s*(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_json_output(self, output: str) -> dict:
        # Find all substrings that look like JSON objects (non-greedy)
        candidates = re.findall(r'\{[\s\S]*?\}', output, re.DOTALL)

        # Also attempt to extract the last balanced JSON object by scanning backwards
        def _extract_last_balanced_json(s: str) -> str | None:
            end_idx = s.rfind('}')
            if end_idx == -1:
                return None
            depth = 0
            for i in range(end_idx, -1, -1):
                if s[i] == '}':
                    depth += 1
                elif s[i] == '{':
                    depth -= 1
                    if depth == 0:
                        return s[i:end_idx+1]
            return None

        def _normalize(obj: dict) -> dict:
            # If script already returned the expected mapping
            expected_keys = {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "preprocessor"}
            if any(k in obj for k in expected_keys):
                return {k: v for k, v in obj.items() if k in expected_keys and isinstance(v, str)}

            # If nested under "preprocessed_data"
            pre = obj.get("preprocessed_data")
            if isinstance(pre, dict):
                # Some scripts may include extra keys like feature_names/target_classes alongside
                return {k: v for k, v in pre.items() if isinstance(v, str)}

            # If keys are *_path style
            key_map = {
                # common outputs
                "train_data_path": "X_train",
                "X_train_path": "X_train",
                "train_features_path": "X_train",
                "test_data_path": "X_test",
                "X_test_path": "X_test",
                "test_features_path": "X_test",
                "train_labels_path": "y_train",
                "y_train_path": "y_train",
                "train_target_path": "y_train",
                "test_labels_path": "y_test",
                "y_test_path": "y_test",
                "test_target_path": "y_test",
                "val_data_path": "X_val",
                "X_val_path": "X_val",
                "val_features_path": "X_val",
                "val_labels_path": "y_val",
                "y_val_path": "y_val",
                "val_target_path": "y_val",
                "preprocessor_path": "preprocessor",
            }
            if any(k in obj for k in key_map.keys()):
                normalized = {}
                for src_key, dst_key in key_map.items():
                    val = obj.get(src_key)
                    if isinstance(val, str) and val:
                        normalized[dst_key] = val
                if normalized:
                    return normalized

            # As a last resort, pick only string values that look like file paths
            fallback = {k: v for k, v in obj.items() if isinstance(v, str) and ("/" in v or "." in v)}
            if fallback:
                return fallback
            return {}

        for candidate in reversed(candidates):  # check from last match backwards
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    normalized = _normalize(parsed)
                    if normalized:
                        return normalized
            except json.JSONDecodeError:
                continue

        # Try balanced extraction if regex didn't yield a usable object
        balanced = _extract_last_balanced_json(output)
        if balanced:
            try:
                parsed = json.loads(balanced)
                if isinstance(parsed, dict):
                    normalized = _normalize(parsed)
                    if normalized:
                        return normalized
            except Exception:
                pass

        raise Exception(f"Failed to decode JSON from output. Full output:\n{output}")

    def _run_in_sandbox(self, code: str, local_file_path: str):
        sbx = None
        try:
            output_dir = "outputs/generated_scripts"
            os.makedirs(output_dir, exist_ok=True)
            script_path = os.path.join(output_dir, "preprocessing_script_latest.py")
            with open(script_path, "w") as f:
                f.write(code)
            print(f"Generated script saved to {script_path}")

            sbx = Sandbox.create() 

            with open(local_file_path, "rb") as f:
                remote_path = sbx.files.write(os.path.basename(local_file_path), f).path

            # Ensure a common fallback path also exists inside the sandbox
            try:
                base_name = os.path.basename(local_file_path)
                fallback_dir = "/tmp/autonomous-ml-agent"
                fallback_path = f"{fallback_dir}/{base_name}"
                sbx.run_code(
                    f"import os, shutil\n"
                    f"os.makedirs('{fallback_dir}', exist_ok=True)\n"
                    f"shutil.copy('{remote_path}', '{fallback_path}')\n",
                    timeout=60
                )
            except Exception:
                # Non-fatal; continue with remote_path only
                pass

            # Safer path substitution: only replace inside explicit file path literals
            base_name = os.path.basename(local_file_path)
            code_to_run = code
            # Replace in pd.read_csv("...basename...")
            pattern_csv = r'(pd\.read_csv\(\s*)([\'\"][^\'\"]*' + re.escape(base_name) + r'[^\'\"]*[\'\"])'
            code_to_run = re.sub(pattern_csv, lambda m: f"{m.group(1)}\"{remote_path}\"", code_to_run)
            # Replace in Path("...basename...")
            pattern_path = r'(Path\(\s*)([\'\"][^\'\"]*' + re.escape(base_name) + r'[^\'\"]*[\'\"])(\s*\))'
            code_to_run = re.sub(pattern_path, lambda m: f"{m.group(1)}\"{remote_path}\"{m.group(3)}", code_to_run)
            # As a last resort, replace exact full path literal if present
            literal = re.escape(local_file_path)
            code_to_run = re.sub(r'([\'\"])' + literal + r'\1', f'"{remote_path}"', code_to_run)

            # Final guard: monkey-patch pandas.read_csv to always use the uploaded path for this CSV
            preamble = (
                "import os\n"
                "try:\n"
                "    import pandas as pd\n"
                "    _orig_read_csv = pd.read_csv\n"
                f"    _target_base = '{base_name}'\n"
                f"    _remote_path = '{remote_path}'\n"
                "    def _patched_read_csv(path, *args, **kwargs):\n"
                "        try:\n"
                "            p = str(path)\n"
                "            if _target_base in os.path.basename(p):\n"
                "                return _orig_read_csv(_remote_path, *args, **kwargs)\n"
                "        except Exception:\n"
                "            pass\n"
                "        return _orig_read_csv(path, *args, **kwargs)\n"
                "    pd.read_csv = _patched_read_csv\n"
                "except Exception:\n"
                "    pass\n"
            )
            code_to_run = preamble + "\n" + code_to_run

            # Capture file snapshot BEFORE running code
            def _snapshot_files() -> list[str]:
                snap_exec = sbx.run_code(
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
                s_out, s_err, _ = self._get_execution_output(snap_exec)
                try:
                    return json.loads(s_out) if s_out else []
                except Exception:
                    return []

            before_files = _snapshot_files()

            # Execute code with robust output/error capture
            stdout_buffer = []
            stderr_buffer = []
            execution_error = None

            # Try with streaming callbacks first
            def on_stdout(data):
                stdout_buffer.append(data.get('line', str(data)) if isinstance(data, dict) else str(data))
            
            def on_stderr(data):
                stderr_buffer.append(data.get('line', str(data)) if isinstance(data, dict) else str(data))
            
            def on_error(error):
                nonlocal execution_error
                execution_error = error

            try:
                # Execute with streaming (may not work with all E2B versions)
                execution = sbx.run_code(
                    code_to_run, 
                    timeout=600,
                    on_stdout=on_stdout,
                    on_stderr=on_stderr,
                    on_error=on_error
                )
            except TypeError:
                # Fallback if streaming parameters not supported
                execution = sbx.run_code(code_to_run, timeout=600)

            # Get output from multiple sources
            stdout = ''.join(stdout_buffer).strip()
            stderr = ''.join(stderr_buffer).strip()
            
            # Also try traditional result parsing as fallback
            if not stdout and not stderr:
                # Try getting output from execution object
                if hasattr(execution, 'stdout'):
                    stdout = str(execution.stdout).strip()
                if hasattr(execution, 'stderr'):
                    stderr = str(execution.stderr).strip()
                
                # Try logs
                if not stdout:
                    logs = getattr(execution, "logs", None)
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
                    results = getattr(execution, "results", None)
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

            # Check for execution errors
            if not execution_error:
                execution_error = getattr(execution, "error", None)

            # Handle errors
            if execution_error:
                err_name = getattr(execution_error, "name", type(execution_error).__name__)
                err_value = getattr(execution_error, "value", str(execution_error))
                err_traceback = getattr(execution_error, "traceback", "")
                error_message = f"{err_name}: {err_value}"
                if err_traceback:
                    error_message += f"\nTraceback:\n{err_traceback}"
                if stdout:
                    error_message += f"\n--- stdout ---\n{stdout}"
                if stderr:
                    error_message += f"\n--- stderr ---\n{stderr}"
                raise Exception(error_message)

            # Check stderr for errors
            if stderr and not stdout:
                raise Exception(f"Code execution failed with stderr:\n{stderr}")

            # Use stdout as primary output
            output = stdout

            # If still no output, try file discovery fallback
            if not output:
                after_files = _snapshot_files()
                created = [p for p in after_files if p not in set(before_files)]
                outputs = {}
                for p in created:
                    lower = p.lower()
                    if lower.endswith('.csv'):
                        name = os.path.splitext(os.path.basename(p))[0]
                        outputs[name] = p
                    elif lower.endswith('.joblib') or lower.endswith('.pkl') or lower.endswith('.pickle'):
                        key = 'preprocessor' if 'preprocess' in lower else os.path.splitext(os.path.basename(p))[0]
                        outputs[key] = p
                
                if outputs:
                    print(f"No stdout captured, but found {len(outputs)} created files. Using file discovery fallback.")
                    processed_files = outputs
                    return {
                        "remote_paths": processed_files,
                        "sandbox": sbx,
                    }
                
                # Last resort: check if execution completed successfully but with no output
                if hasattr(execution, 'exit_code') and execution.exit_code == 0:
                    raise Exception("Script executed successfully but produced no output. Check if the script prints the required JSON.")
                else:
                    raise Exception("Code execution produced no output and no error.")

            # Happy path: we have output

            processed_files = self._extract_json_output(output)

            return {
                "remote_paths": processed_files,
                "sandbox": sbx,
            }
        except Exception as e:
            if sbx:
                sbx.kill()
            raise e

    def _create_detailed_error_context(self, code: str, error: Exception, stdout: str = "", stderr: str = "") -> dict:
        """Create structured error context for LLM feedback"""
        
        error_type = type(error).__name__
        error_message = str(error)
        
        # Parse line numbers from traceback if available
        line_number = None
        if hasattr(error, 'lineno'):
            line_number = error.lineno
        elif 'line' in str(error).lower():
            # Extract line number from error message
            import re
            line_match = re.search(r'line (\d+)', str(error))
            if line_match:
                line_number = int(line_match.group(1))
        
        # Get problematic code section
        problematic_code = ""
        if line_number:
            lines = code.split('\n')
            start = max(0, line_number - 3)
            end = min(len(lines), line_number + 3)
            problematic_code = '\n'.join([
                (f"{i+1}: {lines[i]}" if i+1 != line_number else f"{i+1}: >>> {lines[i]} <<< ERROR HERE")
                for i in range(start, end)
            ])
        
        return {
            "error_type": error_type,
            "error_message": error_message,
            "line_number": line_number,
            "problematic_code": problematic_code,
            "full_code": code,
            "stdout": stdout,
            "stderr": stderr,
            "suggested_fixes": self._generate_suggested_fixes(error_type, error_message, problematic_code)
        }

    def _generate_suggested_fixes(self, error_type: str, error_message: str, code_context: str) -> list:
        """Generate specific fix suggestions based on error type"""
        
        fixes = []
        
        if "SyntaxError" in error_type:
            if "parenthesis" in error_message.lower():
                fixes.append("Check for balanced parentheses, brackets, and quotes")
                fixes.append("Ensure proper string concatenation syntax")
                fixes.append("Use f-strings for complex string formatting")
            elif "invalid syntax" in error_message.lower():
                fixes.append("Review syntax for function calls and string formatting")
                fixes.append("Check for missing colons or incorrect indentation")
        
        elif "NameError" in error_type:
            fixes.append("Verify all variables are properly defined before use")
            fixes.append("Check for typos in variable names")
        
        elif "TypeError" in error_type:
            fixes.append("Check function argument types and method signatures")
            fixes.append("Verify data types match expected inputs")
        
        return fixes

    def _get_common_syntax_patterns(self) -> dict:
        """Provide common syntax patterns and fixes"""
        return {
            "string_concatenation": {
                "wrong": "print('start' + variable, 'end']",
                "correct": "print('start' + str(variable) + 'end')"
            },
            "function_calls": {
                "wrong": "df.drop(['col1', 'col2'], axis=1]",
                "correct": "df.drop(['col1', 'col2'], axis=1)"
            },
            "string_formatting": {
                "wrong": "print('--- SCRIPT STARTED ---' + path + ' col']",
                "correct": "print('--- SCRIPT STARTED ---' + str(path))"
            }
        }

    def _build_error_feedback_prompt(self, error_context: dict, original_prompt: str) -> str:
        """Build detailed error feedback prompt for LLM. Tolerant to partial error_context."""
        
        error_type = error_context.get('error_type', 'UnknownError')
        error_message = error_context.get('error_message', '')
        line_number = error_context.get('line_number', 'Unknown')
        problematic_code = error_context.get('problematic_code', '')
        suggested_fixes = error_context.get('suggested_fixes')
        if not suggested_fixes:
            # Some callers provide a single 'suggested_fix' string
            single_fix = error_context.get('suggested_fix')
            suggested_fixes = [single_fix] if single_fix else []
        stdout = error_context.get('stdout', '')
        stderr = error_context.get('stderr', '')

        fixes_block = "\n".join(f"- {fix}" for fix in suggested_fixes) if suggested_fixes else "- Review recent changes around the reported line\n- Check for unbalanced quotes, brackets, or parentheses"
        stdout_block = f"\n**Stdout (for context):**\n```\n{stdout}\n```" if stdout else ""
        stderr_block = f"\n**Stderr (for context):**\n```\n{stderr}\n```" if stderr else ""
        
        return f'''
        **PREVIOUS ATTEMPT FAILED**
        
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
        Generate a corrected Python script that fixes the identified issues. 
        
        **Specific Instructions:**
        1. Address the syntax error directly
        2. Ensure all parentheses, brackets, and quotes are balanced
        3. Verify proper string formatting and concatenation
        4. Test the corrected code mentally before providing it
        
        **Additional Context:**
        - The error occurred during preprocessing script generation
        - Focus on the specific syntax issue rather than changing the overall approach
        - Ensure the corrected code maintains the original functionality
        
        Provide only the corrected Python code in your response.
        '''

    def _build_prompt(self, exploration_summary: str, model_selection: dict, file_path: str, target_column: str) -> str:
        plan_as_string = "\n".join(f"- {step}" for step in model_selection.get("preprocessing_plan", []))
        
        # This prompt is now modeled after the original, asking for a complete script but with added guidance.
        prompt_template = '''
        You are an expert data scientist. Your task is to write a complete Python script to preprocess a dataset for model training based on the provided plan.

        **CONTEXT:**
        1.  **Data Exploration Summary**:
            ```
            {exploration_summary}
            ```
        2.  **Selected Model**: {recommended_model}
        3.  **Recommended Preprocessing Plan**:
            ```
            {plan_as_string}
            ```

        The dataset is located at the path: `{file_path}`
        The target column is: `{target_column}`

        **TASK:**
        Generate a complete, single Python script that meticulously implements the preprocessing plan.

        **CRITICAL INSTRUCTIONS**:
        1.  **Start Signal**: At the very beginning of your script, after the imports, you **MUST** add the line `print('--- SCRIPT EXECUTION STARTED ---')`. This is essential for debugging.
        2.  **File Path**: Load the dataset using the exact path provided in the context: `df = pd.read_csv("{file_path}")`. The system will replace this path correctly.
        3.  **Dependencies**: Only use pandas, scikit-learn, numpy, and joblib. Do NOT import category_encoders, xgboost, or other external libraries that may not be available.
        4.  **Scikit-learn Version**: You are using a modern version of scikit-learn (1.0+). For `OneHotEncoder`, you **MUST** use the `sparse_output=False` parameter, not the old `sparse=False` parameter.
        5.  **Feature Names**: If all transformers in the pipeline support `get_feature_names_out`, then after fitting the `ColumnTransformer`, derive feature names using `preprocessor.get_feature_names_out()`. If any custom transformer (e.g., outlier cappers) does not implement this method, skip feature-name extraction and write transformed arrays directly to CSV without column names; do not fail the script for missing feature names.
        6.  **Train/Val/Test Split**: You MUST output three splits: train, validation, and test. Aim for 64%/16%/20% (train/val/test). First split off test (20%); then split remaining into train/val (80/20). If stratification fails due to class counts, fall back to random splitting without stratification.
        7.  **Target Column**: Make sure to use the correct target column "{target_column}" - do NOT accidentally use identifier columns like customer_id.
        8.  **Final Output**: Your script **MUST** end by printing a single JSON object to standard output with these exact keys (string file paths): `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`, `preprocessor`. Do not print anything else after this JSON.

        **EXAMPLE SCRIPT STRUCTURE:**
        ```python
        # requirements: pandas, scikit-learn, joblib
        import pandas as pd
        import json
        from sklearn.model_selection import train_test_split
        # ... other imports ...

        # 1. Load data
        df = pd.read_csv("{file_path}")

        # 2. Implement the preprocessing plan...
        # ... (feature engineering, outlier handling, etc.) ...
        
        # 3. Define preprocessor, split data, transform, etc.
        # ... (ColumnTransformer, train_test_split, etc.) ...

        # 4. Save outputs
        # ... (joblib.dump, to_csv, etc.) ...

        # 5. Print final JSON to stdout
        output_data = {{
            "X_train": "/abs/path/to/X_train.csv",
            "X_val": "/abs/path/to/X_val.csv",
            "X_test": "/abs/path/to/X_test.csv",
            "y_train": "/abs/path/to/y_train.csv",
            "y_val": "/abs/path/to/y_val.csv",
            "y_test": "/abs/path/to/y_test.csv",
            "preprocessor": "/abs/path/to/preprocessor.joblib"
        }}
        print(json.dumps(output_data))
        ```

        Now, generate the complete, runnable Python script.
        '''

        return prompt_template.format(
            exploration_summary=exploration_summary,
            recommended_model=model_selection.get('recommended_model'),
            plan_as_string=plan_as_string,
            file_path=file_path,
            target_column=target_column
        )

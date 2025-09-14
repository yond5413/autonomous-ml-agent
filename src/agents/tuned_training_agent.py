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

        return stdout, stderr, err

    def _extract_json_from_text(self, text: str):
        if not text:
            return None
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
        sandbox = preprocessing_output["sandbox"]
        remote_paths = preprocessing_output.get("remote_paths", {})
        download_urls = preprocessing_output.get("download_urls", {})
        local_paths = preprocessing_output.get("local_dataset_paths", {})
        model_name = model_selection.get("recommended_model")
        task_type = model_selection.get("task_type")
        best_params = tuning_results.get("best_params")

        # Prefer absolute file paths inside sandbox over URLs
        def pick_path(key: str) -> str | None:
            path = remote_paths.get(key)
            if isinstance(path, str) and path:
                return path
            url = download_urls.get(key)
            if isinstance(url, str) and url and not url.lower().startswith("http"):
                return url
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
            val_lines = f"\n            - X_val_path: {x_val_path}\n            - y_val_path: {y_val_path}"

        prompt = f"""
        You are an expert data scientist tasked with writing a Python script to train a final, optimized machine learning model and evaluate it.

        Here is the context:
        1.  **Model to be trained**: {model_name}
        2.  **Task Type**: {task_type}
        3.  **Optimal Hyperparameters**: Your script MUST use these exact parameters when instantiating the model.
            ```
            {best_params}
            ```
        4.  **Datasets**: The datasets are available at the following absolute file paths inside the environment. Your script must load the data directly from these file paths (not URLs).
            - X_train_path: {x_train_path}
            - y_train_path: {y_train_path}
            - X_test_path: {x_test_path}
            - y_test_path: {y_test_path}{val_lines}

        Important: Validation files may be missing. Your script MUST NOT fail if validation files are absent; in that case, train only on X_train/y_train and evaluate on X_test/y_test. If validation is present, combine X_train with X_val and y_train with y_val for final training.

        **Your task is to generate a Python script that performs the following steps:**
        1.  **Load Data**: At the top of the script, assign variables for the file paths exactly as provided above (e.g., `X_train_path = r"{x_train_path}"`). Then load the datasets using `pd.read_csv(X_train_path)` etc. Always load training and test data. Load validation data only if both validation paths are provided (handle missing validation gracefully).
        2.  **Instantiate Model**: Import the specified model (`{model_name}`) from scikit-learn and instantiate it using the **Optimal Hyperparameters** provided above.
        3.  **Train Model**: If validation is available, train on the combined training and validation data (X_train + X_val, y_train + y_val); otherwise, train on X_train/y_train only.
        4.  **Evaluate Model**: Evaluate the final model on the held-out test data (X_test, y_test).
            - For **classification** tasks, calculate accuracy, precision, recall, and F1-score.
            - For **regression** tasks, calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared.
        5.  **Feature Importance**: If the model has a `feature_importances_` attribute (like tree-based models), calculate and save the feature importances as a dictionary mapping feature names to their importance scores.
        6.  **Save Model**: Save the trained model to a file named `tuned_model.pickle` using the `pickle` library.
        7.  **Output**: Print a JSON object to standard output containing the calculated evaluation metrics, feature importances (if available), and the path to the saved model file (`/home/user/tuned_model.pickle`).

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
        
        max_retries = 3
        for attempt in range(max_retries):
            response = client.chat.completions.create(
                model="moonshotai/kimi-vl-a3b-thinking:free",#"qwen/qwen3-coder:free", # prefer this but rate limits
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
                execution = sandbox.run_code(code, timeout=1200) # 20 minute timeout for training

                stdout, stderr, err = self._get_execution_output(execution)

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

                    feedback = (
                        f"Your code failed to execute. Here is the error:\n\n"
                        f"{error_details}\n\n"
                        "Please fix the code and provide the corrected full script."
                    )
                    messages.append({"role": "user", "content": feedback})
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

                return training_results

            except Exception as e:
                print(f"\n[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries} failed: An unexpected exception occurred.")
                print(f"Exception: {e}")
                feedback = f"An unexpected error occurred during execution: {e}"
                messages.append({"role": "user", "content": feedback})
                if attempt == max_retries - 1:
                    raise e
        
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

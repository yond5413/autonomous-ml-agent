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

        prompt = self._build_prompt(exploration_summary, model_selection, file_path, target_column) 
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        messages = [
            {"role": "system", "content": "You are a data engineering expert who writes and debugs Python scripts for data preprocessing. Your goal is to produce a single, complete, and runnable Python script that prints a JSON object to stdout."}, 
            {"role": "user", "content": prompt}
        ]

        for attempt in range(self.MAX_RETRIES):
            print(f"[{self.__class__.__name__}] Attempt {attempt + 1}/{self.MAX_RETRIES} to generate and execute script.")
            
            try:
                response = client.chat.completions.create(
                    model="anthropic/claude-3.5-sonnet",
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
                if attempt == self.MAX_RETRIES - 1:
                    raise e
                messages.append({"role": "user", "content": f"Your last attempt failed with the following error:\n\n{e}\n\nPlease fix the script and try again."})
        
        raise Exception(f"Failed to generate and execute valid code after {self.MAX_RETRIES} attempts.")

    def _extract_code(self, text: str) -> str | None:
        match = re.search(r'```(?:python)?\s*(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_json_output(self, output: str) -> dict:
        # Find all substrings that look like JSON objects
        candidates = re.findall(r'\{.*?\}', output, re.DOTALL)

        for candidate in reversed(candidates):  # check from last match backwards
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

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

            # This replacement logic is critical and adapted from DataExplorationAgent
            base_name = os.path.basename(local_file_path)
            code_to_run = re.sub(
                r'([\'\"])(?:(?!\1).)*' + re.escape(base_name) + r'(?:(?!\1).)*\1',
                f'"{remote_path}" ',
                code,
                flags=re.DOTALL,
            )
            code_to_run = code_to_run.replace(local_file_path, remote_path)

            execution = sbx.run_code(code_to_run, timeout=600)

            # --- Start of improved output/error handling ---
            stdout = execution.stdout.strip()
            stderr = execution.stderr.strip()

            # Prioritize the structured error object if it exists
            err = getattr(execution, "error", None)
            if err:
                err_name = getattr(err, "name", type(err).__name__ if err else "ExecutionError")
                err_value = getattr(err, "value", str(err))
                err_traceback = getattr(err, "traceback", "")
                error_message = f"{err_name}: {err_value}\nTraceback:\n{err_traceback}"
                # Include stdout/stderr for full context
                if stdout:
                    error_message += f"\n--- stdout ---\n{stdout}"
                if stderr:
                    error_message += f"\n--- stderr ---\n{stderr}"
                raise Exception(error_message)

            # If there's no structured error, check stderr. Any output in stderr is a failure.
            if stderr:
                raise Exception(f"Code execution failed with the following error:\n{stderr}\n--- stdout ---\n{stdout}")

            # If we are here, stderr is empty. Now check if stdout is empty.
            if not stdout:
                raise Exception("Code execution produced no output and no error.")

            # The happy path: stdout has content and there were no errors.
            output = stdout
            # --- End of improved output/error handling ---

            processed_files = self._extract_json_output(output)

            download_urls = {}
            for name, path in processed_files.items():
                download_urls[name] = sbx.download_url(path, timeout=600)
            
            return {
                "download_urls": download_urls,
                "remote_paths": processed_files,
                "sandbox": sbx,
            }
        except Exception as e:
            if sbx:
                sbx.kill()
            raise e

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
        1.  **Start Signal**: At the very beginning of your script, after the imports, you **MUST** add the line `print(\'--- SCRIPT EXECUTION STARTED ---\'')`. This is essential for debugging.
        2.  **File Path**: Load the dataset using the exact path provided in the context: `df = pd.read_csv("{file_path}")`. The system will replace this path correctly.
        3.  **Dependencies**: If you use libraries beyond pandas, scikit-learn, and joblib, declare them in a `# requirements: library1, library2` comment at the very top of the script.
        4.  **Scikit-learn Version**: You are using a modern version of scikit-learn (1.0+). For `OneHotEncoder`, you **MUST** use the `sparse_output=False` parameter, not the old `sparse=False` parameter.
        5.  **Final Output**: Your script **MUST** end by printing a single JSON object to standard output. This JSON must contain the file paths of the saved datasets and the preprocessor. Do not print anything else.

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
        output_data = {{ ... }}
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

import os
import re
from e2b_code_interpreter import Sandbox
from .base import Agent
from openai import OpenAI
import json

class PreprocessingAgent(Agent):
    def execute(self, model_selection_result: str, exploration_summary: str, file_path: str, target_column: str, **kwargs):
        try:
            model_selection = json.loads(model_selection_result)
        except json.JSONDecodeError:
            raise ValueError("model_selection_result is not a valid JSON.")

        preprocessing_plan = model_selection.get("preprocessing_plan", [])
        plan_as_string = "\n".join(f"- {step}" for step in preprocessing_plan)

        prompt = f"""
You are an expert data scientist tasked with writing a Python script to preprocess a dataset for model training.

Here is the context:
1.  **Data Exploration Summary**:
    ```
    {exploration_summary}
    ```
2.  **Selected Model**: {model_selection.get('recommended_model')}
3.  **Recommended Preprocessing Plan**:
    ```
    {plan_as_string}
    ```

The dataset is located in the sandbox at: `{file_path}`
The name of the target column is `{target_column}`.

**Your primary task is to generate a Python script that meticulously implements the Recommended Preprocessing Plan.**

In addition to the plan, ensure your script also performs these standard operations:
1.  **Load the dataset** from the provided path.
2.  **Identify the target variable** using the provided target column name.
3.  **Data Splitting**: Split the data into training (80%), testing (15%), and validation (5%) sets. **This should be done BEFORE applying preprocessing** to prevent data leakage.
4.  **Apply Transformations**: Apply the preprocessing steps to the different data splits correctly (e.g., fit on training data, transform test and validation data).
5.  **Save Processed Data**: Save the transformed datasets (X_train, X_test, X_val, y_train, y_test, y_val) as CSV files.
6.  **Output**: Print a JSON object to standard output containing the file paths of the saved datasets.

**CRITICAL TECHNICAL NOTE**: When using `ColumnTransformer`, the number of output columns will change. After fitting the transformer (e.g., `preprocessor.fit(X_train)`), you **MUST** use `preprocessor.get_feature_names_out()` to get the new column names. Use these new names when creating the DataFrames for the transformed X_train, X_test, and X_val data to prevent a shape mismatch error.

Please generate only the Python script itself, without any explanations.
"""

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        
        messages = [
            {"role": "system", "content": "You are a data engineering expert who writes and debugs Python scripts for data preprocessing."},
            {"role": "user", "content": prompt}
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            response = client.chat.completions.create(
                model="qwen/qwen3-coder:free", # Using a more powerful model for better debugging
                messages=messages,
                max_tokens=2500
            )
            
            generated_text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": generated_text})

            code_match = re.search(r'```(?:python)?\s*(.*?)```', generated_text, re.DOTALL)
            
            if not code_match:
                print(f"[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries} failed: LLM did not return a Python code block.")
                feedback = "Your response did not contain a Python code block. Please generate only the Python code, enclosed in a ```python ... ``` block."
                messages.append({"role": "user", "content": feedback})
                if attempt == max_retries - 1:
                    raise Exception("No code found in response after multiple attempts.")
                continue

            code = code_match.group(1).strip()

            sbx = None
            try:
                sbx = Sandbox.create()
                with open(file_path, "rb") as f:
                    remote_path = sbx.files.write(os.path.basename(file_path), f).path
                
                code_to_run = code.replace(file_path, remote_path)
                execution = sbx.run_code(code_to_run, timeout=600)
            
                if getattr(execution, "error", None):
                    err = execution.error
                    error_details = (
                        f"Error Name: {getattr(err, 'name', 'ExecutionError')}\n"
                        f"Error Value: {getattr(err, 'value', str(err))}\n"
                        f"Traceback:\n{getattr(err, 'traceback', '')}"
                    )
                    print(f"\n[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries} failed: Code execution error.")
                    print(error_details)

                    feedback = (
                        "Your code failed to execute. Here is the error:\n\n"
                        f"{error_details}\n\n"
                        "Please fix the code and provide the corrected full script."
                    )
                    messages.append({"role": "user", "content": feedback})
                    if attempt == max_retries - 1:
                        raise Exception(f"Code execution failed after {max_retries} attempts. Last error: {getattr(err, 'value', str(err))}")
                    continue

                # Success!
                output = None
                for attr in ["text", "output", "stdout", "result"]:
                    if hasattr(execution, attr):
                        val = getattr(execution, attr)
                        if val:
                            output = val
                            break
                
                if output is None:
                    raise Exception("Could not find output from code execution. The script ran but produced no output.")

                processed_files = json.loads(output)
                
                download_urls = {}
                for name, path in processed_files.items():
                    download_urls[name] = sbx.download_url(path, timeout=600)
                
                # On success, return the sandbox so the caller can close it.
                return {
                    "download_urls": download_urls,
                    "remote_paths": processed_files,
                    "sandbox": sbx,
                }
            except Exception as e:
                if sbx:
                    sbx.kill()
                
                print(f"\n[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries} failed: An unexpected exception occurred.")
                print(f"Exception: {e}")

                feedback = f"An unexpected error occurred during execution: {e}\nThis might be an issue with the generated code or the environment. Please review and fix."
                messages.append({"role": "user", "content": feedback})

                if attempt == max_retries - 1:
                    raise e

        raise Exception("Failed to generate and execute valid code after multiple attempts.")

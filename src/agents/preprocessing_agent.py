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

        prompt = f"""
You are an expert data scientist tasked with writing a Python script to preprocess a dataset for model training.

Here is the context:
1.  **Data Exploration Summary**:
    ```
    {exploration_summary}
    ```
2.  **Selected Model**:
    - Model: {model_selection.get('recommended_model')}
    - Task Type: {model_selection.get('task_type')}
    - Reasoning: {model_selection.get('reasoning')}

The dataset is located in the sandbox at: `{file_path}`
The name of the target column is `{target_column}`.

**Your task is to generate a Python script that performs the following preprocessing steps:**
1.  Load the dataset from the provided path.
2.  Identify the target variable using the provided target column name.
3.  **Handle Outliers**: Detect and remove outliers from numerical columns.
4.  **Data Splitting**: Split the data into training (80%), testing (15%), and validation (5%) sets.
5.  **Preprocessing Pipeline**: Create a `ColumnTransformer` pipeline from `sklearn.compose`.
    - For numerical columns: Apply scaling and normalization.
    - For categorical columns: Apply one-hot encoding.
6.  **Apply Transformations** to the datasets.
7.  **Save Processed Data**: Save the transformed datasets (X_train, X_test, X_val, y_train, y_test, y_val) as CSV files.
8.  **Output**: Print a JSON object to standard output containing the file paths of the saved datasets.

**Important**: After applying the `ColumnTransformer`, the output will be a NumPy array. Use `preprocessor.get_feature_names_out()` to get the correct column names for the transformed data. Ensure the shape of the transformed data matches the number of feature names before creating a DataFrame.

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
                    feedback = (
                        "Your code failed to execute. Here is the error:\n\n"
                        f"Error Name: {getattr(err, 'name', 'ExecutionError')}\n"
                        f"Error Value: {getattr(err, 'value', str(err))}\n\n"
                        f"Traceback:\n{getattr(err, 'traceback', '')}\n\n"
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
                    "sandbox": sbx,
                }
            except Exception as e:
                feedback = f"An unexpected error occurred during execution: {e}\nThis might be an issue with the generated code or the environment. Please review and fix."
                messages.append({"role": "user", "content": feedback})

                if attempt == max_retries - 1:
                    raise e
            finally:
                if sbx:
                    sbx.close()

        raise Exception("Failed to generate and execute valid code after multiple attempts.")

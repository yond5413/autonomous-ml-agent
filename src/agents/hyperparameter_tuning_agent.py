import os
import re
import json
from e2b_code_interpreter import Sandbox
from openai import OpenAI
from .base import Agent

class HyperparameterTuningAgent(Agent):
    def execute(self, model_selection: dict, preprocessing_output: dict, **kwargs):
        # 1. Extract necessary inputs
        download_urls = preprocessing_output["download_urls"]
        sandbox = preprocessing_output["sandbox"]
        model_name = model_selection.get("recommended_model")
        task_type = model_selection.get("task_type")

        prompt = f"""
You are an expert data scientist tasked with writing a Python script to find the optimal hyperparameters for a machine learning model using RandomizedSearchCV.

Here is the context:
1.  **Model to be tuned**: {model_name}
2.  **Task Type**: {task_type}
3.  **Datasets**: The datasets are available at the following URLs. Your script must load the data directly from these URLs.
    - X_train_url: {download_urls['X_train']}
    - y_train_url: {download_urls['y_train']}

**Your task is to generate a Python script that performs the following steps:**
1.  **Load Data**: Load the training data (X_train and y_train) from the provided URLs using pandas.
2.  **Define Search Space**: Based on the model (`{model_name}`), define a comprehensive but reasonable hyperparameter search space. For example, for RandomForest, include n_estimators, max_depth, etc.
3.  **Instantiate Model and Tuner**: 
    - Import the model class from scikit-learn.
    - Import `RandomizedSearchCV` from `sklearn.model_selection`.
    - Instantiate `RandomizedSearchCV` with the model, the parameter distribution, `n_iter=50`, `cv=5`, `n_jobs=-1`, and `random_state=42` for reproducibility.
4.  **Run Tuning**: Fit the `RandomizedSearchCV` object on the training data.
5.  **Extract Best Parameters**: After fitting, get the best hyperparameters from the `.best_params_` attribute of the tuner.
6.  **Output**: Print a JSON object to standard output containing one key, `"best_params"`, with the dictionary of the best parameters as its value.

Please generate only the Python script itself, without any explanations.
"""

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        
        messages = [
            {"role": "system", "content": "You are a data engineering expert who writes and debugs Python scripts for hyperparameter tuning."},
            {"role": "user", "content": prompt}
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            response = client.chat.completions.create(
                model="qwen/qwen3-coder:free",
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
                execution = sandbox.run_code(code, timeout=1800) # 30 minute timeout for tuning

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
                        f"Your code failed to execute. Here is the error:\n\n"
                        f"{error_details}\n\n"
                        "Please fix the code and provide the corrected full script."
                    )
                    messages.append({"role": "user", "content": feedback})
                    if attempt == max_retries - 1:
                        raise Exception(f"Code execution failed after {max_retries} attempts. Last error: {getattr(err, 'value', str(err))}")
                    continue

                # Success
                output = execution.stdout
                if not output:
                    raise Exception("The script ran but produced no output.")

                tuning_results = json.loads(output)
                return tuning_results

            except Exception as e:
                print(f"\n[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries} failed: An unexpected exception occurred.")
                print(f"Exception: {e}")
                feedback = f"An unexpected error occurred during execution: {e}"
                messages.append({"role": "user", "content": feedback})
                if attempt == max_retries - 1:
                    raise e
        
        raise Exception("Failed to generate and execute valid code after multiple attempts.")

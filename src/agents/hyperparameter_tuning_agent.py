import os
import re
import json
from e2b_code_interpreter import Sandbox
from openai import OpenAI
from .base import Agent

class HyperparameterTuningAgent(Agent):
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

        prompt = f"""
You are an expert data scientist tasked with writing a Python script to find the optimal hyperparameters for a machine learning model.

You can use either RandomizedSearchCV or a Bayesian optimization library like Optuna. Optuna is available in the environment.

Here is the context:
1.  **Model to be tuned**: {model_name}
2.  **Task Type**: {task_type}
3.  **Data Exploration Summary**:
    ```
    {preprocessing_output["exploration_summary"]}
    ```
4.  **Datasets**: The datasets are available at the following absolute file paths inside the environment. Your script must load the data directly from these file paths (not URLs).
    - X_train_path: {x_train_path}
    - y_train_path: {y_train_path}

**Your task is to generate a Python script that performs the following steps:**
1.  **Load Data**: At the top of the script, assign variables for the file paths exactly as provided above (e.g., `X_train_path = r"{x_train_path}"`). Then load the training data using `pd.read_csv(X_train_path)` and `pd.read_csv(y_train_path)`.
2.  **Define Search Space**: Based on the model (`{model_name}`) and the data summary, define a comprehensive but reasonable hyperparameter search space. Be mindful of the dataset size and model complexity to avoid excessively long tuning times.
3.  **Instantiate Model and Tuner**: 
    - Import the model class from scikit-learn.
    - Choose and import a suitable hyperparameter tuning library (`RandomizedSearchCV` or `Optuna`).
    - Instantiate the tuner with the model and the parameter distribution.
4.  **Run Tuning**: Fit the tuner object on the training data.
5.  **Extract Best Parameters**: After fitting, get the best hyperparameters.
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
                model="deepseek/deepseek-chat-v3.1:free",
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
                execution = sandbox.run_code(code, timeout=600) # 10 minute timeout for tuning

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

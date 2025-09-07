import os
import re
import json
from e2b_code_interpreter import Sandbox
from openai import OpenAI
from .base import Agent

class SimpleTrainingAgent(Agent):
    def execute(self, model_selection: dict, preprocessing_output: dict, **kwargs):
        # 1. Extract necessary inputs
        download_urls = preprocessing_output["download_urls"]
        sandbox = preprocessing_output["sandbox"]
        model_name = model_selection.get("recommended_model")
        task_type = model_selection.get("task_type")

        prompt = f"""
You are an expert data scientist tasked with writing a Python script to train a machine learning model and evaluate it.

Here is the context:
1.  **Model to be trained**: {model_name}
2.  **Task Type**: {task_type}
3.  **Datasets**: The datasets are available at the following URLs. Your script must load the data directly from these URLs.
    - X_train_url: {download_urls['X_train']}
    - y_train_url: {download_urls['y_train']}
    - X_test_url: {download_urls['X_test']}
    - y_test_url: {download_urls['y_test']}
    - X_val_url: {download_urls['X_val']}
    - y_val_url: {download_urls['y_val']}

**Your task is to generate a Python script that performs the following steps:**
1.  **Load Data**: Load all six datasets from the provided URLs using pandas.
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
                execution = sandbox.run_code(code, timeout=1200) # 20 minute timeout for training

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

                training_results = json.loads(output)
                
                model_path = training_results.get("model_path")
                if model_path:
                    training_results["model_download_url"] = sandbox.download_url(model_path, timeout=600)

                return training_results

            except Exception as e:
                print(f"\n[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries} failed: An unexpected exception occurred.")
                print(f"Exception: {e}")
                feedback = f"An unexpected error occurred during execution: {e}"
                messages.append({"role": "user", "content": feedback})
                if attempt == max_retries - 1:
                    raise e
        
        raise Exception("Failed to generate and execute valid code after multiple attempts.")

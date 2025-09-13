from src.agents.data_exploration_agent import DataExplorationAgent
from src.agents.model_selection_agent import ModelSelectionAgent
from src.agents.preprocessing_agent import PreprocessingAgent
from src.agents.hyperparameter_tuning_agent import HyperparameterTuningAgent
from src.agents.tuned_training_agent import TunedTrainingAgent
from src.agents.deployment_agent import DeploymentAgent
import json
import os
from openai import OpenAI
import inspect

class Orchestrator:
    def __init__(self):
        self.agents = {
            "data_exploration": DataExplorationAgent(),
            "model_selection": ModelSelectionAgent(),
            "preprocessing": PreprocessingAgent(),
            "hyperparameter_tuning": HyperparameterTuningAgent(),
            "tuned_training": TunedTrainingAgent(),
            "deployment": DeploymentAgent(),
        }
        self.state = "IDLE"
        self.results = {}

    def run_pipeline(self, file_path, target_column):
        sandbox = None
        try:
            self._run_step(self._run_data_exploration, file_path=file_path, target_column=target_column)
            self._run_step(self._run_model_selection, file_path=file_path, target_column=target_column)
            self._run_step(self._run_preprocessing, file_path=file_path, target_column=target_column)

            leaderboard = []
            for model in self.results["model_selection"]["recommended_models"]:
                model_name = model["model_name"]
                print(f"--- Running training for model: {model_name} ---")
                training_results = self._run_training_for_model(model)
                leaderboard.append(training_results)
            
            self.results["leaderboard"] = sorted(leaderboard, key=lambda x: x["metrics"].get("f1_score", x["metrics"].get("r2_score", 0)), reverse=True)

            # Run deployment for the best model
            best_model_results = self.results["leaderboard"][0]
            self._run_step(self._run_deployment, best_model_results=best_model_results)

            self.state = "IDLE"
            print(f"State: {self.state}")
            
            # Clean up sandbox object before returning
            if "preprocessing" in self.results and "sandbox" in self.results["preprocessing"]:
                sandbox = self.results["preprocessing"]["sandbox"]
                del self.results["preprocessing"]["sandbox"]

            return self.results

        except Exception as e:
            self.state = "ERROR"
            print(f"Pipeline failed: {e}")
            raise e
        finally:
            if sandbox:
                print("Killing sandbox...")
                sandbox.kill()

    def _run_data_exploration(self, file_path, target_column, **kwargs):
        self.state = "DATA EXPLORATION"
        print(f"State: {self.state}")
        result = self.agents["data_exploration"].execute(file_path=file_path, target_column=target_column, **kwargs)
        if not result:
            raise Exception("Data Exploration Agent returned no result.")
        self.results["data_exploration"] = result

    def _run_model_selection(self, file_path, target_column, **kwargs):
        self.state = "MODEL_SELECTION"
        print(f"State: {self.state}")
        result_str = self.agents["model_selection"].execute(
            exploration_summary=self.results["data_exploration"],
            target_column=target_column,
            **kwargs
        )
        if not result_str:
            raise Exception("Model Selection Agent returned no result.")
        self.results["model_selection"] = json.loads(result_str)

    def _run_preprocessing(self, file_path, target_column, **kwargs):
        self.state = "PREPROCESSING"
        print(f"State: {self.state}")
        result = self.agents["preprocessing"].execute(
            model_selection_result=json.dumps(self.results["model_selection"]),
            exploration_summary=self.results["data_exploration"],
            file_path=file_path,
            target_column=target_column,
            **kwargs
        )
        if not result or "sandbox" not in result:
            raise Exception("Preprocessing Agent failed to return a sandbox instance.")
        self.results["preprocessing"] = result
        
        # Download processed datasets
        self._download_sandbox_files()

    def _download_sandbox_files(self):
        sandbox = self.results["preprocessing"]["sandbox"]
        remote_paths = self.results["preprocessing"].get("remote_paths")
        if remote_paths:
            local_dataset_paths = {}
            output_dir = "outputs/processed_data"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Downloading processed datasets to {output_dir}...")
            for name, remote_path in remote_paths.items():
                file_bytes = sandbox.files.read(remote_path)
                if name == "preprocessor":
                    local_path = os.path.join("outputs", "preprocessor.joblib")
                else:
                    local_path = os.path.join(output_dir, f"{name}.csv")
                with open(local_path, "wb") as f:
                    f.write(file_bytes)
                local_dataset_paths[name] = local_path
            print("Downloads complete.")
            self.results["preprocessing"]["local_dataset_paths"] = local_dataset_paths

    def _run_training_for_model(self, model, **kwargs):
        model_selection = {
            "recommended_model": model["model_name"],
            "task_type": self.results["model_selection"]["task_type"],
        }

        tuning_result = self._run_step(self._run_hyperparameter_tuning, model_selection=model_selection, **kwargs)
        training_result = self._run_step(self._run_tuned_training, model_selection=model_selection, tuning_results=tuning_result, **kwargs)

        return training_result

    def _run_hyperparameter_tuning(self, model_selection, **kwargs):
        self.state = "HYPERPARAMETER_TUNING"
        print(f"State: {self.state}")
        result = self.agents["hyperparameter_tuning"].execute(
            model_selection=model_selection,
            preprocessing_output=self.results["preprocessing"],
            **kwargs
        )
        if not result:
            raise Exception("Hyperparameter Tuning Agent returned no result.")
        return result

    def _run_tuned_training(self, model_selection, tuning_results, **kwargs):
        self.state = "TUNED_TRAINING"
        print(f"State: {self.state}")
        result = self.agents["tuned_training"].execute(
            model_selection=model_selection,
            preprocessing_output=self.results["preprocessing"],
            tuning_results=tuning_results,
            **kwargs
        )
        if not result:
            raise Exception("Tuned Training Agent returned no result.")
        
        # Download the final model
        self._download_final_model(result, model_selection["recommended_model"])
        return result

    def _download_final_model(self, training_results, model_name):
        sandbox = self.results["preprocessing"]["sandbox"]
        final_model_path = training_results.get("model_path")
        if final_model_path:
            print(f"Downloading final model from sandbox path: {final_model_path}")
            model_bytes = sandbox.files.read(final_model_path)
            local_model_path = f"{model_name}.pickle"
            with open(local_model_path, "wb") as f:
                f.write(model_bytes)
            print(f"Final model saved locally to: {local_model_path}")
            del training_results["model_path"]
            if "model_download_url" in training_results:
                del training_results["model_download_url"]
            training_results["local_model_path"] = local_model_path

    def _run_deployment(self, best_model_results, **kwargs):
        self.state = "DEPLOYMENT"
        print(f"State: {self.state}")
        result = self.agents["deployment"].execute(
            preprocessing_output=self.results["preprocessing"],
            tuned_training_output=best_model_results,
            **kwargs
        )
        self.results["deployment"] = result

    def _run_step(self, step_func, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return step_func(**kwargs)
            except Exception as e:
                print(f"An error occurred in step {step_func.__name__} (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"Step {step_func.__name__} failed after {max_retries} attempts.")
                    raise e

                action, new_params = self._handle_error(step_func, e, **kwargs)

                if action == "retry_with_new_params":
                    print("Retrying with new parameters from LLM.")
                    if 'file_path' in new_params:
                        print("LLM attempted to change file_path, but this is not allowed. Ignoring file_path change.")
                        del new_params['file_path']
                    kwargs.update(new_params)
                elif action == "retry":
                    print("Retrying the step.")
                else: # fail
                    print("LLM suggested to fail. Raising exception.")
                    raise e

    def _handle_error(self, step_func, error, **kwargs):
        print(f"Handling error in {step_func.__name__} step...")
        prompt = self._create_error_handling_prompt(step_func.__name__, error, kwargs)
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        response = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system", "content": "You are an expert system that helps resolve errors in a machine learning pipeline."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        solution_json = json.loads(response.choices[0].message.content)
        action = solution_json.get("action")
        params = solution_json.get("params", {})
        return action, params

    def _create_error_handling_prompt(self, step_name, error, params):
        # Sanitize params for prompt to avoid excessive length
        safe_params = {k: str(v)[:500] for k, v in params.items()}

        return f'''
        An error occurred in the '{step_name}' step of a machine learning pipeline.

        **Error:**
        ```
        {error}
        ```

        **Parameters passed to the step:**
        ```
        {safe_params}
        ```

        Please provide a solution as a JSON object with the following format:
        {{
            "action": "<ACTION>",
            "params": {{ <NEW_PARAMS> }}
        }}

        **<ACTION>** can be one of the following:
        - `retry`: If you think the error is transient and the step should be retried with the same parameters.
        - `retry_with_new_params`: If the error can be fixed by modifying the parameters. Provide the new parameters in the `params` field.
        - `fail`: If the error is unrecoverable and the pipeline should be stopped.

        **<NEW_PARAMS>** should be a dictionary of the new parameters to use. Only include the parameters that need to be changed.
        '''
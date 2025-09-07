from src.agents.data_exploration_agent import DataExplorationAgent
from src.agents.model_selection_agent import ModelSelectionAgent
from src.agents.preprocessing_agent import PreprocessingAgent
from src.agents.hyperparameter_tuning_agent import HyperparameterTuningAgent
from src.agents.tuned_training_agent import TunedTrainingAgent
import json
import os

class Orchestrator:
    def __init__(self):
        self.data_exploration_agent = DataExplorationAgent()
        self.model_selection_agent = ModelSelectionAgent()
        self.preprocessing_agent = PreprocessingAgent()
        self.hyperparameter_tuning_agent = HyperparameterTuningAgent()
        self.tuned_training_agent = TunedTrainingAgent()
        self.state = "IDLE"

    def run_pipeline(self, file_path, target_column):
        sandbox = None
        try:
            self.state = "DATA EXPLORATION"
            print(f"State: {self.state}")
            data_exploration_result = self.data_exploration_agent.execute(file_path, target_column)
            if not data_exploration_result:
                raise Exception("Data Exploration Agent returned no result.")

            self.state = "MODEL_SELECTION"
            print(f"State: {self.state}")
            model_selection_result_str = self.model_selection_agent.execute(
                exploration_summary=data_exploration_result,
                target_column=target_column
            )
            if not model_selection_result_str:
                raise Exception("Model Selection Agent returned no result.")
            model_selection_result = json.loads(model_selection_result_str)

            self.state = "PREPROCESSING"
            print(f"State: {self.state}")
            preprocessing_result = self.preprocessing_agent.execute(
                model_selection_result=model_selection_result_str,
                exploration_summary=data_exploration_result,
                file_path=file_path,
                target_column=target_column
            )
            if not preprocessing_result or "sandbox" not in preprocessing_result:
                raise Exception("Preprocessing Agent failed to return a sandbox instance.")
            
            sandbox = preprocessing_result["sandbox"]

            # Download processed datasets before killing the sandbox
            remote_paths = preprocessing_result.get("remote_paths")
            if remote_paths:
                local_dataset_paths = {}
                output_dir = "outputs/processed_data"
                os.makedirs(output_dir, exist_ok=True)
                print(f"Downloading processed datasets to {output_dir}...")
                for name, remote_path in remote_paths.items():
                    file_bytes = sandbox.files.read(remote_path)
                    local_path = os.path.join(output_dir, f"{name}.csv")
                    with open(local_path, "wb") as f:
                        f.write(file_bytes)
                    local_dataset_paths[name] = local_path
                print("Downloads complete.")
                preprocessing_result["local_dataset_paths"] = local_dataset_paths

            self.state = "HYPERPARAMETER_TUNING"
            print(f"State: {self.state}")
            tuning_result = self.hyperparameter_tuning_agent.execute(
                model_selection=model_selection_result,
                preprocessing_output=preprocessing_result
            )
            if not tuning_result:
                raise Exception("Hyperparameter Tuning Agent returned no result.")

            self.state = "TUNED_TRAINING"
            print(f"State: {self.state}")
            training_result = self.tuned_training_agent.execute(
                model_selection=model_selection_result,
                preprocessing_output=preprocessing_result,
                tuning_results=tuning_result
            )
            if not training_result:
                raise Exception("Tuned Training Agent returned no result.")

            # Download the final model before killing the sandbox
            final_model_path = training_result.get("model_path")
            if final_model_path:
                print(f"Downloading final model from sandbox path: {final_model_path}")
                model_bytes = sandbox.files.read(final_model_path)
                local_model_path = "tuned_model.pickle"
                with open(local_model_path, "wb") as f:
                    f.write(model_bytes)
                print(f"Final model saved locally to: {local_model_path}")
                # Replace sandbox path and URL with local path for the final output
                del training_result["model_path"]
                if "model_download_url" in training_result:
                    del training_result["model_download_url"]
                training_result["local_model_path"] = local_model_path

            self.state = "IDLE"
            print(f"State: {self.state}")
            # Clean up sandbox object before returning
            del preprocessing_result["sandbox"]
            return {
                "data_exploration": data_exploration_result,
                "model_selection": model_selection_result,
                "preprocessing": preprocessing_result,
                "hyperparameter_tuning": tuning_result,
                "tuned_training": training_result,
            }

        except Exception as e:
            self.state = "ERROR"
            print(f"Pipeline failed at state {self.state}: {e}")
            raise e
        finally:
            if sandbox:
                print("Killing sandbox...")
                sandbox.kill()

from src.agents.data_exploration_agent import DataExplorationAgent
from src.agents.model_selection_agent import ModelSelectionAgent
from src.agents.preprocessing_agent import PreprocessingAgent

class Orchestrator:
    def __init__(self):
        self.data_exploration_agent = DataExplorationAgent()
        self.model_selection_agent = ModelSelectionAgent()
        self.preprocessing_agent = PreprocessingAgent()
        self.state = "IDLE"

    def run_pipeline(self, file_path, target_column):
        try:
            self.state = "DATA EXPLORATION"
            print(f"State: {self.state}")
            data_exploration_result = self.data_exploration_agent.execute(file_path, target_column)
            if not data_exploration_result:
                raise Exception("Data Exploration Agent returned no result.")

            self.state = "MODEL_SELECTION"
            print(f"State: {self.state}")
            model_selection_result = self.model_selection_agent.execute(
                exploration_summary=data_exploration_result,
                target_column=target_column
            )
            if not model_selection_result:
                raise Exception("Model Selection Agent returned no result.")

            self.state = "PREPROCESSING"
            print(f"State: {self.state}")
            preprocessing_agent_result = self.preprocessing_agent.execute(
                model_selection_result=model_selection_result,
                exploration_summary=data_exploration_result,
                file_path=file_path,
                target_column=target_column
            )
            if not preprocessing_agent_result:
                raise Exception("Preprocessing Agent returned no result.")

            self.state = "IDLE"
            print(f"State: {self.state}")
            return {
                "data_exploration": data_exploration_result,
                "model_selection": model_selection_result,
                "preprocessing": preprocessing_agent_result
            }

        except Exception as e:
            self.state = "ERROR"
            print(f"Pipeline failed at state {self.state}: {e}")
            # Re-raise the exception to be caught by the Streamlit app
            raise e

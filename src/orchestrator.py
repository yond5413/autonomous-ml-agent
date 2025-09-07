from src.agents.preprocessing_agent import PreprocessingAgent
from src.agents.model_selection_agent import ModelSelectionAgent
from src.agents.preprocessing_transformer_agent import PreprocessingTransformerAgent

class Orchestrator:
    def __init__(self):
        self.preprocessing_agent = PreprocessingAgent()
        self.model_selection_agent = ModelSelectionAgent()
        self.preprocessing_transformer_agent = PreprocessingTransformerAgent()
        self.state = "IDLE"

    def run_pipeline(self, df, target_column):
        self.state = "PREPROCESSING"
        print(f"State: {self.state}")
        preprocessing_result = self.preprocessing_agent.execute(df, target_column)

        if preprocessing_result:
            self.state = "MODEL_SELECTION"
            print(f"State: {self.state}")
            model_selection_result = self.model_selection_agent.execute(preprocessing_result)

            if model_selection_result:
                self.state = "PREPROCESSING_TRANSFORMATION"
                print(f"State: {self.state}")
                self.preprocessing_transformer_agent.execute(model_selection_result)

        self.state = "IDLE"
        print(f"State: {self.state}")
        return preprocessing_result

from .base import Agent
from openai import OpenAI
import os
import re
import json
'''
Could add steps for preprocessing like suggested steps, so next agent is not blindly attempting stuff
'''
class ModelSelectionAgent(Agent):
    def execute(self, exploration_summary: str, target_column: str, **kwargs):
        prompt = f"""
As an elite data scientist, you are tasked with selecting the best machine learning models and outlining a preprocessing strategy.

The allowed models are: LogisticRegression, LinearRegression, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, KNeighborsClassifier, KNeighborsRegressor, MLPClassifier, MLPRegressor.

The target column for our model is `{target_column}`.

Here is the summary from the data exploration phase:

{exploration_summary}

Based on this information, please provide the following:
1.  **Recommend the top 3 best model types** from the allowed list.
2.  **Outline a detailed, step-by-step preprocessing plan** that would be suitable for all recommended models.

**When creating the `preprocessing_plan`, please consider including steps for the following where appropriate, based on the data summary:**
- **Outlier Handling**: A strategy for detecting and managing outliers in numerical columns.
- **Numerical Features**: A strategy for scaling or normalizing numerical columns (e.g., using `StandardScaler`).
- **Categorical Features**: A strategy for encoding categorical columns (e.g., `OneHotEncoder`).
- **Pipeline Structure**: It is highly recommended that the plan specifies using a `sklearn.compose.ColumnTransformer` to apply different transformations to different columns. This is a best practice.
- **Technical Note for ColumnTransformer**: If a `ColumnTransformer` is used, the plan should mention that its output needs to be converted back to a DataFrame with correct column names, for example by using the transformer's `get_feature_names_out()` method.

Your response must be a JSON object with the following structure:
{{
  "recommended_models": [
    {{
      "model_name": "MODEL_NAME_1",
      "reasoning": "Your detailed reasoning for choosing this model."
    }},
    {{
      "model_name": "MODEL_NAME_2",
      "reasoning": "Your detailed reasoning for choosing this model."
    }},
    {{
      "model_name": "MODEL_NAME_3",
      "reasoning": "Your detailed reasoning for choosing this model."
    }}
  ],
  "task_type": "classification or regression",
  "preprocessing_plan": [
    "Step 1: Description of preprocessing step.",
    "Step 2: Description of preprocessing step."
  ]
}} """
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {"role": "system", "content": "You are an elite data scientist specializing in model selection."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content

        # Extract JSON from markdown
        json_match = re.search(r'```json\s*\n(.*?)```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        print(response_text)
        return response_text

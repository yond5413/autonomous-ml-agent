from .base import Agent
from openai import OpenAI
import os
import re
import json

class ModelSelectionAgent(Agent):
    def execute(self, exploration_summary: str, target_column: str, **kwargs):
        prompt = f"""
As an elite data scientist, you are tasked with selecting the best machine learning model for a given dataset.

The target column for our model is `{target_column}`.

Here is the summary from the data exploration phase:

{exploration_summary}

Based on this information, please recommend the best model type (e.g., RandomForestClassifier, XGBoostRegressor, LogisticRegression, etc.).

Your response should be a JSON object with the following structure:
{{
  "recommended_model": "MODEL_NAME",
  "reasoning": "Your detailed reasoning for choosing this model.",
  "task_type": "classification or regression"
}}
"""
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

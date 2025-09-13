import os
from .base import Agent
from openai import OpenAI
import re

class DeploymentAgent(Agent):
    def __init__(self):
        super().__init__()

    def execute(self, preprocessing_output, tuned_training_output, **kwargs):
        preprocessor_path = preprocessing_output.get("local_dataset_paths", {}).get("preprocessor")
        model_path = tuned_training_output.get("local_model_path")

        if not preprocessor_path or not model_path:
            raise ValueError("Missing preprocessor or model path for deployment.")

        # Create deployment directory
        os.makedirs("deployment", exist_ok=True)

        # Copy model and preprocessor to deployment directory
        new_preprocessor_path = os.path.join("deployment", os.path.basename(preprocessor_path))
        new_model_path = os.path.join("deployment", os.path.basename(model_path))
        os.rename(preprocessor_path, new_preprocessor_path)
        os.rename(model_path, new_model_path)

        prompt = self._create_fastapi_prompt(new_preprocessor_path, new_model_path)
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        response = client.chat.completions.create(
            model="qwen/qwen3-coder:free",
            messages=[
                {"role": "system", "content": "You are an expert in deploying machine learning models as FastAPI services."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
        )

        fastapi_code = self._extract_code(response.choices[0].message.content)
        with open("deployment/app.py", "w") as f:
            f.write(fastapi_code)

        dockerfile_content = self._create_dockerfile()
        with open("deployment/Dockerfile", "w") as f:
            f.write(dockerfile_content)
            
        readme_content = self._create_readme()
        with open("deployment/README.md", "w") as f:
            f.write(readme_content)

        return {
            "deployment_path": "deployment",
            "fastapi_app_path": "deployment/app.py",
            "dockerfile_path": "deployment/Dockerfile",
            "readme_path": "deployment/README.md"
        }

    def _create_fastapi_prompt(self, preprocessor_path, model_path):
        return f"""
        Generate a Python script for a FastAPI application to serve a machine learning model.

        The application should:
        1. Load a preprocessor from `{preprocessor_path}`.
        2. Load a trained model from `{model_path}`.
        3. Have a `/predict` endpoint that accepts POST requests with JSON data.
        4. The endpoint should use the preprocessor to transform the input data.
        5. The endpoint should use the model to make predictions on the transformed data.
        6. The endpoint should return the predictions as a JSON response.

        Assume the input data for the `/predict` endpoint will be a JSON object where the keys are the feature names and the values are the feature values.
        The script should be a complete, runnable FastAPI application.
        Make sure to include all necessary imports.
        """

    def _extract_code(self, text):
        code_match = re.search(r'```(?:python)?\s*(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return text

    def _create_dockerfile(self):
        return """
        FROM python:3.9-slim

        WORKDIR /app

        COPY . /app

        RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn pandas

        EXPOSE 8000

        CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
        """
        
    def _create_readme(self):
        return """
        # Model Deployment

        This directory contains a deployed machine learning model as a FastAPI application.

        ## How to Run

        1. **Build the Docker image:**
           ```bash
           docker build -t ml-deployment .
           ```

        2. **Run the Docker container:**
           ```bash
           docker run -p 8000:8000 ml-deployment
           ```

        ## API

        ### POST /predict

        Send a POST request with JSON data to get a prediction.

        **Example using curl:**
        ```bash
        curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
          "feature1": "value1",
          "feature2": "value2"
        }'
        ```
        """

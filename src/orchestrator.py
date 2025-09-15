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

            # Complete pipeline for each model: simple training -> hyperparameter tuning -> tuned training
            baseline_leaderboard = []
            tuned_leaderboard = []
            for model in self.results["model_selection"]["recommended_models"]:
                model_name = model["model_name"]
                
                # Step 1: Simple training
                print(f"--- Running simple training for model: {model_name} ---")
                baseline_results = self._run_simple_training(model)
                baseline_results["phase"] = "baseline"
                baseline_results["model_name"] = model_name
                baseline_leaderboard.append(baseline_results)
                
                # Step 2: Hyperparameter tuning + tuned training
                print(f"--- Running hyperparameter tuning for model: {model_name} ---")
                print(f"--- Running tuned training for model: {model_name} ---")
                tuned_results = self._run_training_for_model(model)
                tuned_results["phase"] = "tuned"
                tuned_results["model_name"] = model_name
                tuned_leaderboard.append(tuned_results)
            
            self.results["baseline_leaderboard"] = sorted(
                baseline_leaderboard,
                key=lambda x: x["metrics"].get("f1_score", x["metrics"].get("r2_score", 0)),
                reverse=True,
            )
            self.results["leaderboard"] = sorted(
                tuned_leaderboard,
                key=lambda x: x["metrics"].get("f1_score", x["metrics"].get("r2_score", 0)),
                reverse=True,
            )

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
        if not remote_paths:
            return

        # Canonicalize keys to expected names
        def pick(keys, mapping):
            for k in keys:
                if k in mapping and isinstance(mapping[k], str):
                    return mapping[k]
            # try fuzzy by substring
            for key, val in mapping.items():
                if isinstance(val, str) and any(k in key for k in keys):
                    return val
            return None

        canonical = {
            "X_train": pick(["X_train", "train_data_path", "train_data", "train_features"], remote_paths),
            "y_train": pick(["y_train", "train_labels_path", "train_labels", "train_target"], remote_paths),
            "X_test": pick(["X_test", "test_data_path", "test_data", "test_features"], remote_paths),
            "y_test": pick(["y_test", "test_labels_path", "test_labels", "test_target"], remote_paths),
            "preprocessor": pick(["preprocessor", "preprocessor_path"], remote_paths),
        }

        # If validation not present, derive from training inside sandbox
        if not remote_paths.get("X_val") or not remote_paths.get("y_val"):
            if canonical["X_train"] and canonical["y_train"]:
                split_code = (
                    "import os, json, pandas as pd\n"
                    f"x_path = r'{canonical['X_train']}'\n"
                    f"y_path = r'{canonical['y_train']}'\n"
                    "X = pd.read_csv(x_path)\n"
                    "y = pd.read_csv(y_path)\n"
                    "# Ensure aligned indices\n"
                    "X = X.reset_index(drop=True)\n"
                    "y = y.reset_index(drop=True)\n"
                    "val_idx = X.sample(frac=0.2, random_state=42).index\n"
                    "X_val = X.loc[val_idx]\n"
                    "y_val = y.loc[val_idx]\n"
                    "X_train2 = X.drop(index=val_idx).reset_index(drop=True)\n"
                    "y_train2 = y.drop(index=val_idx).reset_index(drop=True)\n"
                    "x_train_out = os.path.abspath('X_train_split.csv')\n"
                    "y_train_out = os.path.abspath('y_train_split.csv')\n"
                    "x_val_out = os.path.abspath('X_val_split.csv')\n"
                    "y_val_out = os.path.abspath('y_val_split.csv')\n"
                    "X_train2.to_csv(x_train_out, index=False)\n"
                    "y_train2.to_csv(y_train_out, index=False)\n"
                    "X_val.to_csv(x_val_out, index=False)\n"
                    "y_val.to_csv(y_val_out, index=False)\n"
                    "print(json.dumps({\"X_train\": x_train_out, \"y_train\": y_train_out, \"X_val\": x_val_out, \"y_val\": y_val_out}))\n"
                )
                exec_res = sandbox.run_code(split_code, timeout=180)
                out = getattr(exec_res, "stdout", "") or getattr(exec_res, "text", "") or ""
                try:
                    split_paths = json.loads(out.strip()) if out else {}
                    if split_paths.get("X_train") and split_paths.get("y_train"):
                        canonical["X_train"] = split_paths["X_train"]
                        canonical["y_train"] = split_paths["y_train"]
                        remote_paths.update({
                            "X_train": split_paths["X_train"],
                            "y_train": split_paths["y_train"],
                            "X_val": split_paths.get("X_val"),
                            "y_val": split_paths.get("y_val"),
                        })
                except Exception:
                    pass

        # Add canonical keys back to remote_paths if missing
        for k, v in canonical.items():
            if v and k not in remote_paths:
                remote_paths[k] = v

        # Helper to resolve absolute paths inside sandbox
        def resolve_abs(path: str) -> str:
            if not isinstance(path, str):
                return path
            if path.startswith("/"):
                return path
            try:
                code = (
                    "import os, json\n"
                    f"p = r'{path}'\n"
                    "print(os.path.abspath(p))\n"
                )
                res = sandbox.run_code(code, timeout=60)
                s = getattr(res, "stdout", "") or getattr(res, "text", "") or ""
                return s.strip() or path
            except Exception:
                return path

        # Prepare download URLs for later agents
        download_urls = {}
        for key in ["X_train", "y_train", "X_test", "y_test", "X_val", "y_val"]:
            path = remote_paths.get(key)
            if isinstance(path, str):
                try:
                    abs_path = resolve_abs(path)
                    download_urls[key] = sandbox.download_url(abs_path, timeout=600)
                except Exception:
                    # Fallback to absolute path inside sandbox
                    abs_path = resolve_abs(path)
                    download_urls[key] = abs_path

        self.results["preprocessing"]["download_urls"] = download_urls
        self.results["preprocessing"]["exploration_summary"] = self.results.get("data_exploration")

        # Download processed datasets locally (best-effort)
        local_dataset_paths = {}
        output_dir = "outputs/processed_data"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Downloading processed datasets to {output_dir}...")
        for name, remote_path in remote_paths.items():
            if not isinstance(remote_path, str):
                continue
            try:
                abs_path = resolve_abs(remote_path)
                file_bytes = sandbox.files.read(abs_path)
                if isinstance(file_bytes, str):
                    file_bytes = file_bytes.encode("utf-8")
                if name == "preprocessor":
                    local_path = os.path.join("outputs", "preprocessor.joblib")
                else:
                    local_path = os.path.join(output_dir, f"{name}.csv")
                with open(local_path, "wb") as f:
                    f.write(file_bytes)
                local_dataset_paths[name] = local_path
            except Exception as e:
                print(f"Failed to download '{name}' from '{remote_path}': {e}")
                continue
        print("Downloads complete.")
        self.results["preprocessing"]["local_dataset_paths"] = local_dataset_paths

    def _run_simple_training(self, model, **kwargs):
        from src.agents.simple_training_agent import SimpleTrainingAgent
        self.state = "SIMPLE_TRAINING"
        print(f"State: {self.state}")
        agent = SimpleTrainingAgent()
        model_selection = {
            "recommended_model": model["model_name"],
            "task_type": self.results["model_selection"]["task_type"],
        }
        # Retry with error_context propagation
        max_retries = 3
        last_error = None
        error_context = None
        for attempt in range(max_retries):
            try:
                params = dict(kwargs)
                if error_context:
                    params["error_context"] = error_context
                result = agent.execute(
                    model_selection=model_selection,
                    preprocessing_output=self.results["preprocessing"],
                    **params,
                )
                # clear any prior context on success
                error_context = None
                break
            except Exception as e:
                print(f"[SimpleTraining] Attempt {attempt + 1}/{max_retries} failed: {e}")
                last_error = e
                # Prefer agent-provided context if available
                agent_ctx = getattr(agent, "last_error_context", None)
                # Ask LLM planner for retry plan
                action, new_params = self._handle_error(self._run_simple_training, e, model=model, **kwargs)
                llm_ctx = new_params.get("error_context") if isinstance(new_params, dict) else None
                # Merge contexts (LLM first, then agent details)
                merged = {}
                if isinstance(llm_ctx, dict):
                    merged.update(llm_ctx)
                if isinstance(agent_ctx, dict):
                    for k, v in agent_ctx.items():
                        merged.setdefault(k, v)
                error_context = merged or {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "suggested_fixes": ["Inspect traceback and fix indicated line", "Ensure training and test preprocessing match"],
                }
                if action == "retry_with_new_params" or action == "retry":
                    continue
                else:
                    raise e
        else:
            raise last_error if last_error else Exception("Simple training failed after retries")

        if not result:
            raise Exception("Simple Training Agent returned no result.")

        # Download the baseline model if provided (prefer download URL; sandbox paths may be from a different sandbox)
        model_download_url = result.get("model_download_url")
        model_bytes = result.get("model_bytes")
        model_path = result.get("model_path")
        if model_bytes or model_download_url or model_path:
            models_dir = os.path.join('outputs', 'models')
            os.makedirs(models_dir, exist_ok=True)
            local_model_path = os.path.join(models_dir, f"{model['model_name']}_baseline.pickle")
            try:
                if model_bytes:
                    print(f"Saving baseline model from direct download")
                    with open(local_model_path, 'wb') as f:
                        f.write(model_bytes)
                    print(f"Baseline model saved locally to: {local_model_path}")
                    result["local_model_path"] = local_model_path
                elif model_download_url:
                    import requests
                    print(f"Downloading baseline model from URL: {model_download_url}")
                    resp = requests.get(model_download_url, timeout=600)
                    resp.raise_for_status()
                    with open(local_model_path, 'wb') as f:
                        f.write(resp.content)
                    print(f"Baseline model saved locally to: {local_model_path}")
                    result["local_model_path"] = local_model_path
                elif model_path:
                    # Best-effort: this path likely lives in the training sandbox which is not available here.
                    # Attempt to read from preprocessing sandbox only if it exists there; otherwise skip without failing the pipeline.
                    sandbox = self.results["preprocessing"].get("sandbox")
                    if sandbox:
                        print(f"Attempting to read baseline model from preprocessing sandbox path: {model_path}")
                        try:
                            model_bytes = sandbox.files.read(model_path)
                            if isinstance(model_bytes, str):
                                try:
                                    model_bytes = model_bytes.encode('latin-1')
                                except Exception:
                                    model_bytes = model_bytes.encode('utf-8', 'ignore')
                            with open(local_model_path, 'wb') as f:
                                f.write(model_bytes)
                            print(f"Baseline model saved locally to: {local_model_path}")
                            result["local_model_path"] = local_model_path
                        except Exception as e:
                            print(f"Could not read model from preprocessing sandbox: {e}. Skipping baseline download.")
                # Clean up fields we shouldn't propagate
                if "model_path" in result:
                    del result["model_path"]
                if "model_download_url" in result:
                    del result["model_download_url"]
            except Exception as e:
                print(f"Failed to download baseline model artifact: {e}. Proceeding without local baseline artifact.")

        return result

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
            if isinstance(model_bytes, str):
                try:
                    model_bytes = model_bytes.encode('latin-1')
                except Exception:
                    model_bytes = model_bytes.encode('utf-8', 'ignore')
            models_dir = os.path.join('outputs', 'models')
            os.makedirs(models_dir, exist_ok=True)
            local_model_path = os.path.join(models_dir, f"{model_name}.pickle")
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
        # Always provide fallback error_context for downstream tools
        if "error_context" not in params or not isinstance(params.get("error_context"), dict):
            params["error_context"] = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "suggested_fixes": [
                    "Inspect traceback and fix the indicated line",
                    "Ensure consistent preprocessing between train/test",
                ],
            }
        return action, params

    def _create_error_handling_prompt(self, step_name, error, params):
        # Sanitize params for prompt to avoid excessive length
        safe_params = {k: str(v)[:500] for k, v in params.items()}

        return f'''
        An error occurred in the '{step_name}' step of a machine learning pipeline.

        **Error Details:**
        ```
        {error}
        ```

        **Parameters passed to the step:**
        ```
        {safe_params}
        ```

        Analyze the error and propose a targeted retry plan. Reply ONLY with a JSON object of the form:
        {{
          "action": "retry_with_new_params" | "retry" | "fail",
          "params": {{
            "error_context": {{
              "error_type": "<class like SyntaxError/NameError/TypeError/...>",
              "error_message": "<short message>",
              "suggested_fixes": ["<actionable fix 1>", "<actionable fix 2>"]
            }}
          }}
        }}

        Hints by error type:
        - SyntaxError: balance quotes/parentheses, fix function call syntax, remove stray characters.
        - NameError: define missing variables or correct typos.
        - TypeError: check argument types and function signatures.
        Keep suggestions concise and implementable in one attempt.
        '''
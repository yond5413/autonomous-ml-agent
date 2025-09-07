import os
import re
from e2b_code_interpreter import Sandbox
from .base import Agent
from openai import OpenAI

class DataExplorationAgent(Agent):
    def execute(self, file_path: str, target_column: str, **kwargs):
        # Basic validation
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        # Create sandbox and upload file
        sbx = Sandbox.create(api_key=os.environ.get("E2B_API_KEY"))
        with open(file_path, "rb") as f:
            dataset_path = sbx.files.write(os.path.basename(file_path), f)
        
        # Simple prompt - let the LLM figure out the data structure
        prompt = f"""
I have a dataset saved at {dataset_path.path}. The target column is '{target_column}'.

Generate Python code that:
1. Loads the data and gathers the following information:
    - Basic info (shape, columns, dtypes)
    - Missing values per column
    - Summary statistics for numerical columns
    - Value counts for categorical columns (top 5)
    - Target column distribution analysis
    - Correlations with the target (if numeric)
2. Consolidates all the gathered information into a single JSON object.
3. Prints the final JSON object to standard output.

Make sure to handle any data loading issues gracefully.
"""
        
        # Get code from OpenRouter
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        
        messages = [{"role": "user", "content": prompt}]
        max_retries = 3

        for attempt in range(max_retries):
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=messages,
                max_tokens=1500
            )
            
            generated_text = response.choices[0].message.content
            
            # A more robust regex to find a code block
            code_match = re.search(r'```(?:python)?\s*(.*?)```', generated_text, re.DOTALL)
            
            if code_match:
                code = code_match.group(1).strip()
                # Found code, break the loop and proceed
                break
            else:
                # No code found, add feedback and retry
                messages.append({"role": "assistant", "content": generated_text})
                messages.append({"role": "user", "content": "Your response did not contain a Python code block. Please generate only the Python code, enclosed in a ```python ... ``` block."})
                
                if attempt == max_retries - 1:
                    # Last attempt failed, raise exception
                    raise Exception("No code found in response after multiple attempts.")
        
        print("Generated text from LLM:", generated_text)
        print("Extracted code:", code)

        # Normalize dataset path usage in generated code by replacing any string literal
        # or Path(...) that contains the uploaded file's basename with the remote path.
        remote_path = dataset_path.path
        base_name = os.path.basename(file_path)

        # Replace Path("...<basename>...") -> Path(remote_path)
        code_to_run = re.sub(
            r'Path\((["\'])(?:(?!\1).)*' + re.escape(base_name) + r'(?:(?!\1).)*\1\)',
            f'Path("{remote_path}")',
            code,
            flags=re.DOTALL,
        )
        # Replace any "...<basename>..." string literal -> remote_path
        code_to_run = re.sub(
            r'(["\'])(?:(?!\1).)*' + re.escape(base_name) + r'(?:(?!\1).)*\1',
            f'"{remote_path}"',
            code_to_run,
            flags=re.DOTALL,
        )
        # Also replace the exact local path if present
        code_to_run = code_to_run.replace(file_path, remote_path)

        # Run the code
        try:
            execution = sbx.run_code(code_to_run, timeout=600) # Increased timeout to 10 minutes
            
            # Safely get execution results
            output = None
            
            # Prefer common output attributes without touching .stdout
            for attr in ["text", "output", "result"]:
                val = getattr(execution, attr, None)
                if val:
                    output = val
                    break
            
            # Fall back to logs if available
            if not output:
                logs = getattr(execution, "logs", None)
                if isinstance(logs, str):
                    output = logs.strip()
                elif isinstance(logs, (list, tuple)):
                    try:
                        # Many SDKs provide structured logs; attempt to extract message/content
                        pieces = []
                        for item in logs:
                            if isinstance(item, str):
                                pieces.append(item)
                            else:
                                msg = getattr(item, "message", None) or getattr(item, "content", None) or str(item)
                                pieces.append(str(msg))
                        output = "".join(pieces).strip()
                    except Exception:
                        output = "".join(str(x) for x in logs)
            
            # Capture any error and raise with context
            err = getattr(execution, "error", None)
            if err:
                err_name = getattr(err, "name", type(err).__name__ if err else "ExecutionError")
                err_value = getattr(err, "value", str(err))
                err_traceback = getattr(err, "traceback", "")
                partial = (output[:1000] if isinstance(output, str) else str(output)) if output else ""
                raise Exception(f"{err_name}: {err_value}\n{err_traceback}\n{partial}")
            
            # Return the output or fallback to stringified execution object
            return output if output else str(execution)
            
        except Exception as e:
            raise Exception(f"Code execution failed: {e}")
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
        
        # Run the code
        try:
            execution = sbx.run_code(code, timeout=600) # Increased timeout to 10 minutes
            
            # Safely get execution results
            output = None
            error = None
            
            # Try to get output from various possible attributes
            if hasattr(execution, 'text'):
                output = execution.text
            elif hasattr(execution, 'stdout'):
                output = execution.stdout
            elif hasattr(execution, 'output'):
                output = execution.output
            elif hasattr(execution, 'result'):
                output = execution.result
            
            # Try to get error
            if hasattr(execution, 'error'):
                error = execution.error
            
            print("Execution output:", output)
            if error:
                print("Execution error:", error)
            
            if error:
                raise Exception(f"Code execution failed: {error}")
            
            # Return the output
            return output if output else "No output generated"
            
        except AttributeError as e:
            print(f"AttributeError during execution: {e}")
            print("Execution object type:", type(execution))
            print("Execution object attributes:", [attr for attr in dir(execution) if not attr.startswith('_')])
            
            # Try to get any available output as string representation
            return str(execution) if execution else "Execution completed but no output available"
        except Exception as e:
            print(f"Exception during code execution: {e}")
            raise Exception(f"Code execution failed: {e}")
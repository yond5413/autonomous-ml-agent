from e2b_code_interpreter import sandb
from .base import Agent
import pandas as pd

class PreprocessingAgent(Agent):
    def execute(self, df: pd.DataFrame, target_column: str, **kwargs):
        with CodeInterpreter() as sandbox:
            # Upload the dataframe to the sandbox
            sandbox.upload_file(df.to_csv(index=False).encode(), "/tmp/data.csv")

            # Run the sandbox_explore.py script
            process = sandbox.process.start(
                f"python /workspaces/autonomous-ml-agent/scripts/sandbox_explore.py /tmp/data.csv {target_column}"
            )
            process.wait()

            if process.exit_code != 0:
                print("Error in sandbox:")
                print(process.stderr)
                return None

            # The script prints the JSON to stdout
            return process.stdout

#!/usr/bin/env python3
"""
Test script for the data exploration agent
"""

import os
import sys
import tempfile
import pandas as pd
from src.agents.data_exploration_agent import DataExplorationAgent

def create_test_csv():
    """Create a test CSV file for testing the data exploration agent"""
    # Create sample data
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'target': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
    }
    
    df = pd.DataFrame(data)
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name

def test_data_exploration():
    """Test the data exploration agent"""
    print("Creating test CSV file...")
    test_file = create_test_csv()
    
    try:
        print(f"Test file created at: {test_file}")
        
        # Initialize the agent
        agent = DataExplorationAgent()
        
        print("Running data exploration...")
        result = agent.execute(test_file, target_column='target')
        
        print("Data exploration completed successfully!")
        print("Result:")
        print(result)
        
        # Try to parse as JSON to verify it's valid
        import json
        try:
            parsed_result = json.loads(result)
            print("✓ Result is valid JSON")
            print("✓ Data exploration agent is working correctly")
        except json.JSONDecodeError as e:
            print(f"✗ Result is not valid JSON: {e}")
            print("Raw result:", result)
        
    except Exception as e:
        print(f"✗ Data exploration failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"Cleaned up test file: {test_file}")

if __name__ == "__main__":
    test_data_exploration()
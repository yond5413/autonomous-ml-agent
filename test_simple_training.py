#!/usr/bin/env python3
"""
Test script to verify simple training agent fixes
"""
import os
import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from agents.simple_training_agent import SimpleTrainingAgent
from e2b_code_interpreter import Sandbox

def create_test_data():
    """Create simple test dataset"""
    np.random.seed(42)
    n_samples = 100
    
    # Create features
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randint(0, 3, n_samples)
    })
    
    # Create target (classification)
    y = pd.DataFrame({
        'target': np.random.randint(0, 2, n_samples)
    })
    
    return X, y

def test_simple_training():
    """Test the simple training agent with mock data"""
    print("Creating test data...")
    X, y = create_test_data()
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        X_train_path = os.path.join(temp_dir, 'X_train.csv')
        y_train_path = os.path.join(temp_dir, 'y_train.csv')
        X_test_path = os.path.join(temp_dir, 'X_test.csv')
        y_test_path = os.path.join(temp_dir, 'y_test.csv')
        
        # Split data
        split_idx = 80
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Save to CSV
        X_train.to_csv(X_train_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        y_test.to_csv(y_test_path, index=False)
        
        print(f"Test data saved to {temp_dir}")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        # Create sandbox and upload files
        try:
            sandbox = Sandbox.create()
            print("Sandbox created successfully")
            
            # Upload files to sandbox
            with open(X_train_path, 'rb') as f:
                remote_X_train = sandbox.files.write('X_train.csv', f).path
            with open(y_train_path, 'rb') as f:
                remote_y_train = sandbox.files.write('y_train.csv', f).path
            with open(X_test_path, 'rb') as f:
                remote_X_test = sandbox.files.write('X_test.csv', f).path
            with open(y_test_path, 'rb') as f:
                remote_y_test = sandbox.files.write('y_test.csv', f).path
            
            print("Files uploaded to sandbox")
            
            # Create mock preprocessing output
            preprocessing_output = {
                'sandbox': sandbox,
                'remote_paths': {
                    'X_train': remote_X_train,
                    'y_train': remote_y_train,
                    'X_test': remote_X_test,
                    'y_test': remote_y_test
                },
                'download_urls': {},
                'local_dataset_paths': {}
            }
            
            # Create model selection
            model_selection = {
                'recommended_model': 'RandomForestClassifier',
                'task_type': 'classification'
            }
            
            # Test the agent
            print("Testing SimpleTrainingAgent...")
            agent = SimpleTrainingAgent()
            
            try:
                result = agent.execute(
                    model_selection=model_selection,
                    preprocessing_output=preprocessing_output
                )
                
                print("SUCCESS! Training completed.")
                print(f"Result keys: {list(result.keys())}")
                if 'metrics' in result:
                    print(f"Metrics: {result['metrics']}")
                
                return True
                
            except Exception as e:
                print(f"FAILED: {e}")
                return False
                
        except Exception as e:
            print(f"Sandbox creation failed: {e}")
            return False
        finally:
            try:
                if 'sandbox' in locals():
                    sandbox.kill()
            except:
                pass

if __name__ == "__main__":
    if not os.environ.get("E2B_API_KEY"):
        print("E2B_API_KEY not set, skipping test")
        sys.exit(0)
    
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set, skipping test")
        sys.exit(0)
    
    success = test_simple_training()
    sys.exit(0 if success else 1)

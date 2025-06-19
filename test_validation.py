#!/usr/bin/env python3
"""
Test script for the improved KaLactica validation system.
This script demonstrates the new sampling and validation capabilities.
"""

import pandas as pd
import json
import tempfile
from pathlib import Path
from kalactica.validation import DataValidator

def create_test_data():
    """Create a small test dataset to demonstrate validation."""
    
    # Create test kernel versions data
    kernel_data = {
        'Id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Title': [
            'Python Data Analysis',
            'R Statistical Modeling', 
            'Julia Machine Learning',
            'JavaScript Visualization',
            'SQL Database Query',
            'Python Deep Learning',
            'R Data Visualization',
            'Python NLP Tutorial',
            'Julia Optimization',
            'JavaScript Web App'
        ],
        'ScriptLanguageId': [1, 2, 3, 4, 5, 1, 2, 1, 3, 4],
        'CreationDate': [
            '2023-01-01 10:00:00',
            '2023-01-02 11:00:00',
            '2023-01-03 12:00:00',
            '2023-01-04 13:00:00',
            '2023-01-05 14:00:00',
            '2023-01-06 15:00:00',
            '2023-01-07 16:00:00',
            '2023-01-08 17:00:00',
            '2023-01-09 18:00:00',
            '2023-01-10 19:00:00'
        ],
        'VersionNumber': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    
    # Create test language mapping
    language_data = {
        'Id': [1, 2, 3, 4, 5],
        'Name': ['Python', 'R', 'Julia', 'JavaScript', 'SQL']
    }
    
    # Create test notebooks
    notebooks = {}
    for i in range(1, 11):
        notebook = {
            'cells': [
                {
                    'cell_type': 'code',
                    'source': f'import pandas as pd\nimport numpy as np\n\n# Test notebook {i}\ndf = pd.DataFrame({{"col1": [1, 2, 3], "col2": ["a", "b", "c"]}})\nprint(df.head())',
                    'metadata': {},
                    'execution_count': 1,
                    'outputs': []
                },
                {
                    'cell_type': 'markdown',
                    'source': f'# Test Notebook {i}\n\nThis is a test notebook for validation.',
                    'metadata': {}
                }
            ],
            'metadata': {
                'kernelspec': {
                    'display_name': 'Python 3',
                    'language': 'python',
                    'name': 'python3'
                }
            },
            'nbformat': 4,
            'nbformat_minor': 4
        }
        notebooks[i] = notebook
    
    return kernel_data, language_data, notebooks

def test_validation():
    """Test the validation system with sample data."""
    
    print("Creating test data...")
    kernel_data, language_data, notebooks = create_test_data()
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save test files
        kernel_df = pd.DataFrame(kernel_data)
        kernel_df.to_csv(temp_path / 'KernelVersions.csv', index=False)
        
        lang_df = pd.DataFrame(language_data)
        lang_df.to_csv(temp_path / 'KernelLanguages.csv', index=False)
        
        # Create notebooks directory and save notebooks
        notebooks_dir = temp_path / 'notebooks'
        notebooks_dir.mkdir()
        
        for notebook_id, notebook in notebooks.items():
            with open(notebooks_dir / f'{notebook_id}.json', 'w') as f:
                json.dump(notebook, f, indent=2)
        
        print(f"Test data created in {temp_path}")
        print(f"Kernel versions: {len(kernel_df)} records")
        print(f"Languages: {len(lang_df)} types")
        print(f"Notebooks: {len(notebooks)} files")
        
        # Test validation with different sample sizes
        test_cases = [
            ("No sampling (all data)", None),
            ("Sample size 5", 5),
            ("Sample size 8", 8),
        ]
        
        for test_name, sample_size in test_cases:
            print(f"\n{'='*50}")
            print(f"Testing: {test_name}")
            print(f"{'='*50}")
            
            validator = DataValidator(sample_size=sample_size, random_seed=42)
            results = validator.validate_dataset(
                kernel_versions_path=temp_path / 'KernelVersions.csv',
                notebooks_dir=notebooks_dir,
                languages_path=temp_path / 'KernelLanguages.csv'
            )
            
            print(f"Total notebooks in dataset: {results['total_notebooks']}")
            print(f"Notebooks sampled for validation: {results['sampled_notebooks']}")
            print(f"Valid notebooks: {results['valid_notebooks']}")
            print(f"Invalid notebooks: {results['invalid_notebooks']}")
            
            if results['language_distribution']:
                print(f"\nLanguage Distribution:")
                for lang, count in results['language_distribution'].items():
                    print(f"  - {lang}: {count}")
            
            print(f"\nValidation Errors:")
            for error_type, count in results['validation_errors'].items():
                if count > 0:
                    print(f"  - {error_type}: {count}")
            
            print(f"\nSample Info:")
            for key, value in results['sample_info'].items():
                print(f"  - {key}: {value}")

def test_error_handling():
    """Test validation with problematic data."""
    
    print(f"\n{'='*50}")
    print("Testing Error Handling")
    print(f"{'='*50}")
    
    # Create data with some issues
    problematic_data = {
        'Id': [1, 2, 3, 4, 5],
        'Title': ['Valid', 'Valid', 'Valid', 'Valid', 'Valid'],
        'ScriptLanguageId': [1, 2, 999, 4, 5],  # Invalid language ID
        'CreationDate': [
            '2023-01-01 10:00:00',
            '2023-01-02 11:00:00',
            'invalid-date',  # Invalid date
            '2023-01-04 13:00:00',
            '2023-01-05 14:00:00'
        ],
        'VersionNumber': [1, 1, 1, 1, 1]
    }
    
    language_data = {
        'Id': [1, 2, 4, 5],
        'Name': ['Python', 'R', 'JavaScript', 'SQL']
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save problematic files
        kernel_df = pd.DataFrame(problematic_data)
        kernel_df.to_csv(temp_path / 'KernelVersions.csv', index=False)
        
        lang_df = pd.DataFrame(language_data)
        lang_df.to_csv(temp_path / 'KernelLanguages.csv', index=False)
        
        # Create notebooks directory
        notebooks_dir = temp_path / 'notebooks'
        notebooks_dir.mkdir()
        
        # Create some valid notebooks
        for i in range(1, 6):
            notebook = {
                'cells': [
                    {
                        'cell_type': 'code',
                        'source': f'print("Hello from notebook {i}")',
                        'metadata': {},
                        'execution_count': 1,
                        'outputs': []
                    }
                ],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
            
            with open(notebooks_dir / f'{i}.json', 'w') as f:
                json.dump(notebook, f, indent=2)
        
        # Test validation
        validator = DataValidator(sample_size=None, random_seed=42)
        results = validator.validate_dataset(
            kernel_versions_path=temp_path / 'KernelVersions.csv',
            notebooks_dir=notebooks_dir,
            languages_path=temp_path / 'KernelLanguages.csv'
        )
        
        print(f"Total notebooks: {results['total_notebooks']}")
        print(f"Valid notebooks: {results['valid_notebooks']}")
        print(f"Invalid notebooks: {results['invalid_notebooks']}")
        
        print(f"\nValidation Errors:")
        for error_type, count in results['validation_errors'].items():
            if count > 0:
                print(f"  - {error_type}: {count}")

if __name__ == "__main__":
    print("KaLactica Validation System Test")
    print("=" * 50)
    
    try:
        test_validation()
        test_error_handling()
        print(f"\n{'='*50}")
        print("All tests completed successfully!")
        print(f"{'='*50}")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 
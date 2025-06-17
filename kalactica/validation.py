"""Data validation utilities for KaLactica."""

import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates Kaggle notebook data before preprocessing."""
    
    def __init__(self):
        self.valid_languages = {'python', 'r', 'julia'}
        self.min_content_length = 10
        self.max_content_length = 10000
        self.validation_results = {
            'total_notebooks': 0,
            'valid_notebooks': 0,
            'invalid_notebooks': 0,
            'validation_errors': {
                'missing_metadata': 0,
                'invalid_code': 0,
                'invalid_markdown': 0,
                'invalid_language': 0,
                'duplicate_ids': 0,
                'invalid_timestamps': 0
            }
        }

    def validate_kernel_versions(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate KernelVersions.csv structure and content."""
        errors = []
        
        # Check required columns
        required_cols = {
            'Id': int,
            'Title': str,
            'CurrentKernelVersionId': int,
            'Language': str,
            'CreationDate': str,  # For timestamp validation
            'VersionNumber': int   # For version validation
        }
        
        for col, dtype in required_cols.items():
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
            elif not df[col].dtype == dtype:
                errors.append(f"Invalid data type for column {col}: expected {dtype}, got {df[col].dtype}")
        
        # Validate language codes
        invalid_languages = df[~df['Language'].str.lower().isin(self.valid_languages)]
        if not invalid_languages.empty:
            errors.append(f"Invalid language codes found: {invalid_languages['Language'].unique()}")
            self.validation_results['validation_errors']['invalid_language'] += len(invalid_languages)
        
        # Check for duplicate IDs
        duplicates = df[df.duplicated(['Id'], keep=False)]
        if not duplicates.empty:
            errors.append(f"Duplicate notebook IDs found: {duplicates['Id'].unique()}")
            self.validation_results['validation_errors']['duplicate_ids'] += len(duplicates)
        
        # Validate timestamps
        try:
            df['CreationDate'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        except ValueError as e:
            errors.append(f"Invalid timestamp format: {str(e)}")
            self.validation_results['validation_errors']['invalid_timestamps'] += 1
        
        return len(errors) == 0, errors

    def validate_notebook_structure(self, notebook: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate notebook JSON structure."""
        errors = []
        
        if not isinstance(notebook, dict):
            errors.append("Notebook must be a dictionary")
            return False, errors
        
        required_fields = {'cells', 'metadata', 'nbformat', 'nbformat_minor'}
        missing_fields = required_fields - set(notebook.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            self.validation_results['validation_errors']['missing_metadata'] += 1
        
        return len(errors) == 0, errors

    def validate_cell(self, cell: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single notebook cell."""
        errors = []
        
        if 'cell_type' not in cell:
            errors.append("Missing cell_type")
            return False, errors
        
        if cell['cell_type'] not in {'code', 'markdown'}:
            errors.append(f"Invalid cell_type: {cell['cell_type']}")
            return False, errors
        
        if 'source' not in cell:
            errors.append("Missing source content")
            return False, errors
        
        content = cell['source']
        if not content or len(content) < self.min_content_length:
            errors.append(f"Content too short: {len(content)} chars")
            return False, errors
        
        if len(content) > self.max_content_length:
            errors.append(f"Content too long: {len(content)} chars")
            return False, errors
        
        # Validate based on cell type
        if cell['cell_type'] == 'code':
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"Invalid Python syntax: {str(e)}")
                self.validation_results['validation_errors']['invalid_code'] += 1
        elif cell['cell_type'] == 'markdown':
            # Basic markdown validation
            if not any(char in content for char in '#*`'):
                errors.append("Markdown cell appears to have no formatting")
                self.validation_results['validation_errors']['invalid_markdown'] += 1
        
        return len(errors) == 0, errors

    def validate_notebook(self, notebook_path: Path) -> Tuple[bool, List[str]]:
        """Validate a single notebook file."""
        errors = []
        
        try:
            with open(notebook_path) as f:
                notebook = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
            return False, errors
        except Exception as e:
            errors.append(f"Error reading notebook: {str(e)}")
            return False, errors
        
        # Validate notebook structure
        is_valid, structure_errors = self.validate_notebook_structure(notebook)
        if not is_valid:
            errors.extend(structure_errors)
            return False, errors
        
        # Validate each cell
        for cell in notebook.get('cells', []):
            is_valid, cell_errors = self.validate_cell(cell)
            if not is_valid:
                errors.extend(cell_errors)
        
        return len(errors) == 0, errors

    def validate_dataset(self, kernel_versions_path: Path, notebooks_dir: Path) -> Dict[str, Any]:
        """Validate entire dataset."""
        try:
            df = pd.read_csv(kernel_versions_path)
        except Exception as e:
            logger.error(f"Error reading KernelVersions.csv: {str(e)}")
            return self.validation_results
        
        # Validate kernel versions
        is_valid, errors = self.validate_kernel_versions(df)
        if not is_valid:
            logger.error("KernelVersions.csv validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
        
        # Validate each notebook
        self.validation_results['total_notebooks'] = len(df)
        for _, row in df.iterrows():
            notebook_path = notebooks_dir / f"{row['Id']}.json"
            if not notebook_path.exists():
                logger.warning(f"Notebook not found: {notebook_path}")
                self.validation_results['invalid_notebooks'] += 1
                continue
            
            is_valid, errors = self.validate_notebook(notebook_path)
            if is_valid:
                self.validation_results['valid_notebooks'] += 1
            else:
                self.validation_results['invalid_notebooks'] += 1
                logger.warning(f"Notebook validation failed for {notebook_path}:")
                for error in errors:
                    logger.warning(f"  - {error}")
        
        return self.validation_results

def main():
    """Command-line interface for data validation."""
    import argparse
    parser = argparse.ArgumentParser(description="Validate Kaggle notebook data")
    parser.add_argument("--kernel-versions", type=str, required=True,
                      help="Path to KernelVersions.csv")
    parser.add_argument("--notebooks-dir", type=str, required=True,
                      help="Directory containing notebook JSON files")
    parser.add_argument("--output", type=str,
                      help="Output JSON file for validation results")
    args = parser.parse_args()
    
    validator = DataValidator()
    results = validator.validate_dataset(
        Path(args.kernel_versions),
        Path(args.notebooks_dir)
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print("\nValidation Summary:")
    print(f"Total notebooks: {results['total_notebooks']}")
    print(f"Valid notebooks: {results['valid_notebooks']}")
    print(f"Invalid notebooks: {results['invalid_notebooks']}")
    print("\nValidation Errors:")
    for error_type, count in results['validation_errors'].items():
        print(f"  - {error_type}: {count}")

if __name__ == "__main__":
    main() 
"""Data validation utilities for KaLactica."""

import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from datetime import datetime
import logging
import random
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates Kaggle notebook data before preprocessing."""
    
    def __init__(self, sample_size: Optional[int] = None, random_seed: int = 42):
        # Expanded valid languages to include common variations
        self.valid_languages = {
            'python', 'r', 'julia', 'javascript', 'sql', 'scala', 'java', 'c++', 'c#',
            'go', 'rust', 'swift', 'kotlin', 'php', 'ruby', 'perl', 'matlab', 'octave',
            'sas', 'stata', 'spss', 'unknown'  # Include 'unknown' as valid
        }
        self.min_content_length = 10
        self.max_content_length = 10000
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.validation_results = {
            'total_notebooks': 0,
            'sampled_notebooks': 0,
            'valid_notebooks': 0,
            'invalid_notebooks': 0,
            'validation_errors': {
                'missing_metadata': 0,
                'invalid_code': 0,
                'invalid_markdown': 0,
                'invalid_language': 0,
                'duplicate_ids': 0,
                'invalid_timestamps': 0,
                'missing_notebooks': 0
            },
            'language_distribution': {},
            'sample_info': {
                'sample_size': sample_size,
                'random_seed': random_seed,
                'sampling_method': 'stratified' if sample_size else 'none'
            }
        }

    def sample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample the dataframe for validation."""
        if self.sample_size is None or self.sample_size >= len(df):
            return df
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Stratified sampling by language if possible
        # 
        if 'Language' in df.columns and len(df['Language'].unique()) > 1:
            try:
                # Stratified sampling with minimum samples per language
                min_samples_per_language = max(1, self.sample_size // len(df['Language'].unique()))
                sampled_df = pd.DataFrame()
                
                for language in df['Language'].unique():
                    lang_df = df[df['Language'] == language]
                    if len(lang_df) <= min_samples_per_language:
                        sampled_df = pd.concat([sampled_df, lang_df])
                    else:
                        sampled = lang_df.sample(n=min_samples_per_language, random_state=self.random_seed)
                        sampled_df = pd.concat([sampled_df, sampled])
                
                # If we still need more samples, add random samples
                if len(sampled_df) < self.sample_size:
                    remaining = self.sample_size - len(sampled_df)
                    remaining_df = df[~df.index.isin(sampled_df.index)]
                    if len(remaining_df) > 0:
                        additional_samples = remaining_df.sample(n=min(remaining, len(remaining_df)), random_state=self.random_seed)
                        sampled_df = pd.concat([sampled_df, additional_samples])
                
                logger.info(f"Stratified sampling: {len(sampled_df)} samples from {len(df)} total")
                return sampled_df
                
            except Exception as e:
                logger.warning(f"Stratified sampling failed, falling back to random: {e}")
        
        # Fallback to simple random sampling
        sampled_df = df.sample(n=min(self.sample_size, len(df)), random_state=self.random_seed)
        logger.info(f"Random sampling: {len(sampled_df)} samples from {len(df)} total")
        return sampled_df

    def validate_kernel_versions(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate KernelVersions.csv structure and content."""
        errors = []
        
        # Check required columns (relaxed requirements)
        required_cols = {
            'Id': int,
            'Title': str,
        }
        
        optional_cols = {
            'CurrentKernelVersionId': int,
            'Language': str,
            'CreationDate': str,
            'VersionNumber': int,
            'ScriptLanguageId': int
        }
        
        # Check required columns
        for col, dtype in required_cols.items():
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
            elif not df[col].dtype == dtype:
                errors.append(f"Invalid data type for column {col}: expected {dtype}, got {df[col].dtype}")
        
        # Check optional columns
        for col, dtype in optional_cols.items():
            if col in df.columns and not df[col].dtype == dtype:
                errors.append(f"Invalid data type for column {col}: expected {dtype}, got {df[col].dtype}")
        
        # Validate language codes if Language column exists
        if 'Language' in df.columns:
            # Count language distribution
            lang_counts = df['Language'].value_counts()
            self.validation_results['language_distribution'] = lang_counts.to_dict()
            
            # Check for invalid languages (case-insensitive)
            df_lower = df['Language'].str.lower()
            invalid_languages = df[~df_lower.isin([lang.lower() for lang in self.valid_languages])]
            if not invalid_languages.empty:
                unique_invalid = invalid_languages['Language'].unique()
                errors.append(f"Invalid language codes found: {unique_invalid[:10]}...")  # Show first 10
                self.validation_results['validation_errors']['invalid_language'] += len(invalid_languages)
                logger.warning(f"Found {len(unique_invalid)} unique invalid languages")
        
        # Check for duplicate IDs
        duplicates = df[df.duplicated(['Id'], keep=False)]
        if not duplicates.empty:
            errors.append(f"Duplicate notebook IDs found: {len(duplicates)} rows")
            self.validation_results['validation_errors']['duplicate_ids'] += len(duplicates)
        
        # Validate timestamps if CreationDate exists
        if 'CreationDate' in df.columns:
            try:
                # Try multiple timestamp formats
                timestamp_errors = 0
                for date_str in df['CreationDate'].dropna():
                    try:
                        datetime.strptime(str(date_str), '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        try:
                            datetime.strptime(str(date_str), '%Y-%m-%d')
                        except ValueError:
                            timestamp_errors += 1
                
                if timestamp_errors > 0:
                    errors.append(f"Invalid timestamp format: {timestamp_errors} errors")
                    self.validation_results['validation_errors']['invalid_timestamps'] += timestamp_errors
            except Exception as e:
                errors.append(f"Error validating timestamps: {str(e)}")
        
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

    def validate_dataset(self, kernel_versions_path: Path, notebooks_dir: Path, languages_path: Optional[Path] = None) -> Dict[str, Any]:
        """Validate entire dataset with optional sampling."""
        try:
            df = pd.read_csv(kernel_versions_path)
            logger.info(f"Loaded {len(df)} notebooks from {kernel_versions_path}")
            
            # Always map ScriptLanguageId to language name before any validation
            if languages_path and languages_path.exists():
                try:
                    lang_df = pd.read_csv(languages_path)
                    lang_map = dict(zip(lang_df['Id'], lang_df['Name']))
                    if 'ScriptLanguageId' in df.columns:
                        df['Language'] = df['ScriptLanguageId'].map(lang_map).fillna('unknown')
                        logger.info(f"Mapped languages using {languages_path}")
                except Exception as e:
                    logger.warning(f"Error mapping languages: {e}")
                    df['Language'] = 'unknown'
            elif 'Language' not in df.columns:
                df['Language'] = 'unknown'
                logger.info("No language mapping available, using 'unknown'")
            
            # Parse CreationDate using pd.to_datetime with errors='coerce'
            if 'CreationDate' in df.columns:
                df['CreationDate'] = pd.to_datetime(df['CreationDate'], errors='coerce')
            
            # MC sampling on notebook length (TotalLines or fallback)
            length_col = None
            for col in ['TotalLines', 'LinesOfCode', 'LinesInsertedFromPrevious']:
                if col in df.columns:
                    length_col = col
                    break
            if self.sample_size is not None and self.sample_size < len(df):
                if length_col:
                    # MC sampling weighted by notebook length
                    weights = df[length_col].fillna(0).astype(float) + 1e-3  # avoid zero weights
                    weights = weights / weights.sum()
                    sampled_df = df.sample(n=self.sample_size, weights=weights, random_state=self.random_seed)
                    logger.info(f"MC sampling on {length_col}: {len(sampled_df)} samples from {len(df)} total")
                else:
                    sampled_df = df.sample(n=self.sample_size, random_state=self.random_seed)
                    logger.info(f"Uniform random sampling: {len(sampled_df)} samples from {len(df)} total")
                df = sampled_df
            self.validation_results['total_notebooks'] = len(df) if self.sample_size is None else len(df)
            self.validation_results['sampled_notebooks'] = len(df)
            
            # Validate kernel versions
            is_valid, errors = self.validate_kernel_versions(df)
            if not is_valid:
                logger.error("KernelVersions.csv validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
            
            # Validate each notebook in the sample
            logger.info(f"Validating {len(df)} notebooks...")
            for idx, row in df.iterrows():
                if idx % 1000 == 0:
                    logger.info(f"Processed {idx}/{len(df)} notebooks...")
                
                notebook_path = notebooks_dir / f"{row['Id']}.json"
                if not notebook_path.exists():
                    self.validation_results['validation_errors']['missing_notebooks'] += 1
                    continue
                
                is_valid, errors = self.validate_notebook(notebook_path)
                if is_valid:
                    self.validation_results['valid_notebooks'] += 1
                else:
                    self.validation_results['invalid_notebooks'] += 1
                    # Only log first few errors to avoid spam
                    if self.validation_results['invalid_notebooks'] <= 5:
                        logger.warning(f"Notebook validation failed for {notebook_path}:")
                        for error in errors[:3]:  # Limit to first 3 errors
                            logger.warning(f"  - {error}")
            
            logger.info("Validation completed")
            
            # Missingness/NaN checks
            missingness_summary = df.isnull().sum().to_dict()
            self.validation_results['missingness'] = missingness_summary
            logger.info("Missingness Summary:")
            for column, count in missingness_summary.items():
                if count > 0:
                    logger.warning(f"Warning: {column} has {count} missing values")
            
            # Warn for float64 columns that should be int
            float_to_int_columns = [col for col in df.columns if df[col].dtype == 'float64' and col.lower().endswith('id')]
            for column in float_to_int_columns:
                if df[column].isnull().sum() > 0:
                    logger.warning(f"Warning: {column} is float64 but has {df[column].isnull().sum()} NaN values. Consider filling or dropping.")
            
            # Derive summary stats from sample
            describe_stats = df.describe(include='all').to_dict()
            self.validation_results['sample_summary'] = describe_stats
            self.validation_results['language_distribution'] = df['Language'].value_counts().to_dict()
            
            # Data Constitution check
            constitution_path = Path(kernel_versions_path).parent / 'DATA_CONSTITUTION.txt'
            if not constitution_path.exists():
                logger.warning(f"No Data Constitution file found at {constitution_path}. Consider creating one to document data rules and schema.")
                self.validation_results['data_constitution'] = 'NOT FOUND'
            else:
                with open(constitution_path) as f:
                    self.validation_results['data_constitution'] = f.read(2048)  # preview first 2KB
            
        except Exception as e:
            logger.error(f"Error reading KernelVersions.csv: {str(e)}")
            return self.validation_results
        
        return self.validation_results

def parse_constitution(path):
    """Parse the DATA_CONSTITUTION.txt into a dict of dataset schemas."""
    with open(path) as f:
        text = f.read()
    datasets = {}
    current = None
    import re
    for line in text.splitlines():
        if line.startswith("## ") and ".csv" in line:
            current = line[3:].strip().replace("`", "")
            datasets[current] = {"columns": {}, "foreign_keys": [], "notes": []}
        elif current and re.match(r"^- [A-Za-z0-9_]+:", line):
            col, rest = line[2:].split(":", 1)
            dtype = rest.split("(")[0].strip()
            datasets[current]["columns"][col.strip()] = dtype
        elif current and "Foreign Keys:" in line:
            datasets[current]["foreign_keys"] = []
        elif current and "→" in line:
            datasets[current]["foreign_keys"].append(line.strip("- ").strip())
        elif current and line.strip().startswith("**Special Notes:**"):
            datasets[current]["notes"].append(line.strip())
    return datasets

def sample_df(df, sample_size=1000, length_col=None):
    if length_col and length_col in df.columns:
        weights = df[length_col].fillna(0).astype(float) + 1e-3
        weights = weights / weights.sum()
        return df.sample(n=min(sample_size, len(df)), weights=weights, random_state=42)
    return df.sample(n=min(sample_size, len(df)), random_state=42)

def check_types(df, schema):
    mismatches = []
    for col, dtype in schema.items():
        if col not in df.columns:
            mismatches.append(f"Missing column: {col}")
        else:
            # Simple dtype check
            if dtype == "int64" and not pd.api.types.is_integer_dtype(df[col]):
                mismatches.append(f"{col} expected int64, got {df[col].dtype}")
            elif dtype == "float64" and not pd.api.types.is_float_dtype(df[col]):
                mismatches.append(f"{col} expected float64, got {df[col].dtype}")
            elif dtype == "string" and not pd.api.types.is_string_dtype(df[col]):
                mismatches.append(f"{col} expected string, got {df[col].dtype}")
            elif dtype == "datetime" and not np.issubdtype(df[col].dtype, np.datetime64):
                mismatches.append(f"{col} expected datetime, got {df[col].dtype}")
    return mismatches

def check_missingness(df, schema):
    missing = {}
    for col, dtype in schema.items():
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                missing[col] = int(n_missing)
    return missing

def validate_meta_kaggle_dataset(csv_path, schema, sample_size=1000):
    print(f"\nValidating {csv_path} ...")
    df = pd.read_csv(csv_path)
    # Date parsing
    for col, dtype in schema.items():
        if dtype == "datetime" and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # MC sampling
    length_col = next((c for c in ["TotalLines", "LinesOfCode"] if c in df.columns), None)
    df_sample = sample_df(df, sample_size=sample_size, length_col=length_col)
    # Type checks
    type_issues = check_types(df_sample, schema)
    # Missingness
    missingness = check_missingness(df_sample, schema)
    # Summary
    print(f"Sampled {len(df_sample)} rows.")
    if type_issues:
        print("Type issues:", type_issues)
    if missingness:
        print("Missingness:", missingness)
    print("Describe:\n", df_sample.describe(include='all').T)
    return {"type_issues": type_issues, "missingness": missingness}

def main():
    """Command-line interface for data validation."""
    import argparse
    parser = argparse.ArgumentParser(description="Validate Kaggle notebook data or meta-kaggle datasets")
    parser.add_argument("--kernel-versions", type=str,
                      help="Path to KernelVersions.csv (for notebook validation)")
    parser.add_argument("--notebooks-dir", type=str,
                      help="Directory containing notebook JSON files (for notebook validation)")
    parser.add_argument("--output", type=str,
                      help="Output JSON file for validation results")
    parser.add_argument("--languages", type=str, required=False,
                      default="/kaggle/input/meta-kaggle/KernelLanguages.csv",
                      help="Path to KernelLanguages.csv")
    parser.add_argument("--sample-size", type=int, default=1000,
                      help="Number of notebooks to sample for validation (default: 1000)")
    parser.add_argument("--random-seed", type=int, default=42,
                      help="Random seed for sampling (default: 42)")
    parser.add_argument("--no-sampling", action="store_true",
                      help="Disable sampling and validate all notebooks (not recommended for large datasets)")
    parser.add_argument("--meta-kaggle", action="store_true",
                      help="If set, validate all meta-kaggle datasets listed in DATA_CONSTITUTION.txt in the data directory.")
    parser.add_argument("--data-dir", type=str, default="data",
                      help="Directory containing meta-kaggle CSVs (for --meta-kaggle mode)")
    args = parser.parse_args()

    if args.meta_kaggle:
        constitution = parse_constitution("DATA_CONSTITUTION.txt")
        for dataset, info in constitution.items():
            csv_path = Path(args.data_dir) / dataset
            if csv_path.exists():
                validate_meta_kaggle_dataset(csv_path, info["columns"], sample_size=args.sample_size)
            else:
                print(f"WARNING: {csv_path} not found.")
        return

    # Default: notebook validation
    if not args.kernel_versions or not args.notebooks_dir:
        print("Error: --kernel-versions and --notebooks-dir are required unless --meta-kaggle is set.")
        return
    sample_size = None if args.no_sampling else args.sample_size
    validator = DataValidator(sample_size=sample_size, random_seed=args.random_seed)
    results = validator.validate_dataset(
        Path(args.kernel_versions),
        Path(args.notebooks_dir),
        Path(args.languages) if args.languages else None
    )
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Validation results saved to {args.output}")
    # Print summary
    print("\nValidation Summary:")
    print(f"Total notebooks in dataset: {results['total_notebooks']:,}")
    print(f"Notebooks sampled for validation: {results['sampled_notebooks']:,}")
    print(f"Valid notebooks: {results['valid_notebooks']:,}")
    print(f"Invalid notebooks: {results['invalid_notebooks']:,}")
    if results['language_distribution']:
        print(f"\nLanguage Distribution (top 10):")
        for lang, count in list(results['language_distribution'].items())[:10]:
            print(f"  - {lang}: {count:,}")
    print(f"\nValidation Errors:")
    for error_type, count in results['validation_errors'].items():
        if count > 0:
            print(f"  - {error_type}: {count:,}")
    print(f"\nSample Info:")
    for key, value in results['sample_info'].items():
        print(f"  - {key}: {value}")

if __name__ == "__main__":
    main() 
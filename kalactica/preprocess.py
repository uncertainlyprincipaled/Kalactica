"""Data preprocessing utilities for KaLactica."""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import logging

from .config import DATA_DIR
from .validation import DataValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_kernel_versions(csv_path: str) -> pd.DataFrame:
    """Load and validate the KernelVersions.csv file."""
    df = pd.read_csv(csv_path)
    required_cols = ['Id', 'Title', 'CurrentKernelVersionId', 'Language']
    assert all(col in df.columns for col in required_cols), \
        f"Missing required columns: {required_cols}"
    return df

def wrap_content(content: str, content_type: str) -> str:
    """Wrap content in appropriate tags."""
    if content_type == "code":
        return f"<code>{content}</code>"
    elif content_type == "markdown":
        return f"<markdown>{content}</markdown>"
    elif content_type == "dataset":
        return f"<dataset>{content}</dataset>"
    return content

def process_notebook(notebook: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a single notebook into a list of cells."""
    cells = []
    
    # Process code cells
    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'code':
            cells.append({
                'type': 'code',
                'content': wrap_content(cell['source'], 'code'),
                'metadata': cell.get('metadata', {})
            })
        elif cell['cell_type'] == 'markdown':
            cells.append({
                'type': 'markdown',
                'content': wrap_content(cell['source'], 'markdown'),
                'metadata': cell.get('metadata', {})
            })
    
    return cells

def main():
    parser = argparse.ArgumentParser(description="Preprocess Kaggle notebook data")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to KernelVersions.csv")
    parser.add_argument("--sample", type=int, default=None,
                      help="Number of notebooks to sample")
    parser.add_argument("--output", type=str,
                      default=str(DATA_DIR / "processed.jsonl"),
                      help="Output JSONL file path")
    parser.add_argument("--skip-validation", action="store_true",
                      help="Skip data validation step")
    args = parser.parse_args()

    # Validate data first
    if not args.skip_validation:
        logger.info("Validating data...")
        validator = DataValidator()
        validation_results = validator.validate_dataset(
            Path(args.input),
            Path(args.input).parent
        )
        
        if validation_results['invalid_notebooks'] > 0:
            logger.warning(f"Found {validation_results['invalid_notebooks']} invalid notebooks")
            if input("Continue with preprocessing? (y/n): ").lower() != 'y':
                return
    
    # Load and sample data
    df = pd.read_csv(args.input)
    if args.sample:
        df = df.sample(n=args.sample, random_state=42)
    
    # Process notebooks
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    with open(output_path, 'w') as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Load notebook JSON
                notebook_path = DATA_DIR / f"{row['Id']}.json"
                if not notebook_path.exists():
                    logger.warning(f"Notebook not found: {notebook_path}")
                    error_count += 1
                    continue
                
                with open(notebook_path) as nf:
                    notebook = json.load(nf)
                
                # Process and write cells
                cells = process_notebook(notebook)
                for cell in cells:
                    cell['notebook_id'] = row['Id']
                    cell['title'] = row['Title']
                    cell['language'] = row['Language']
                    f.write(json.dumps(cell) + '\n')
                
                processed_count += 1
            
            except Exception as e:
                logger.error(f"Error processing notebook {row['Id']}: {e}")
                error_count += 1
                continue
    
    logger.info(f"Processing complete:")
    logger.info(f"  - Processed notebooks: {processed_count}")
    logger.info(f"  - Errors: {error_count}")

if __name__ == "__main__":
    main() 
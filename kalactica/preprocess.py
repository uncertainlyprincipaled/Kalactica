"""Data preprocessing utilities for KaLactica."""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from .config import DATA_DIR

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
    args = parser.parse_args()

    # Load and sample data
    df = load_kernel_versions(args.input)
    if args.sample:
        df = df.sample(n=args.sample, random_state=42)
    
    # Process notebooks
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Load notebook JSON
                notebook_path = DATA_DIR / f"{row['Id']}.json"
                if not notebook_path.exists():
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
            
            except Exception as e:
                print(f"Error processing notebook {row['Id']}: {e}")
                continue

if __name__ == "__main__":
    main() 
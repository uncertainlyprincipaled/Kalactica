"""End-to-end demonstration of KaLactica."""

import argparse
from pathlib import Path
import json
import os

from kalactica.preprocess import load_kernel_versions, process_notebook
from kalactica.retrieval import Retriever
from kalactica.model import Generator
from kalactica.safety import TopologyFilter
from kalactica.config import AWS_CONFIG

def main():
    parser = argparse.ArgumentParser(description="KaLactica Demo")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--sample", type=int, help="Number of notebooks to sample")
    parser.add_argument("--output", default="outputs/processed.jsonl", help="Output JSONL file")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocess notebooks
    print("Loading kernel versions...")
    kernel_versions = load_kernel_versions(args.input)
    
    if args.sample:
        print(f"Sampling {args.sample} notebooks...")
        kernel_versions = kernel_versions.sample(n=args.sample)
    
    print("Processing notebooks...")
    with open(args.output, "w") as f:
        for _, row in kernel_versions.iterrows():
            try:
                cells = process_notebook(row["notebook_json"])
                for cell in cells:
                    f.write(json.dumps(cell) + "\n")
            except Exception as e:
                print(f"Error processing notebook {row['id']}: {e}")
    
    # Build index
    print("Building search index...")
    retriever = Retriever(
        opensearch_host=AWS_CONFIG["opensearch_host"],
        s3_bucket=AWS_CONFIG["s3_bucket"],
        region=AWS_CONFIG["region"]
    )
    retriever.build_index(args.output)
    
    # Generate notebook
    print("Generating Titanic notebook...")
    generator = Generator(
        model_path=os.getenv("MODEL_PATH", "facebook/galactica-1.3b"),
        retriever_path=None  # Using AWS retriever
    )
    safety_filter = TopologyFilter()
    
    notebook = generator.generate_notebook(
        "Create a data analysis notebook for the Titanic dataset, "
        "including data loading, exploration, feature engineering, "
        "and a simple model to predict survival."
    )
    
    is_safe, stats = safety_filter.is_safe(notebook)
    if is_safe:
        output_path = output_dir / "titanic_demo.ipynb"
        with open(output_path, "w") as f:
            json.dump(notebook, f, indent=2)
        print(f"\nGenerated notebook saved to {output_path}")
        
        print("\nCitations:")
        for chunk in generator.last_citations:
            print(f"- {chunk['title']}")
    else:
        print("Generated notebook did not pass safety checks.")
        print("\nSafety stats:", json.dumps(stats, indent=2))

if __name__ == "__main__":
    main() 
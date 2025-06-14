"""End-to-end demonstration of KaLactica."""

import argparse
from pathlib import Path
from kalactica.preprocess import load_kernel_versions, process_notebook
from kalactica.retrieval import build_index, search
from kalactica.model import Generator
from kalactica.safety import TopologyFilter
import json

def main():
    parser = argparse.ArgumentParser(description="KaLactica Demo")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to KernelVersions.csv")
    parser.add_argument("--sample", type=int, default=1000,
                      help="Number of notebooks to sample")
    parser.add_argument("--output", type=str, default="data/processed.jsonl",
                      help="Output JSONL file path")
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Preprocessing notebooks...")
    # Load and sample notebooks
    df = load_kernel_versions(args.input)
    if args.sample:
        df = df.sample(n=args.sample, random_state=42)
    
    # Process notebooks
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            try:
                # Load notebook JSON
                notebook_path = Path("data") / f"{row['Id']}.json"
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
    
    print("Step 2: Building FAISS index...")
    # Build index
    build_index(str(output_path), "indices")
    
    print("Step 3: Generating Titanic notebook...")
    # Initialize generator
    generator = Generator("checkpoints/model.pt", "indices")
    
    # Load safety filter
    safety_filter = TopologyFilter()
    
    # Generate notebook
    prompt = "Generate a Jupyter notebook for the Titanic dataset"
    response, chunks = generator(prompt, max_tokens=400)
    
    # Check safety
    is_safe, stats = safety_filter.is_safe(response, "tabular")
    
    if is_safe:
        # Save notebook
        output_path = Path("outputs/titanic_demo.ipynb")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(response)
        
        print(f"\nNotebook saved to {output_path}")
        print("\nCitations:")
        for chunk in chunks:
            print(f"- {chunk['title']}")
    else:
        print("Error: Generated notebook failed safety check")
        print("\nDebug info:", json.dumps(stats, indent=2))

if __name__ == "__main__":
    main() 
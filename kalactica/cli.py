"""Command-line interface for KaLactica."""

import argparse
import json
from pathlib import Path
from typing import Optional

from .preprocess import load_kernel_versions, process_notebook
from .retrieval import build_index, search
from .train_qlora import train
from .model import Generator
from .safety import TopologyFilter

def preprocess_cmd(args):
    """Handle preprocess command."""
    from .preprocess import main as preprocess_main
    preprocess_main()

def index_cmd(args):
    """Handle index command."""
    build_index(args.jsonl_path, args.out_dir)

def train_cmd(args):
    """Handle train command."""
    train(args)

def chat_cmd(args):
    """Handle chat command."""
    # Initialize generator
    generator = Generator(args.ckpt, args.retriever)
    
    # Load safety filter
    safety_filter = TopologyFilter(sig_db_path=args.sig_db)
    
    print("KaLactica Chat (type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        # Get user input
        prompt = input("\nYou: ").strip()
        if prompt.lower() == "exit":
            break
        
        # Generate response
        response, chunks = generator(prompt, max_tokens=args.max_tokens)
        
        # Check safety
        is_safe, stats = safety_filter.is_safe(response, args.domain)
        
        if is_safe:
            print("\nKaLactica:", response)
            if args.show_citations:
                print("\nCitations:")
                for chunk in chunks:
                    print(f"- {chunk['title']}")
        else:
            print("\nKaLactica: I apologize, but I cannot generate a safe response for this query.")
            if args.debug:
                print("\nDebug info:", json.dumps(stats, indent=2))

def nb_cmd(args):
    """Handle notebook generation command."""
    # Initialize generator
    generator = Generator(args.ckpt, args.retriever)
    
    # Load safety filter
    safety_filter = TopologyFilter(sig_db_path=args.sig_db)
    
    # Generate notebook
    prompt = f"Generate a Jupyter notebook for {args.task}"
    response, chunks = generator(prompt, max_tokens=args.max_tokens)
    
    # Check safety
    is_safe, stats = safety_filter.is_safe(response, args.domain)
    
    if is_safe:
        # Save notebook
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(response)
        
        print(f"Notebook saved to {output_path}")
        
        if args.show_citations:
            print("\nCitations:")
            for chunk in chunks:
                print(f"- {chunk['title']}")
    else:
        print("Error: Generated notebook failed safety check")
        if args.debug:
            print("\nDebug info:", json.dumps(stats, indent=2))

def main():
    parser = argparse.ArgumentParser(description="KaLactica CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess",
                                            help="Preprocess Kaggle notebooks")
    preprocess_parser.add_argument("--input", type=str, required=True,
                                 help="Path to KernelVersions.csv")
    preprocess_parser.add_argument("--sample", type=int,
                                 help="Number of notebooks to sample")
    preprocess_parser.add_argument("--output", type=str,
                                 help="Output JSONL file path")
    
    # Index command
    index_parser = subparsers.add_parser("index",
                                       help="Build FAISS index")
    index_parser.add_argument("jsonl_path", type=str,
                            help="Path to JSONL file")
    index_parser.add_argument("--out-dir", type=str,
                            help="Output directory")
    
    # Train command
    train_parser = subparsers.add_parser("train",
                                       help="Train model with QLoRA")
    train_parser.add_argument("--base-model", type=str,
                            help="Base model checkpoint")
    train_parser.add_argument("--data-path", type=str, required=True,
                            help="Path to training data")
    train_parser.add_argument("--output-dir", type=str, required=True,
                            help="Output directory")
    train_parser.add_argument("--num-epochs", type=int, default=3,
                            help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int,
                            help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float,
                            help="Learning rate")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat",
                                      help="Interactive chat mode")
    chat_parser.add_argument("--ckpt", type=str, required=True,
                           help="Model checkpoint path")
    chat_parser.add_argument("--retriever", type=str, required=True,
                           help="FAISS index directory")
    chat_parser.add_argument("--sig-db", type=str,
                           help="Signature database path")
    chat_parser.add_argument("--domain", type=str, default="nlp",
                           help="Target domain")
    chat_parser.add_argument("--max-tokens", type=int, default=400,
                           help="Maximum tokens to generate")
    chat_parser.add_argument("--show-citations", action="store_true",
                           help="Show citations")
    chat_parser.add_argument("--debug", action="store_true",
                           help="Show debug info")
    
    # Notebook command
    nb_parser = subparsers.add_parser("nb",
                                    help="Generate notebook")
    nb_parser.add_argument("--ckpt", type=str, required=True,
                         help="Model checkpoint path")
    nb_parser.add_argument("--retriever", type=str, required=True,
                         help="FAISS index directory")
    nb_parser.add_argument("--sig-db", type=str,
                         help="Signature database path")
    nb_parser.add_argument("--domain", type=str, default="nlp",
                         help="Target domain")
    nb_parser.add_argument("--task", type=str, required=True,
                         help="Notebook task description")
    nb_parser.add_argument("--output", type=str, required=True,
                         help="Output notebook path")
    nb_parser.add_argument("--max-tokens", type=int, default=400,
                         help="Maximum tokens to generate")
    nb_parser.add_argument("--show-citations", action="store_true",
                         help="Show citations")
    nb_parser.add_argument("--debug", action="store_true",
                         help="Show debug info")
    
    args = parser.parse_args()
    
    # Run command
    if args.command == "preprocess":
        preprocess_cmd(args)
    elif args.command == "index":
        index_cmd(args)
    elif args.command == "train":
        train_cmd(args)
    elif args.command == "chat":
        chat_cmd(args)
    elif args.command == "nb":
        nb_cmd(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
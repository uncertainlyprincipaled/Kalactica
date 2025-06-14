"""Command-line interface for KaLactica."""

import argparse
import json
from pathlib import Path
from typing import Optional

from .preprocess import load_kernel_versions, process_notebook
from .retrieval import Retriever
from .train_qlora import train
from .model import Generator
from .safety import TopologyFilter
from .config import AWS_CONFIG

def preprocess_cmd(args):
    """Handle preprocess command."""
    kernel_versions = load_kernel_versions(args.input)
    process_notebook(args.input, args.output)

def index_cmd(args):
    """Handle index command."""
    retriever = Retriever(
        opensearch_host=AWS_CONFIG["opensearch_host"],
        s3_bucket=AWS_CONFIG["s3_bucket"],
        region=AWS_CONFIG["region"]
    )
    retriever.build_index(args.input, args.output)

def train_cmd(args):
    """Handle train command."""
    train(
        base_model=args.base_model,
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

def chat_cmd(args):
    """Handle chat command."""
    generator = Generator(
        model_path=args.model,
        retriever_path=args.retriever if args.retriever else None
    )
    safety_filter = TopologyFilter()
    
    print("Starting chat session. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break
        
        response = generator.generate(user_input)
        is_safe, stats = safety_filter.is_safe(response)
        
        if is_safe:
            print(f"\nAssistant: {response}")
            if args.show_citations:
                print("\nCitations:")
                for chunk in generator.last_citations:
                    print(f"- {chunk['title']}")
        else:
            print("\nAssistant: I apologize, but I cannot provide that response as it may not be safe.")

def nb_cmd(args):
    """Handle notebook generation command."""
    generator = Generator(
        model_path=args.model,
        retriever_path=args.retriever if args.retriever else None
    )
    safety_filter = TopologyFilter()
    
    notebook = generator.generate_notebook(args.task)
    is_safe, stats = safety_filter.is_safe(notebook)
    
    if is_safe:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(notebook, f, indent=2)
        
        print(f"Generated notebook saved to {output_path}")
        if args.show_citations:
            print("\nCitations:")
            for chunk in generator.last_citations:
                print(f"- {chunk['title']}")
    else:
        print("Generated notebook did not pass safety checks.")

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="KaLactica CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess notebooks")
    preprocess_parser.add_argument("--input", required=True, help="Input CSV file")
    preprocess_parser.add_argument("--output", required=True, help="Output JSONL file")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Build search index")
    index_parser.add_argument("--input", required=True, help="Input JSONL file")
    index_parser.add_argument("--output", help="Output directory")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--base-model", default="facebook/galactica-1.3b", help="Base model")
    train_parser.add_argument("--data", required=True, help="Training data path")
    train_parser.add_argument("--output", required=True, help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    train_parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start chat session")
    chat_parser.add_argument("--model", required=True, help="Model path")
    chat_parser.add_argument("--retriever", help="Retriever path")
    chat_parser.add_argument("--show-citations", action="store_true", help="Show citations")
    
    # Notebook command
    nb_parser = subparsers.add_parser("nb", help="Generate notebook")
    nb_parser.add_argument("--model", required=True, help="Model path")
    nb_parser.add_argument("--retriever", help="Retriever path")
    nb_parser.add_argument("--task", required=True, help="Task description")
    nb_parser.add_argument("--output", required=True, help="Output notebook path")
    nb_parser.add_argument("--show-citations", action="store_true", help="Show citations")
    
    args = parser.parse_args()
    
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
"""QLoRA training script for KaLactica."""

import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Any, List
from tqdm import tqdm

from .config import MODEL_CONFIG
from .model import KaLactica

class NotebookDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        """Initialize dataset from JSONL file."""
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize content
        inputs = self.tokenizer(
            example["content"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Add labels for causal LM training
        inputs["labels"] = inputs["input_ids"].clone()
        
        return {k: v.squeeze(0) for k, v in inputs.items()}

def train(args):
    """Train model using QLoRA."""
    # Load model and tokenizer
    model = KaLactica(args.base_model)
    tokenizer = model.tokenizer
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Create dataset and dataloader
    dataset = NotebookDataset(args.data_path, tokenizer)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    # Train model
    trainer.train()
    
    # Save final model
    trainer.save_model(args.output_dir)

def main():
    parser = argparse.ArgumentParser(description="Train KaLactica with QLoRA")
    parser.add_argument("--base_model", type=str, default=MODEL_CONFIG["base_model"],
                      help="Base model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to training data JSONL")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,
                      default=MODEL_CONFIG["batch_size"],
                      help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                      default=MODEL_CONFIG["gradient_accumulation_steps"],
                      help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float,
                      default=MODEL_CONFIG["learning_rate"],
                      help="Learning rate")
    parser.add_argument("--lora_rank", type=int,
                      default=MODEL_CONFIG["lora_rank"],
                      help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int,
                      default=MODEL_CONFIG["lora_alpha"],
                      help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float,
                      default=MODEL_CONFIG["lora_dropout"],
                      help="LoRA dropout")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main() 
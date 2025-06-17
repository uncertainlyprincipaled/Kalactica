"""Model and generation module for KaLactica."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .retrieval import Retriever
from .config import MODEL_CONFIG

class Generator:
    def __init__(self, model_path: str, retriever_path: Optional[str] = None):
        """Initialize generator.
        
        Args:
            model_path: Path to model checkpoint or model name
            retriever_path: Path to retriever (optional)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Initialize retriever
        self.retriever = None
        if retriever_path:
            self.retriever = Retriever()
            self.retriever.load_index(retriever_path)
        
        self.last_citations = []
    
    def generate(self, prompt: str, max_tokens: int = 400) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        # Get relevant chunks if retriever is available
        chunks = []
        if self.retriever:
            chunks = self.retriever.search(prompt)
            self.last_citations = chunks
            
            # Add citations to prompt
            if chunks:
                prompt += "\n\nRelevant information:\n"
                for chunk in chunks:
                    prompt += f"- {chunk['content']}\n"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=MODEL_CONFIG["temperature"],
            top_p=MODEL_CONFIG["top_p"],
            top_k=MODEL_CONFIG["top_k"],
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_notebook(self, task: str) -> Dict[str, Any]:
        """Generate a Jupyter notebook.
        
        Args:
            task: Task description
        
        Returns:
            Notebook as dictionary
        """
        # Generate notebook structure
        prompt = f"""Create a Jupyter notebook for the following task:
{task}

The notebook should include:
1. Data loading and preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Model training and evaluation
5. Conclusions and next steps

Format the response as a valid Jupyter notebook JSON structure."""

        response = self.generate(prompt, max_tokens=1000)
        
        try:
            # Try to parse as JSON
            notebook = json.loads(response)
        except json.JSONDecodeError:
            # If not valid JSON, create a basic structure
            notebook = {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [f"# {task}\n\n" + response]
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
        
        return notebook

class KaLactica:
    """Placeholder for KaLactica main model class. Implement as needed."""
    pass 
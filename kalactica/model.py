"""Core model architecture for KaLactica."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Any, List, Optional, Tuple

from .config import MODEL_CONFIG
from .retrieval import Retriever

class KaLactica(nn.Module):
    def __init__(self, base_ckpt: str = MODEL_CONFIG["base_model"],
                 lora_path: Optional[str] = None):
        """Initialize KaLactica model.
        
        Args:
            base_ckpt: Path to base model checkpoint
            lora_path: Optional path to LoRA weights
        """
        super().__init__()
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_ckpt,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=MODEL_CONFIG["lora_rank"],
            lora_alpha=MODEL_CONFIG["lora_alpha"],
            lora_dropout=MODEL_CONFIG["lora_dropout"],
            target_modules=["q_proj", "v_proj"]
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        
        # Load LoRA weights if provided
        if lora_path:
            self.model.load_state_dict(torch.load(lora_path), strict=False)
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(self, prompt: str, max_tokens: int = 400,
                temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class Generator:
    def __init__(self, ckpt: str, retriever: Retriever):
        """Initialize generator with model and retriever.
        
        Args:
            ckpt: Path to model checkpoint
            retriever: FAISS retriever instance
        """
        self.model = KaLactica(ckpt)
        self.retriever = retriever
    
    def __call__(self, prompt: str, max_tokens: int = 400,
                 k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate text with retrieved context.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            k: Number of retrieved chunks to include
        
        Returns:
            Tuple of (generated_text, retrieved_chunks)
        """
        # Retrieve relevant chunks
        chunks = self.retriever.search(prompt, k=k)
        
        # Construct prompt with retrieved context
        context = "\n\n".join(chunk["content"] for _, chunk in chunks)
        full_prompt = f"Context:\n{context}\n\nPrompt: {prompt}\n\nResponse:"
        
        # Generate response
        response = self.model.generate(full_prompt, max_tokens=max_tokens)
        
        return response, [chunk for _, chunk in chunks]
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path)) 
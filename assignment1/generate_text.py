# text_generation.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from transformer import TransformerLM
from tokenizer import Tokenizer
import regex as re
import json

def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply softmax with temperature scaling.
    
    Args:
        logits: Unnormalized logits of shape (..., vocab_size)
        temperature: Temperature parameter (τ)
                   τ → 0: more deterministic (greedy)
                   τ = 1: standard softmax
                   τ → ∞: more uniform
    
    Returns:
        Temperature-scaled probability distribution
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Apply softmax
    return F.softmax(scaled_logits, dim=-1)

def top_p_filtering(probs: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling filter.
    
    Args:
        probs: Probability distribution of shape (..., vocab_size)
        top_p: Cumulative probability threshold (p)
    
    Returns:
        Filtered probability distribution with low-probability tokens set to zero
    """
    if top_p < 0 or top_p > 1:
        raise ValueError("top_p must be between 0 and 1")
    
    if top_p == 1.0:
        return probs  # No filtering
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for tokens to keep (remove tokens with cumulative probability > top_p)
    keep_mask = cumulative_probs <= top_p
    
    # Always keep at least one token
    if not keep_mask.any():
        keep_mask[..., 0] = True
    
    # Create the filtered distribution
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(-1, sorted_indices, sorted_probs * keep_mask.float())
    
    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    return filtered_probs

def sample_from_distribution(probs: torch.Tensor) -> torch.Tensor:
    """
    Sample from a probability distribution.
    
    Args:
        probs: Probability distribution of shape (..., vocab_size)
    
    Returns:
        Sampled token indices of shape (...)
    """
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

class TextGenerator:
    def __init__(self, model: TransformerLM, tokenizer: Tokenizer, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Get special tokens
        self.eos_token_id = tokenizer.token_to_id.get(b'<|endoftext|>')
        if self.eos_token_id is None:
            # Try to find it in the vocabulary
            for token_id, token_bytes in tokenizer.id_to_token.items():
                if token_bytes == b'<|endoftext|>':
                    self.eos_token_id = token_id
                    break
        
        print(f"EOS token ID: {self.eos_token_id}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 30,
        temperature: float = 1.0,
        top_p: float = 1.0,
        include_prompt: bool = True
    ) -> str:
        """
        Generate text completion for a given prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Temperature for sampling (1.0 = standard softmax)
            top_p: Top-p sampling parameter (1.0 = no filtering)
            include_prompt: Whether to include the prompt in the output
        
        Returns:
            Generated text completion
        """
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        
        max_prompt_length = self.model.context_length - 1
        if len(prompt_tokens) > max_prompt_length:
            prompt_tokens = prompt_tokens[:max_prompt_length]
            print(f"Warning: Prompt truncated to {len(prompt_tokens)} tokens")
        
        if not prompt_tokens:
            raise ValueError("Prompt tokenization failed")
        
        # Initialize generation
        generated_tokens = prompt_tokens.copy()
        current_length = len(generated_tokens)
        
        with torch.no_grad():
            for step in range(max_length):
                # Prepare input tensor
                current_length = len(generated_tokens)
                if current_length >= self.model.context_length:
                    input_tokens = generated_tokens[-self.model.context_length:]
                else:
                    input_tokens = generated_tokens
                    
                input_tokens = torch.tensor([input_tokens], device=self.device)
                
                # Get model predictions
                logits = self.model(input_tokens)  # (batch_size, seq_len, vocab_size)
                
                # Get the last token's logits (prediction for next token)
                next_token_logits = logits[0, -1, :]  # (vocab_size,)
                
                # Apply temperature scaling
                probs = softmax_with_temperature(next_token_logits, temperature)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    probs = top_p_filtering(probs.unsqueeze(0), top_p).squeeze(0)
                
                # Sample next token
                next_token = sample_from_distribution(probs.unsqueeze(0)).item()
                
                # print(f"Step {step+1}: Next token ID: {next_token}, Token: {self.tokenizer.decode([next_token])}")
                # Append to generated tokens
                generated_tokens.append(next_token)
                
                # Check for end-of-sequence token
                if next_token == self.eos_token_id:
                    break
        
        # Decode tokens to text
        if include_prompt:
            generated_text = self.tokenizer.decode(generated_tokens)
        else:
            generated_text = self.tokenizer.decode(generated_tokens[len(prompt_tokens):])
        
        return generated_text
    
    def generate_multiple(
        self,
        prompts: List[str],
        max_length: int = 30,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate multiple text completions.
        
        Args:
            prompts: List of input text prompts
            max_length: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            num_samples: Number of samples to generate per prompt
        
        Returns:
            List of generated text completions
        """
        all_generations = []
        
        for prompt in prompts:
            for _ in range(num_samples):
                generation = self.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    include_prompt=True
                )
                all_generations.append(generation)
        
        return all_generations

def load_model_from_checkpoint(
    checkpoint_path: str,
    model_config: dict,
    tokenizer: Tokenizer,
    device: str = 'cpu'
) -> TextGenerator:
    """
    Load a trained model from checkpoint and create a TextGenerator.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config: Dictionary with model configuration
        tokenizer: Tokenizer instance
        device: Device to load model on
    
    Returns:
        TextGenerator instance
    """
    # Initialize model
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create text generator
    generator = TextGenerator(model, tokenizer, device)
    
    return generator

# Example usage and testing
def main():
    # Configuration (adjust based on your trained model)
    model_config = {
        'vocab_size': 10000, # Adjust based on your model
        'context_length': 256, # Adjust based on your model
        'd_model': 512,  # Adjust based on your model
        'num_layers': 4,  # Adjust based on your model
        'num_heads': 16,   # Adjust based on your model
        'd_ff': 1344,     # Adjust based on your model
    }
    
    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    try:
        from tokenizer import Tokenizer
        tokenizer = Tokenizer.from_files(
            vocab_filepath="tinystories_vocab.json",
            merges_filepath="tinystories_merges.txt",  # Adjust path if needed
            special_tokens=["<|endoftext|>"]
        )
    except:
        print("Using basic tokenizer...")
        # Fallback tokenizer creation
        from tokenizer import Tokenizer
        import json
        with open("tests/fixtures/gpt2_vocab.json", "r") as f:
            raw_vocab = json.load(f)
        vocab = {i: token.encode("utf-8") for i, token in enumerate(raw_vocab.keys())}
        tokenizer = Tokenizer(vocab=vocab, merges=[], special_tokens=["<|endoftext|>"])
    
    # Load model
    checkpoint_path = "checkpoints/baseline_small/best_model.pt"  # Adjust path if needed
    generator = load_model_from_checkpoint(checkpoint_path, model_config, tokenizer, device)
    
    # Test prompts
    test_prompts = [
        "Once upon a time",
        "In a galaxy far, far away",
        "The quick brown fox"
    ]
    
    # Generate with different settings
    print("=== Text Generation Examples ===")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Prompt {i+1}: '{prompt}' ---")
        
        # Different sampling strategies
        strategies = [
            ("Greedy (temp=0.1)", 0.1, 1.0),
            ("Creative (temp=1.0)", 1.0, 1.0),
            ("Very Creative (temp=1.5)", 1.5, 1.0),
            ("Top-p (p=0.9)", 1.0, 0.9),
            ("Focused (temp=0.8, p=0.9)", 0.8, 0.9),
        ]
        
        for name, temp, top_p in strategies:
            try:
                generated = generator.generate(
                    prompt=prompt,
                    max_length=30,
                    temperature=temp,
                    top_p=top_p
                )
                print(f"{name}: {generated}")
            except Exception as e:
                print(f"{name}: Error - {e}")



if __name__ == "__main__":
    main()
    # debug_tokenizer_detailed()
    # test_fixed_tokenizer()
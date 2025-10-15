import sys
import os
import argparse
import timeit
import numpy as np
import torch

# Add the assignment1 directory to Python path
assignment1_dir = os.path.join(os.path.dirname(__file__), '..', 'assignment1')
sys.path.insert(0, assignment1_dir)

from transformer import *

# Import NVTX for profiling annotations
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    print("Warning: NVTX not available, profiling annotations will be disabled")


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention with NVTX annotations for profiling.
    
    Args:
        Q: Query tensor [batch_size, num_heads, seq_len, d_k]
        K: Key tensor [batch_size, num_heads, seq_len, d_k]
        V: Value tensor [batch_size, num_heads, seq_len, d_k]
        mask: Optional attention mask
    
    Returns:
        Output tensor and attention weights
    """
    d_k = Q.size(-1)
    
    if NVTX_AVAILABLE:
        with nvtx.range("computing attention scores"):
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
            
            # Apply mask if provided
            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
                
        with nvtx.range("computing softmax"):
            # Softmax over the last dimension
            attention_weights = torch.nn.functional.softmax(scores, dim=-1)
            
        with nvtx.range("final matmul"):
            # Weighted sum of values
            output = torch.matmul(attention_weights, V)
        
    else:
        # Non-annotated version
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
    return output, attention_weights


def profile_model(
    model,
    optimizer,
    inputs,
    targets,
    num_warmup=5,
    num_iterations=10,
    forward_only=False,
    device='cuda',
    use_nvtx=True
):
    """
    Profile model with NVTX annotations.
    
    Args:
        model: The transformer model to profile
        optimizer: Optimizer (e.g., AdamW)
        inputs: Input tensor [batch_size, context_length]
        targets: Target tensor [batch_size, context_length]
        num_warmup: Number of warmup iterations
        num_iterations: Number of profiling iterations
        forward_only: If True, only run forward pass
        device: Device to run on
        use_nvtx: Whether to use NVTX annotations
    """
    model.train()
    
    # Warmup phase (excluded from profiling)
    if use_nvtx and NVTX_AVAILABLE:
        with nvtx.range("warmup"):
            print(f"Running {num_warmup} warmup iterations...")
            for i in range(num_warmup):
                logits = model(inputs)
                if not forward_only:
                    loss = cross_entropy_loss(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
    else:
        print(f"Running {num_warmup} warmup iterations...")
        for i in range(num_warmup):
            logits = model(inputs)
            if not forward_only:
                loss = cross_entropy_loss(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()
                
    # Profiling phase
    print(f"Running {num_iterations} profiling iterations...")
    
    for i in range(num_iterations):
        if use_nvtx and NVTX_AVAILABLE:
            with nvtx.range(f"iteration_{i}"):
                # Forward pass
                with nvtx.range("forward_pass"):
                    logits = model(inputs)
                
                if not forward_only:
                    # Compute loss
                    with nvtx.range("compute_loss"):
                        loss = cross_entropy_loss(
                            logits.view(-1, logits.size(-1)),
                            targets.view(-1)
                        )
                    
                    # Backward pass
                    with nvtx.range("backward_pass"):
                        loss.backward()
                    
                    # Optimizer step
                    with nvtx.range("optimizer_step"):
                        optimizer.step()
                        optimizer.zero_grad()
        else:
            # Non-annotated version
            logits = model(inputs)
            if not forward_only:
                loss = cross_entropy_loss(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
        
        print(f"  Iteration {i+1}/{num_iterations} completed")
        
        
def main():
    parser = argparse.ArgumentParser(description='Profile Transformer Model with Nsight Systems')
    
    # Model hyperparameters (from Table 1)
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=256, 
                       help='Context length (128, 256, 512, 1024)')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Profiling parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_warmup', type=int, default=5, help='Number of warmup iterations')
    parser.add_argument('--num_iterations', type=int, default=10, 
                       help='Number of profiling iterations')
    parser.add_argument('--forward_only', action='store_true', 
                       help='Only profile forward pass (inference)')
    parser.add_argument('--no_nvtx', action='store_true', 
                       help='Disable NVTX annotations')
    
    # Model configuration
    parser.add_argument('--dtype', type=str, default='float32', 
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Model dtype')
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                       help='Learning rate for optimizer')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Set dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    if args.dtype == 'bfloat16' and device != 'cuda':
        print("Warning: bfloat16 not well supported on CPU/MPS, using float32")
        dtype = torch.float32
    
    # Print configuration
    print(f"\n{'='*60}")
    print("Profiling Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.num_layers}L, {args.d_model}D, {args.num_heads}H, {args.d_ff}FF")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    print(f"Dtype: {args.dtype}")
    print(f"Warmup steps: {args.num_warmup}")
    print(f"Profiling steps: {args.num_iterations}")
    print(f"Mode: {'Forward only (inference)' if args.forward_only else 'Full training step'}")
    print(f"NVTX annotations: {'Disabled' if args.no_nvtx else 'Enabled'}")
    print(f"{'='*60}\n")
    
    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype
    )
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Optionally swap attention implementation with annotated version
    if not args.no_nvtx and NVTX_AVAILABLE:
        print("Using annotated scaled_dot_product_attention for detailed profiling")
        import cs336_basics.model
        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    # Generate random data
    print("Generating random data...")
    torch.manual_seed(42)
    inputs = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.context_length),
        device=device
    )
    targets = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.context_length),
        device=device
    )
    
    # Run profiling
    print(f"\nStarting profiling...")
    print(f"Run with: nsys profile -o result --python-backtrace=cuda python {sys.argv[0]}")
    print(f"Or with PyTorch annotations: nsys profile -o result --pytorch python {sys.argv[0]}\n")
    
    profile_model(
        model=model,
        optimizer=optimizer,
        inputs=inputs,
        targets=targets,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        forward_only=args.forward_only,
        device=device,
        use_nvtx=not args.no_nvtx
    )
    
    print(f"\nProfiling complete!")
    
    # Memory usage (if CUDA)
    if device == 'cuda':
        print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
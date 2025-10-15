import sys
import os

# Add the assignment1 directory to Python path
assignment1_dir = os.path.join(os.path.dirname(__file__), '..', 'assignment1')
sys.path.insert(0, assignment1_dir)

# Now import from transformer
from transformer import *

# Import what we need from train_transformer if available
try:
    from train_transformer import *
except ImportError:
    print("Warning: Could not import from train_transformer, some features may not be available")


import argparse
import timeit
import numpy as np 
import torch 

def benchmark_model(
    model,
    inputs,
    targets,
    num_warmup=5,
    num_measurements=10,
    forward_only=False,
    device='cuda'
):
    """
    Benchmark forward and/or backward passes of a model.
    
    Args:
        model: The transformer model to benchmark
        inputs: Input tensor [batch_size, context_length]
        targets: Target tensor [batch_size, context_length]
        num_warmup: Number of warmup iterations
        num_measurements: Number of measurement iterations
        forward_only: If True, only measure forward pass
        device: Device to run on ('cuda', 'mps', or 'cpu')
    
    Returns:
        dict: Dictionary containing timing statistics
    """
    model.train()
    # Warmup phase
    print(f"Running {num_warmup} warmup iterations ...")
    for _ in range(num_warmup):
        logits = model(inputs)
        if not forward_only:
            loss = cross_entropy_loss(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            loss.backward()
            model.zero_grad()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
            
    # Measurement phase
    print(f"Running {num_measurements} measurement iterations ...")
    timings = []
    
    for i in range(num_measurements):
        start_time = timeit.default_timer()
        
        # Forward pass
        logits = model(inputs)
        
        # Backward pass (if enabled)
        if not forward_only:
            loss = cross_entropy_loss(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            loss.backward()
            model.zero_grad()
        
        # Synchronize to ensure GPU complete all operations 
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
        
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        timings.append(elapsed)
        
        print(f"    Iteration {i+1}/{num_measurements}: {elapsed*1000:.2f} ms")
        
    # Compute statistics
    timings = np.array(timings)
    results = {
        'mean': np.mean(timings),
        'std': np.std(timings),
        'min': np.min(timings),
        'max': np.max(timings),
        'median': np.median(timings),
        'timings': timings
    }
    
    return results
    

def print_results(results, pass_type="Forward+Backward"):
    """ 
    Print benchmark results in a nice format.
    """
    print(f"\n{'='*60}")
    print(f"{pass_type} Pass Timing Results")
    print(f"\n{'='*60}")
    print(f"Mean:  {results['mean']*1000:.2f} ms ({results['mean']:.6f} s)")
    print(f"Std:    {results['std']*1000:.2f} ms ({results['std']:.6f} s)")
    print(f"Min:    {results['min']*1000:.2f} ms ({results['min']:.6f} s)")
    print(f"Max:    {results['max']*1000:.2f} ms ({results['max']:.6f} s)")
    print(f"Median: {results['median']*1000:.2f} ms ({results['median']:.6f} s)")
    print(f"CV:     {(results['std']/results['mean']*100):.2f}%")
    print(f"{'='*60}\n")
    

def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer Model')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=256, help='Context length')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Benchmark parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_warmup', type=int, default=5, help='Number of warmup iterations')
    parser.add_argument('--num_measurements', type=int, default=10, help='Number of measurement iterations')
    parser.add_argument('--forward_only', action='store_true', help='Only measure forward pass')
    
    # Model configuration
    parser.add_argument('--dtype', type=str, default='float32', 
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Model dtype')
    
    parser.add_argument('--compile', action='store_true', 
                       help='Use torch.compile for optimization')
    
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
    print("Benchmark Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.num_layers}L, {args.d_model}D, {args.num_heads}H, {args.d_ff}FF")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    print(f"Dtype: {args.dtype}")
    print(f"Warmup steps: {args.num_warmup}")
    print(f"Measurement steps: {args.num_measurements}")
    print(f"Mode: {'Forward only' if args.forward_only else 'Forward + Backward'}")
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
    
    # Compile model if requested
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)   
        
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
    
    # Run benchmark
    print(f"\nStarting benchmark...")
    results = benchmark_model(
        model=model,
        inputs=inputs,
        targets=targets,
        num_warmup=args.num_warmup,
        num_measurements=args.num_measurements,
        forward_only=args.forward_only,
        device=device
    )
    
    # Print results
    pass_type = "Forward" if args.forward_only else "Forward+Backward"
    print_results(results, pass_type)
    
    # Calculate throughput
    tokens_per_step = args.batch_size * args.context_length
    throughput = tokens_per_step / results['mean']
    print(f"Throughput: {throughput:,.0f} tokens/second")
    print(f"Throughput: {throughput/1000:.2f} tokens/second")
    
    # Memory usage (if CUDA)
    if device == 'cuda':
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB\n")


if __name__=="__main__":
    main()
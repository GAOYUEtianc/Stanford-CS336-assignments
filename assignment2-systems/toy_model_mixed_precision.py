import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    
def analyze_dtypes_with_autocast(device='cuda', autocast_dtype=torch.float16):
    """Analyze data types during forward pass with autocast."""
    
    print("="*80)
    print(f"Mixed Precision Analysis with {autocast_dtype}")
    print("="*80)
    print()
    
    # Create model and move to GPU
    model = ToyModel(in_features=512, out_features=10).to(device)
    
    # Verify initial parameter dtype
    print("BEFORE autocast context:")
    print(f"  Model parameters dtype: {next(model.parameters()).dtype}")
    print()
    
    # Create input
    x = torch.randn(32, 512, device=device)
    
    # Storage for intermediate outputs
    outputs = {}
    
    # Register hooks to capture intermediate values
    def make_hook(name):
        def hook(module, input, output):
            outputs[name] = output
        return hook
    
    model.fc1.register_forward_hook(make_hook('fc1_output'))
    model.ln.register_forward_hook(make_hook('ln_output'))
    model.fc2.register_forward_hook(make_hook('fc2_output'))
    
    # Run with autocast
    with torch.autocast(device_type=device, dtype=autocast_dtype):
        print("INSIDE autocast context:")
        
        # Check parameter dtypes inside context
        print(f"  1. Model parameters dtype: {next(model.parameters()).dtype}")
        print(f"     (Parameters are NOT automatically cast)")
        print()
        
        # Forward pass
        logits = model(x)
        
        # Check intermediate outputs
        print(f"  2. fc1 output (Linear + ReLU) dtype: {outputs['fc1_output'].dtype}")
        print(f"     (MatMul operations are cast to {autocast_dtype})")
        print()
        
        print(f"  3. LayerNorm output dtype: {outputs['ln_output'].dtype}")
        print(f"     (Normalization are cast to {autocast_dtype})")
        print()
        
        print(f"  4. fc2 output (logits) dtype: {logits.dtype}")
        print(f"     (Final MatMul cast to {autocast_dtype})")
        print()
        
        # Compute loss
        targets = torch.randint(0, 10, (32,), device=device)
        loss = nn.CrossEntropyLoss()(logits, targets)
        
        print(f"  5. Loss dtype: {loss.dtype}")
        print(f"     (Loss computation kept in FP32)")
        print()
    
    # Backward pass (outside autocast)
    loss.backward()
    
    # Check gradient dtypes
    print("AFTER backward pass (outside autocast):")
    print(f"  6. fc1.weight.grad dtype: {model.fc1.weight.grad.dtype}")
    print(f"     (Gradients match parameter dtype: FP32)")
    print()
    
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"""
With autocast({autocast_dtype}):

1. Model parameters: FP32 (never automatically cast)
2. fc1 output (Linear): {autocast_dtype} (MatMul eligible for autocasting)
3. LayerNorm output: fp32 (normalization kept in high precision)
4. Logits: {autocast_dtype} (MatMul eligible for autocasting)
5. Loss: FP32 (reductions kept in high precision)
6. Gradients: FP32 (match parameter dtype)

Key Points:
- Parameters stay in FP32; only operations are cast
- Linear layers (MatMul), → cast to {autocast_dtype}
- LayerNorm, reductions, loss → kept in FP32
- Gradients computed in FP32 for stability
- Backward pass uses same rules as forward pass
""")
    

def compare_layernorm_sensitivity():
    """Demonstrate why LayerNorm is sensitive to precision."""
    
    print("="*80)
    print("LayerNorm Sensitivity Analysis")
    print("="*80)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test input
    x = torch.randn(2, 512, device=device) * 10  # Scale up to test dynamic range
    
    # LayerNorm in different precisions
    ln_fp32 = nn.LayerNorm(512).to(device)
    ln_fp16 = nn.LayerNorm(512).to(device).half()
    ln_bf16 = nn.LayerNorm(512).to(device).bfloat16()
    
    # Test FP32
    out_fp32 = ln_fp32(x)
    
    # Test FP16
    with torch.autocast(device_type=device, dtype=torch.float16):
        out_fp16_autocast = ln_fp32(x)  # autocast keeps LN in FP32
    out_fp16_manual = ln_fp16(x.half())  # Force FP16
    
    # Test BF16
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out_bf16_autocast = ln_fp32(x)  # autocast keeps LN in FP32
    out_bf16_manual = ln_bf16(x.bfloat16())  # Force BF16
    
    print("LayerNorm output comparison:")
    print(f"  FP32:                mean={out_fp32.mean():.6f}, std={out_fp32.std():.6f}")
    print(f"  FP16 (autocast):     mean={out_fp16_autocast.mean():.6f}, std={out_fp16_autocast.std():.6f}")
    print(f"  FP16 (manual):       mean={out_fp16_manual.float().mean():.6f}, std={out_fp16_manual.float().std():.6f}")
    print(f"  BF16 (autocast):     mean={out_bf16_autocast.mean():.6f}, std={out_bf16_autocast.std():.6f}")
    print(f"  BF16 (manual):       mean={out_bf16_manual.float().mean():.6f}, std={out_bf16_manual.float().std():.6f}")
    print()
    
    print("Why LayerNorm is sensitive:")
    print("""
  1. VARIANCE COMPUTATION: LayerNorm computes variance = mean((x - mean(x))^2)
     - Small differences become even smaller when squared
     - FP16 precision loss compounds in this computation
     - Can lead to division by very small numbers (instability)
  
  2. NORMALIZATION: Output = (x - mean) / sqrt(variance + epsilon)
     - Division by small variance can cause overflow in FP16
     - FP16 dynamic range: ~6e-5 to 65504 (limited!)
     - Large variance → small 1/sqrt(var) → underflow
     - Small variance → large 1/sqrt(var) → overflow
  
  3. LEARNED PARAMETERS: gamma (scale) and beta (shift)
     - These are typically ~1.0, but can grow during training
     - FP16 overflow risk if gamma or normalized values get large
  
  BF16 vs FP16 for LayerNorm:
  - BF16: Same exponent range as FP32 (8 bits), better dynamic range
  - BF16: Can handle variance computation more stably
  - BF16: Less likely to overflow/underflow
  - BF16: Still less precise than FP32, but usually acceptable
  
  In practice:
  - PyTorch autocast keeps LayerNorm in FP32 with FP16
  - With BF16, some frameworks allow LayerNorm in BF16
  - But keeping LN in FP32 is safer and has minimal overhead
""")

    
    
def main():
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (autocast behavior may differ)")
        device = 'cpu'
    else:
        device = 'cuda'
    
    # Problem (a) - Analyze FP16 autocast
    print("\n" + "="*80)
    print("PROBLEM (a): Data Types with FP16 Autocast")
    print("="*80 + "\n")
    analyze_dtypes_with_autocast(device, torch.float16)
    
    # Compare with BF16
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("COMPARISON: Data Types with BF16 Autocast")
        print("="*80 + "\n")
        analyze_dtypes_with_autocast(device, torch.bfloat16)
        
    # Problem (b) - LayerNorm sensitivity
    print("\n" + "="*80)
    print("PROBLEM (b): LayerNorm Sensitivity")
    print("="*80 + "\n")
    compare_layernorm_sensitivity()
    
    print("\n" + "="*80)
    print("ANSWER TO PROBLEM (b)")
    print("="*80)
    print("""
LayerNorm is sensitive to mixed precision because it involves:
(1) variance computation with small values that lose precision in FP16,
(2) division operations that risk overflow/underflow with FP16's limited
    dynamic range (~6e-5 to 65504), and
(3) accumulation of squared differences that compounds precision errors.

With BF16 instead of FP16:
- BF16 has the same dynamic range as FP32 (8 exponent bits) so overflow/
  underflow is less likely
- BF16 still has reduced precision (7 mantissa bits vs 23), so variance
  computation can still be noisy
- In practice, BF16 LayerNorm is more stable than FP16 and often acceptable,
  but PyTorch still defaults to FP32 for maximum safety

Recommendation: Keep LayerNorm in FP32 for both FP16 and BF16 mixed precision
training, as the computational cost is minimal compared to MatMuls.
""")
        

if __name__ == "__main__":
    main()
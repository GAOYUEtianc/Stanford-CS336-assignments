import torch

print("="*80)
print("Mixed Precision Accumulation Experiment")
print("="*80)
print()

# Expected result: 10.0 (1000 * 0.01)
expected = 10.0

print("1. FP32 accumulator + FP32 values")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(f"   Result: {s.item():.10f}")
print(f"   Expected: {expected:.10f}")
print(f"   Error: {abs(s.item() - expected):.2e}")
print()

print("2. FP16 accumulator + FP16 values")
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"   Result: {s.item():.10f}")
print(f"   Expected: {expected:.10f}")
print(f"   Error: {abs(s.item() - expected):.2e}")
print()

print("3. FP32 accumulator + FP16 values (no conversion)")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"   Result: {s.item():.10f}")
print(f"   Expected: {expected:.10f}")
print(f"   Error: {abs(s.item() - expected):.2e}")
print()

print("4. FP32 accumulator + FP16 values (explicit conversion)")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(f"   Result: {s.item():.10f}")
print(f"   Expected: {expected:.10f}")
print(f"   Error: {abs(s.item() - expected):.2e}")
print()

print("="*80)
print("Analysis")
print("="*80)
print("""
1. FP32 + FP32: Perfect accuracy due to sufficient precision and dynamic range.

2. FP16 + FP16: WORST accuracy! Two problems:
   - FP16 has limited precision (10 mantissa bits vs 23 in FP32)
   - When accumulator gets large (>10), adding small values (0.01) causes
     catastrophic cancellation - the small value is too small relative to 
     the accumulator and gets rounded away.
   - This is why gradient accumulation in FP16 causes training instability.

3. FP32 + FP16: Better than case 2, but still has error from FP16 representation.
   - The 0.01 value in FP16 is slightly inaccurate (~0.00999...)
   - But accumulator in FP32 maintains this error without compounding.
   - PyTorch implicitly casts FP16 to FP32 during mixed-type operations.

4. FP32 + FP16 (explicit cast): Same as case 3.
   - Explicit conversion doesn't help because the error is in the FP16
     representation of 0.01, not in the accumulation.
   - Once you represent 0.01 in FP16, you've already lost precision.

KEY INSIGHT: Always keep accumulators (sums, losses, optimizer states) in FP32,
even when using FP16/BF16 for forward/backward computations. This is why 
mixed precision training keeps certain operations in FP32.
""")

print("="*80)
print("FP16 vs BF16 Representation")
print("="*80)

# Show the actual representation differences
print("\nRepresenting 0.01 in different precisions:")
fp32_val = torch.tensor(0.01, dtype=torch.float32)
fp16_val = torch.tensor(0.01, dtype=torch.float16)
bf16_val = torch.tensor(0.01, dtype=torch.bfloat16)

print(f"FP32:  {fp32_val.item():.20f}")
print(f"FP16:  {fp16_val.item():.20f}")
print(f"BF16:  {bf16_val.item():.20f}")

print(f"\nFP16 error: {abs(fp16_val.item() - 0.01):.2e}")
print(f"BF16 error: {abs(bf16_val.item() - 0.01):.2e}")

print("\nBF16 test (same as case 2, but with BF16):")
s = torch.tensor(0, dtype=torch.bfloat16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.bfloat16)
print(f"   Result: {s.item():.10f}")
print(f"   Expected: {expected:.10f}")
print(f"   Error: {abs(s.item() - expected):.2e}")

s = torch.tensor(0, dtype=torch.bfloat16)
single_step = torch.tensor(0.01, dtype=torch.bfloat16)

print("Step-by-step accumulation:")
for i in range(5):
    old_s = s.clone()
    s += single_step
    print(f"Step {i+1}: {old_s.item():.15f} + {single_step.item():.15f} = {s.item():.15f}")

print(f"\nAfter {i+1} additions: {s.item()}")

large_num = torch.tensor(4.0, dtype=torch.bfloat16)
small_num = torch.tensor(0.01, dtype=torch.bfloat16)

print(f"Large number: {large_num.item()}")
print(f"Small number: {small_num.item()}")
print(f"4.0 + 0.01 in BF16: {(large_num + small_num).item()}")
print(f"Are they equal? {large_num + small_num == large_num}")

# Check bf precision in different range
print("==============Check next representable value in bf16 precision==========")
values = [0.1, 1.0, 2.0, 4.0, 8.0]
for v in values:
    tensor = torch.tensor(v, dtype=torch.bfloat16)
    # find next representable value
    next_val = torch.nextafter(tensor, torch.tensor(float('inf'), dtype=torch.bfloat16))
    precision = next_val.item() - tensor.item()
    print(f"Value {v:4.1f}: precision = {precision:.5f}, min step > {precision/2:.5f}")
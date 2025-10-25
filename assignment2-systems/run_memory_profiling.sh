# 2.7B model configuration (from Table 1)
MODEL_2_7B="--d_model 2560 --num_layers 32 --num_heads 32 --d_ff 10240"

echo "=========================================="
echo "Memory Profiling - 2.7B Model"
echo "=========================================="
echo ""

# Create output directory
mkdir -p memory_profiles

# Problem (a) & (b): Profile different context lengths

echo "Part (a) & (b): Profiling different context lengths"
echo "-----------------------------------"

# Context length 128
echo ""
echo "1. Context length 128 - Forward only (FP32)"
python memory_profiling_script.py \
    $MODEL_2_7B \
    --context_length 128 \
    --batch_size 4 \
    --forward_only \
    --num_warmup 3 \
    --num_profile_iters 5 \
    --output memory_profiles/2.7b_ctx128_forward_fp32.pickle \
    | tee memory_profiles/2.7b_ctx128_forward_fp32.log

echo ""
echo "2. Context length 128 - Full training step (FP32)"
python memory_profiling_script.py \
    $MODEL_2_7B \
    --context_length 128 \
    --batch_size 4 \
    --num_warmup 3 \
    --num_profile_iters 5 \
    --output memory_profiles/2.7b_ctx128_training_fp32.pickle \
    | tee memory_profiles/2.7b_ctx128_training_fp32.log

echo ""
echo "3. Context length 256 - Forward only (FP32)"
python memory_profiling_script.py \
    $MODEL_2_7B \
    --context_length 256 \
    --batch_size 4 \
    --forward_only \
    --num_warmup 3 \
    --num_profile_iters 5 \
    --output memory_profiles/2.7b_ctx256_forward_fp32.pickle \
    | tee memory_profiles/2.7b_ctx256_forward_fp32.log

echo ""
echo "4. Context length 256 - Full training step (FP32)"
python memory_profiling_script.py \
    $MODEL_2_7B \
    --context_length 256 \
    --batch_size 4 \
    --num_warmup 3 \
    --num_profile_iters 5 \
    --output memory_profiles/2.7b_ctx256_training_fp32.pickle \
    | tee memory_profiles/2.7b_ctx256_training_fp32.log

echo ""
echo "5. Context length 512 - Forward only (FP32)"
python memory_profiling_script.py \
    $MODEL_2_7B \
    --context_length 512 \
    --batch_size 2 \
    --forward_only \
    --num_warmup 3 \
    --num_profile_iters 5 \
    --output memory_profiles/2.7b_ctx512_forward_fp32.pickle \
    | tee memory_profiles/2.7b_ctx512_forward_fp32.log

echo ""
echo "6. Context length 512 - Full training step (FP32)"
python memory_profiling_script.py \
    $MODEL_2_7B \
    --context_length 512 \
    --batch_size 2 \
    --num_warmup 3 \
    --num_profile_iters 5 \
    --output memory_profiles/2.7b_ctx512_training_fp32.pickle \
    | tee memory_profiles/2.7b_ctx512_training_fp32.log

# Problem (c): Mixed precision profiling

echo ""
echo "-----------------------------------"
echo "Part (c): Mixed precision profiling"
echo "-----------------------------------"

echo ""
echo "7. Context 256 - Forward only (BF16)"
python memory_profiling_script.py \
    $MODEL_2_7B \
    --context_length 256 \
    --batch_size 4 \
    --forward_only \
    --use_amp \
    --amp_dtype bfloat16 \
    --num_warmup 3 \
    --num_profile_iters 5 \
    --output memory_profiles/2.7b_ctx256_forward_bf16.pickle \
    | tee memory_profiles/2.7b_ctx256_forward_bf16.log

echo ""
echo "8. Context 256 - Full training step (BF16)"
python memory_profiling_script.py \
    $MODEL_2_7B \
    --context_length 256 \
    --batch_size 4 \
    --use_amp \
    --amp_dtype bfloat16 \
    --num_warmup 3 \
    --num_profile_iters 5 \
    --output memory_profiles/2.7b_ctx256_training_bf16.pickle \
    | tee memory_profiles/2.7b_ctx256_training_bf16.log

# Summary
echo ""
echo "=========================================="
echo "All profiling complete!"
echo "=========================================="
echo ""
echo "Generated files in memory_profiles/:"
ls -lh memory_profiles/*.pickle
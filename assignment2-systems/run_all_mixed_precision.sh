echo "Running mixed precision benchmarks..."
echo "=========================================="

# Test small model
echo "Testing SMALL model (124M params)..."
echo "-----------------------------------"

echo "  a) FP32 baseline..."
python benchmarking_script.py \
    --d_model 768 --num_layers 12 --num_heads 12 --d_ff 3072 \
    --batch_size 8 --context_length 256 \
    --forward_only --num_warmup 5 --num_measurements 10 \
    | tee results_small_fp32.txt

echo ""
echo "  b) BF16 mixed precision..."
python benchmarking_script.py \
    --d_model 768 --num_layers 12 --num_heads 12 --d_ff 3072 \
    --batch_size 8 --context_length 256 \
    --forward_only --num_warmup 5 --num_measurements 10 \
    --use_amp --amp_dtype bfloat16 \
    | tee results_small_bf16.txt

echo ""
echo "-----------------------------------"
echo "Small model complete!"
echo ""

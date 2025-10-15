#!/bin/bash

# Script to run profiling experiments with different configurations
# Based on Table 1 model sizes and various context lengths

# Create output directory for profiling results
mkdir -p profiling_results

# Model configurations from Table 1
# Small model: 124M parameters
MODEL_SMALL="--d_model 768 --num_layers 12 --num_heads 12 --d_ff 3072"

# Medium model: 350M parameters  
MODEL_MEDIUM="--d_model 1024 --num_layers 24 --num_heads 16 --d_ff 4096"

# Large model: 774M parameters
MODEL_LARGE="--d_model 1280 --num_layers 36 --num_heads 20 --d_ff 5120"

# Context lengths to test
CONTEXT_LENGTHS=(128 256 512 1024)

# Function to run a single profiling experiment
run_profile() {
    local model_name=$1
    local model_config=$2
    local context_length=$3
    local mode=$4  # "inference" or "training"
    
    local output_file="profiling_results/${model_name}_ctx${context_length}_${mode}"
    
    echo "=========================================="
    echo "Profiling: ${model_name}, context=${context_length}, mode=${mode}"
    echo "=========================================="
    
    if [ "$mode" == "inference" ]; then
        nsys profile \
            -o "$output_file" \
            --trace=cuda,nvtx \
            --force-overwrite=true \
            --stats=true \
            python profile_with_nsys.py \
                $model_config \
                --context_length $context_length \
                --forward_only \
                --num_warmup 5 \
                --num_iterations 10 \
                --batch_size 8 \
                2>&1 | tee "${output_file}.log"
    else
        nsys profile \
            -o "$output_file" \
            --trace=cuda,nvtx \
            --force-overwrite=true \
            --stats=true \
            python profile_with_nsys.py \
                $model_config \
                --context_length $context_length \
                --num_warmup 5 \
                --num_iterations 10 \
                --batch_size 8 \
                2>&1 | tee "${output_file}.log"
    fi
    
    echo "Results saved to: ${output_file}.nsys-rep"
    echo ""
}

# Main execution

echo "Starting profiling experiments..."
echo "This will generate .nsys-rep files that can be viewed in Nsight Systems"
echo ""

# Example: Profile small model with different context lengths (inference)
echo "=== Profiling Small Model (Inference) ==="
for ctx in "${CONTEXT_LENGTHS[@]}"; do
    run_profile "small" "$MODEL_SMALL" $ctx "inference"
done

# Example: Profile small model with context length 256 (full training step)
echo "=== Profiling Small Model (Training) ==="
run_profile "small" "$MODEL_SMALL" 256 "training"


echo "=== Profiling Medium Model (Inference) ==="
for ctx in "${CONTEXT_LENGTHS[@]}"; do
    run_profile "medium" "$MODEL_MEDIUM" $ctx "inference"
done

echo "=== Profiling Large Model (Inference) ==="
for ctx in 128 256; do  # Only shorter contexts to avoid OOM
    run_profile "large" "$MODEL_LARGE" $ctx "inference"
done

echo "=========================================="
echo "All profiling complete!"
echo "View results with: nsight-sys profiling_results/*.nsys-rep"
echo "=========================================="
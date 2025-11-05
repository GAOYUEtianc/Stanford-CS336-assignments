#!/usr/bin/env python3
"""
FlashAttention-2 Implementation
Problem: flash_forward, flash_backward, flash_benchmarking
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# Part (a): Pure PyTorch FlashAttention-2 Forward Pass
# ============================================================================

class FlashAttention2PyTorch(torch.autograd.Function):
    """
    Pure PyTorch implementation of FlashAttention-2 forward pass.
    Follows Algorithm 1 from the assignment.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 forward pass in PyTorch (tiled).
        
        Args:
            Q: [batch, seq_len_q, d_head] or [batch, n_heads, seq_len_q, d_head]
            K: [batch, seq_len_k, d_head] or [batch, n_heads, seq_len_k, d_head]
            V: [batch, seq_len_k, d_head] or [batch, n_heads, seq_len_k, d_head]
            is_causal: bool, whether to apply causal masking
        
        Returns:
            O: Same shape as Q
        """
        # Handle both 3D and 4D inputs
        assert Q.dim() == 3, "Input must be 3D: [batch, seq, dim]"
        
        batch, seq_len_q, d_head = Q.shape
        _, seq_len_k, _ = K.shape
        
        # Tile sizes
        Q_TILE_SIZE = 64  # Bq
        K_TILE_SIZE = 64  # Bk
        
        scale = 1.0 / math.sqrt(d_head)
        
        # Initialize output and logsumexp
        O = torch.zeros_like(Q)
        L = torch.zeros(batch, seq_len_q, device=Q.device, dtype=torch.float32)
        
        # Split Q into tiles
        Tq = math.ceil(seq_len_q / Q_TILE_SIZE)
        Tk = math.ceil(seq_len_k / K_TILE_SIZE)
        
        for b in range(batch):
            for i in range(Tq):
                # Query tile bounds
                q_start = i * Q_TILE_SIZE
                q_end = min((i + 1) * Q_TILE_SIZE, seq_len_q)
                Qi = Q[b, q_start:q_end, :]  # [Bq, d]
                
                # Initialize accumulators for this query tile
                Oi = torch.zeros(q_end - q_start, d_head, device=Q.device, dtype=torch.float32)
                li = torch.zeros(q_end - q_start, device=Q.device, dtype=torch.float32)
                mi = torch.full((q_end - q_start,), float(-1e6), device=Q.device, dtype=torch.float32)
                
                for j in range(Tk):
                    # Key/Value tile bounds
                    k_start = j * K_TILE_SIZE
                    k_end = min((j + 1) * K_TILE_SIZE, seq_len_k)
                    Kj = K[b, k_start:k_end, :]  # [Bk, d]
                    Vj = V[b, k_start:k_end, :]  # [Bk, d]
                    
                    # Compute attention scores: Sij = Qi @ Kj^T / sqrt(d)
                    Sij = torch.matmul(Qi, Kj.t()) * scale  # [Bq, Bk]
                    
                    # Apply causal mask if needed
                    if is_causal:
                        # Create causal mask
                        q_idx = torch.arange(q_start, q_end, device=Q.device)[:, None]
                        k_idx = torch.arange(k_start, k_end, device=Q.device)[None, :]
                        mask = q_idx >= k_idx
                        Sij = Sij.masked_fill(~mask, float(-1e6))
                    
                    # Compute new max: m_new = max(mi, rowmax(Sij))
                    mi_new = torch.maximum(mi, Sij.max(dim=1).values)
                    
                    # Compute unnormalized softmax: P_tilde = exp(Sij - m_new)
                    P_tilde = torch.exp(Sij - mi_new[:, None])
                    
                    # Update running sum: li_new = exp(mi - mi_new) * li + rowsum(P_tilde)
                    li_new = torch.exp(mi - mi_new) * li + P_tilde.sum(dim=1)
                    
                    # Update output: Oi = diag(exp(mi - mi_new)) @ Oi + P_tilde @ Vj
                    Oi = torch.exp(mi - mi_new)[:, None] * Oi + torch.matmul(P_tilde, Vj)
                    
                    # Update running statistics
                    mi = mi_new
                    li = li_new
                
                # Final normalization: Oi = Oi / li
                Oi = Oi / li[:, None]
                
                # Compute logsumexp: Li = mi + log(li)
                Li = mi + torch.log(li)
                
                # Write to output
                O[b, q_start:q_end, :] = Oi
                L[b, q_start:q_end] = Li
        
        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        
        return O
    
    @staticmethod
    def backward(ctx, dO):
        """Backward pass using recomputation."""
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        
        d_head = Q.shape[-1]
        scale = 1.0 / math.sqrt(d_head)
        
        # Use compiled backward function
        dQ, dK, dV = flash_attention_backward_compiled(
            Q, K, V, O, dO, L, scale, is_causal
        )
        
        return dQ, dK, dV, None


# ============================================================================
# Part (b): Triton Kernel for FlashAttention-2 Forward Pass
# ============================================================================

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    FlashAttention-2 forward kernel in Triton.
    Each program instance processes one query tile for one batch element.
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    # Load Q tile
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # Initialize accumulators (use float32 for precision)
    O_accum = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    l = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m = tl.full([Q_TILE_SIZE], value=float('-inf'), dtype=tl.float32)
    
    # Query indices for causal masking
    q_offset = query_tile_index * Q_TILE_SIZE
    q_indices = q_offset + tl.arange(0, Q_TILE_SIZE)
    
    # Loop over key tiles
    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    
    for k_tile_idx in range(num_k_tiles):
        # Load K, V tiles
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Compute attention scores: S = Q @ K^T * scale
        S = tl.dot(Q, tl.trans(K_tile))  # [Q_TILE_SIZE, K_TILE_SIZE]
        S = S * scale
        
        # Apply causal mask if needed
        if is_causal:
            k_offset = k_tile_idx * K_TILE_SIZE
            k_indices = k_offset + tl.arange(0, K_TILE_SIZE)
            # Causal mask: q_idx >= k_idx
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            S = tl.where(causal_mask, S, float('-inf'))
        
        # Compute new max: m_new = max(m, rowmax(S))
        m_new = tl.maximum(m, tl.max(S, axis=1))
        
        # Compute unnormalized softmax: P_tilde = exp(S - m_new)
        P_tilde = tl.exp(S - m_new[:, None])
        
        # Update running sum: l_new = exp(m - m_new) * l + rowsum(P_tilde)
        l_new = tl.exp(m - m_new) * l + tl.sum(P_tilde, axis=1)
        
        # Update output accumulator
        # O_new = diag(exp(m - m_new)) @ O + P_tilde @ V
        scale_factor = tl.exp(m - m_new)
        O_accum = O_accum * scale_factor[:, None]
        
        # Cast P_tilde to V's dtype before matmul
        P_tilde = P_tilde.to(V_tile.dtype)
        O_accum += tl.dot(P_tilde, V_tile, acc=O_accum.to(tl.float32))
        
        # Update statistics
        m = m_new
        l = l_new
        
        # Advance K, V pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    # Final normalization: O = O / l
    O_final = O_accum / l[:, None]
    
    # Compute logsumexp: L = m + log(l)
    L = m + tl.log(l)
    
    # Cast output to correct dtype and store
    O_final = O_final.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_final, boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0,))


class FlashAttention2Triton(torch.autograd.Function):
    """
    FlashAttention-2 using Triton kernel for forward pass.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Args:
            Q: [batch, seq_len_q, d_head]
            K: [batch, seq_len_k, d_head] 
            V: [batch, seq_len_k, d_head]
            is_causal: bool
        
        Returns:
            O: Same shape as Q
        """
        # Check inputs
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        
        # Handle both 3D and 4D inputs
        input_is_3d = Q.dim() == 3
        assert input_is_3d, "Input must be 3D: [batch, seq, dim]"
        
        batch, seq_len_q, d_head = Q.shape
        _, seq_len_k, _ = K.shape
    
        
        # Tile sizes (tune these!)
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        
        scale = 1.0 / math.sqrt(d_head)
        
        # Allocate output
        O = torch.empty_like(Q)
        L = torch.empty(batch, seq_len_q, device=Q.device, dtype=torch.float32)
        
        # Launch grid: (num_query_tiles, batch)
        grid = (triton.cdiv(seq_len_q, Q_TILE_SIZE), batch)
        
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_len_q, seq_len_k,
            scale,
            D=d_head,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        
        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale
        
        return O
    
    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass using recomputation (PyTorch + torch.compile).
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        
        
        # Use compiled backward function
        dQ, dK, dV = flash_attention_backward_compiled(
            Q, K, V, O, dO, L, scale, is_causal
        )
        
        return dQ, dK, dV, None


# ============================================================================
# Part (c): Backward Pass with Recomputation (torch.compile)
# ============================================================================

@torch.compile
def flash_attention_backward_fn(Q, K, V, O, dO, L, scale, is_causal):
    """
    FlashAttention-2 backward pass with recomputation.
    Follows Equations 13-19 from the assignment.
    
    Args:
        Q, K, V, O, dO, L: tensors from forward pass
        scale: 1/sqrt(d_head)
        is_causal: bool
    
    Returns:
        dQ, dK, dV
    """
    batch, seq_len_q, d_head = Q.shape
    _, seq_len_k, _ = K.shape
    
    # Pre-compute D = rowsum(O * dO)
    D = torch.sum(O * dO, dim=-1, keepdim=True)  # [batch, seq_len_q, 1]
    
    # Recompute attention scores: S = Q @ K^T / sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if is_causal:
        # Create causal mask
        q_idx = torch.arange(seq_len_q, device=Q.device)[:, None]
        k_idx = torch.arange(seq_len_k, device=Q.device)[None, :]
        mask = q_idx >= k_idx
        S = S.masked_fill(~mask, float(-1e6))
    
    # Recompute attention weights using saved L: P = exp(S - L)
    P = torch.exp(S - L.unsqueeze(-1))  # [batch, seq_len_q, seq_len_k]
    
    # Compute gradients
    # dV = P^T @ dO
    dV = torch.matmul(P.transpose(-2, -1), dO)
    
    # dP = dO @ V^T
    dP = torch.matmul(dO, V.transpose(-2, -1))
    
    # dS = P * (dP - D)
    dS = P * (dP - D)
    
    # dQ = dS @ K / sqrt(d)
    dQ = torch.matmul(dS, K) * scale
    
    # dK = dS^T @ Q / sqrt(d)
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale
    
    return dQ, dK, dV


# Compile the backward function
flash_attention_backward_compiled = torch.compile(flash_attention_backward_fn)


# ============================================================================
# Testing and Benchmarking
# ============================================================================

def test_flashattention_correctness():
    """Test FlashAttention against reference implementation."""
    torch.manual_seed(42)
    
    batch = 2
    n_heads = 4
    seq_len = 256
    d_head = 64
    
    device = 'cuda'
    
    Q = torch.randn(batch, n_heads, seq_len, d_head, device=device, requires_grad=True)
    K = torch.randn(batch, n_heads, seq_len, d_head, device=device, requires_grad=True)
    V = torch.randn(batch, n_heads, seq_len, d_head, device=device, requires_grad=True)
    
    # Reference implementation
    def ref_attention(Q, K, V):
        scale = 1.0 / math.sqrt(d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        return torch.matmul(attn, V)
    
    # Test PyTorch version
    print("Testing PyTorch FlashAttention...")
    out_ref = ref_attention(Q, K, V)
    out_flash_pytorch = FlashAttention2PyTorch.apply(Q, K, V, False)
    
    print(f"  Max difference: {(out_ref - out_flash_pytorch).abs().max().item():.6f}")
    print(f"  Mean difference: {(out_ref - out_flash_pytorch).abs().mean().item():.6f}")
    
    # Test Triton version
    print("\nTesting Triton FlashAttention...")
    out_flash_triton = FlashAttention2Triton.apply(Q, K, V, False)
    
    print(f"  Max difference vs ref: {(out_ref - out_flash_triton).abs().max().item():.6f}")
    print(f"  Mean difference vs ref: {(out_ref - out_flash_triton).abs().mean().item():.6f}")
    
    # Test backward
    print("\nTesting backward pass...")
    dO = torch.randn_like(out_ref)
    
    out_ref.backward(dO)
    dQ_ref, dK_ref, dV_ref = Q.grad, K.grad, V.grad
    
    Q.grad = K.grad = V.grad = None
    
    out_flash_triton.backward(dO)
    dQ_flash, dK_flash, dV_flash = Q.grad, K.grad, V.grad
    
    print(f"  dQ max diff: {(dQ_ref - dQ_flash).abs().max().item():.6f}")
    print(f"  dK max diff: {(dK_ref - dK_flash).abs().max().item():.6f}")
    print(f"  dV max diff: {(dV_ref - dV_flash).abs().max().item():.6f}")


def benchmark_flashattention():
    """
    Benchmark FlashAttention vs standard attention.
    """
    import triton.testing
    
    print("\n" + "="*80)
    print("FlashAttention Benchmarking")
    print("="*80)
    
    results = []
    
    # Configurations to test
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    d_heads = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    
    batch = 1
    n_heads = 8
    
    for seq_len in seq_lens:
        for d_head in d_heads:
            for dtype in dtypes:
                try:
                    Q = torch.randn(batch, n_heads, seq_len, d_head, device='cuda', dtype=dtype, requires_grad=True)
                    K = torch.randn(batch, n_heads, seq_len, d_head, device='cuda', dtype=dtype, requires_grad=True)
                    V = torch.randn(batch, n_heads, seq_len, d_head, device='cuda', dtype=dtype, requires_grad=True)
                    
                    # Reference attention
                    def ref_fwd_bwd():
                        scale = 1.0 / math.sqrt(d_head)
                        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
                        # Causal mask
                        mask = torch.tril(torch.ones(seq_len, seq_len, device='cuda')).bool()
                        scores = scores.masked_fill(~mask, float(-1e6))
                        attn = torch.nn.functional.softmax(scores, dim=-1)
                        out = torch.matmul(attn, V)
                        out.sum().backward()
                        Q.grad = K.grad = V.grad = None
                    
                    # FlashAttention
                    def flash_fwd_bwd():
                        out = FlashAttention2Triton.apply(Q, K, V, True)
                        out.sum().backward()
                        Q.grad = K.grad = V.grad = None
                    
                    # Benchmark
                    ref_time = triton.testing.do_bench(ref_fwd_bwd, warmup=25, rep=100)
                    flash_time = triton.testing.do_bench(flash_fwd_bwd, warmup=25, rep=100)
                    
                    speedup = ref_time / flash_time
                    
                    results.append({
                        'seq_len': seq_len,
                        'd_head': d_head,
                        'dtype': str(dtype).split('.')[-1],
                        'ref_time': ref_time,
                        'flash_time': flash_time,
                        'speedup': speedup
                    })
                    
                    print(f"seq={seq_len:5d}, d={d_head:3d}, dtype={str(dtype).split('.')[-1]:8s}: "
                          f"Ref={ref_time:6.2f}ms, Flash={flash_time:6.2f}ms, Speedup={speedup:.2f}x")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"seq={seq_len:5d}, d={d_head:3d}, dtype={str(dtype).split('.')[-1]:8s}: OOM")
                        torch.cuda.empty_cache()
                    else:
                        raise
    
    return results


if __name__ == "__main__":
    print("Testing FlashAttention-2 Implementation")
    print("="*80)
    
    test_flashattention_correctness()
    
    print("\n\nRunning benchmarks...")
    results = benchmark_flashattention()
    
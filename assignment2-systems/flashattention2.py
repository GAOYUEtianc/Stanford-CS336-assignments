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
            Q: [batch, n_heads, seq_len_q, d_head]
            K: [batch, n_heads, seq_len_k, d_head]
            V: [batch, n_heads, seq_len_k, d_head]
            is_causal: bool, whether to apply causal masking
        
        Returns:
            O: [batch, n_heads, seq_len_q, d_head]
        """
        batch, n_heads, seq_len_q, d_head = Q.shape
        _, _, seq_len_k, _ = K.shape
        
        # Tile sizes
        Q_TILE_SIZE = 64  # Bq
        K_TILE_SIZE = 64  # Bk
        
        scale = 1.0 / math.sqrt(d_head)
        
        # Initialize output and logsumexp
        O = torch.zeros_like(Q)
        L = torch.zeros(batch, n_heads, seq_len_q, device=Q.device, dtype=torch.float32)
        
        # Split Q into tiles
        Tq = math.ceil(seq_len_q / Q_TILE_SIZE)
        Tk = math.ceil(seq_len_k / K_TILE_SIZE)
        
        for b in range(batch):
            for h in range(n_heads):
                for i in range(Tq):
                    # query tile bounds
                    q_start = i * Q_TILE_SIZE
                    q_end = min((i+1) * Q_TILE_SIZE, seq_len_q)
                    Qi = Q[b, h, q_start:q_end, :] # [Bq, d]
                    
                    # Initialize accumulators for this query tile
                    Oi = torch.zeros(q_end - q_start, d_head, device=Q.device, dtype=torch.float32)
                    li = torch.zeros(q_end - q_start, device=Q.device, dtype=torch.float32)
                    mi = torch.full((q_end - q_start,), float('-inf'), device=Q.device, dtype=torch.float32)
                    
                    for j in range(Tk):
                        # Key/Value tile bounds
                        k_start = j * K_TILE_SIZE
                        k_end = min((j + 1) * K_TILE_SIZE, seq_len_k)
                        Kj = K[b, h, k_start:k_end, :]  # [Bk, d]
                        Vj = V[b, h, k_start:k_end, :]  # [Bk, d]

                        # Compute attention scores : Sij = Qi @ Kj^T / sqrt(d)
                        Sij = torch.matmul(Qi, Kj.t()) * scale  # [Bq, Bk]
                        
                        # Apply causal mask if needed
                        if is_causal:
                            # create causal mask
                            q_idx = torch.arange(q_start, q_end, device=Q.device)[:, None]
                            k_idx = torch.arange(k_start, k_end, device=Q.device)[None, :]
                            mask = q_idx >= k_idx
                            Sij = Sij.masked_fill(~mask, float('-inf'))

                        # compute new max : m_new = max(mi, rowmax(Sij))
                        mi_new = torch.maximum(mi, Sij.max(dim=1).values)
                        
                        # Compute unnormalized softmax: P_tilde = exp(Sij - m_new)
                        P_tilde = torch.exp(Sij - mi_new[:, None])
                        
                        # Update running sum : li_new = exp(mi - mi_new) * li + rowsum(P_tilde)
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
                    O[b, h, q_start:q_end, :] = Oi
                    L[b, h, q_start:q_end] = Li
                    
        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        
        return 0
    
    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("Backward not implemented for PyTorch version")
                    
                    
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
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # once for one q tile
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # need to iterate over all K
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0), # Need to iterate over all V
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # consistent with Q, recoding output
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,), # consistent with Q, recoding exp sum
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    # Load Q tile
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # Initialize accumulators (use float32 for precision)
    O_accum = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32) # Store the output
    l = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) # store the sum of exp 
    m = tl.full([Q_TILE_SIZE], value=float('-inf'), dtype=tl.float32) # store the max of exp
    
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

    
import torch.nn as nn
import torch
from torch.optim.optimizer import Optimizer

from typing import Optional, List, Iterable
import matplotlib.pyplot as plt
import numpy as np
import math


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        # Initialize weight parameter with shape (out_features, in_features)
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Compute standard deviation for initialization
        std = (2/(in_features + out_features)) ** 0.5
        # Initialize weights with truncated normal distribution
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x shape is (..., in_features), self.W shape is (out_features, in_features)
        # Output shape should be (..., out_features)
        return torch.einsum('... i, o i -> ... o', x, self.W)
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        # Create embedding matrix (of dim vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3.0, b=3.0)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids is of shape (batch_size, seq_len), dtype: long
        # Output : (batch_size, seq_len, embedding_dim)
        one_hot = torch.nn.functional.one_hot(token_ids, num_classes = self.weight.shape[0]).to(self.weight.dtype)
        out = torch.einsum('... n, n d -> ... d', one_hot, self.weight)
        return out
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # Learnable gain parameter initialized to ones
        self.gain = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original dtype
        in_dtype = x.dtype
        # Upcast to float32 for numerical stability
        x = x.to(torch.float32)
        
        # Compute RMX over the last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x / rms # shape (..., d_model)
        
        # Apply learnable gain
        out = normed * self.gain
        
        # Downcast back to original dtype
        return out.to(in_dtype)
    

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            # Calculate d_ff to be approsimately 8/3 * d_model
            d_ff = int(8 * d_model / 3)
            # Round up to nearrest multiple of 64 for hardware efficiency
            d_ff = ((d_ff + 63) // 64) * 64
            
        # Three linear transformations
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Implements: FFN(x) = W2(SiLU(W1(x)) ⊙ W3(x))
        where SiLU(x) = x * σ(x) = x / (1 + e^(-x))
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Output tensor of shape (..., d_model)
        """
        # W1(x): (..., d_ff)
        gate = self.W1(x)
        
        # SiLU(W1(x)): (..., d_ff)
        silu_gate = gate * torch.sigmoid(gate)
        
        # W3(x): (..., d_ff)
        linear = self.W3(x)
        
        # Element-wise multiplication: (..., d_ff)
        gated = silu_gate * linear
        
        # Final output transformation (..., d_model)
        output = self.W2(gated)
        
        return output
    
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k: int, max_seq_len: int, device=None):
        """
        Initialize RoPE module.
        
        Args:
            theta: Base value Θ for computing rotation angles
            d_k: Dimension of query/key vectors (must be even)
            max_seq_len: Maximum sequence length
            device: Device to store buffers on
        """
        super().__init__()
        
        assert d_k % 2 == 0
        
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # Compute frequency for each dimension pair
        # For dimension pair k (1, 2, 3, ...., d_k /2 - 1), 
        # freq : 1/(Θ^(2*k/d_k))
        k = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)  # (d_k/2,)
        freqs = 1.0 / (theta ** (k/d_k)) # (d_k/2,)
        
        # Precompute cos and sin for all positions 
        # positions: (max_seq_len,)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (max_seq_len,)
        
        # Outer product to get angles : (max_seq_len, 1) * (1, d_k/2) -> (max_seq_len, d_k/2)
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0) # (max_seq_len, d_k/2)
        
        # Compute cos and sin
        # Repeat each value twice to match the interleaved structure
        # cos: [cos(theta_0), cos(theta_0), cos(theta_1), cos(theta_1), ...]
        # sin: [sin(theta_0), sin(theta_0), sin(theta_1), sin(theta_1), ...]
        cos = torch.cos(angles) # (max_seq_len, d_k/2)
        sin = torch.sin(angles) # (max_seq_len, d_k/2)
        
        # Repeat each value twice to match the interleaved structure
        cos = cos.repeat_interleave(2, dim=-1)  # (max_seq_len, d_k)
        sin = sin.repeat_interleave(2, dim=-1)  # (max_seq_len, d_k)
        
        # Register as no-persistent buffers (not saved in state_dict)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        
    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        The rotation formula for each pair [x_{2k}, x_{2k+1}] is:
        [x'_{2k}  ]   [cos(θ)  -sin(θ)] [x_{2k}  ]
        [x'_{2k+1}] = [sin(θ)   cos(θ)] [x_{2k+1}]
        
        Which expands to:
        x'_{2k}   = x_{2k} * cos(θ) - x_{2k+1} * sin(θ)
        x'_{2k+1} = x_{2k} * sin(θ) + x_{2k+1} * cos(θ)
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)
        
        Returns:
            Rotated tensor of shape (..., seq_len, d_k)
        """
        # Get cos and sin values for the given positions
        # token_posisions: (..., seq_len)
        # cos/sin buffers : (max_seq_len, d_k)
        
        # Index into precomputed cos and sin using token_positions
        cos = self.cos[token_position]  # (..., seq_len, d_k)
        sin = self.sin[token_position]  # (..., seq_len, d_k)
        
        # Split x into even and odd indices
        # x: (..., seq_len, d_k)
        # x_even: (..., seq_len, d_k/2) elements at 0,2,4,...
        # x_odd:  (..., seq_len, d_k/2) elements at 1,3,5,...
        x_even = x[..., 0::2] 
        x_odd = x[..., 1::2]
        
        # Apply rotation 
        # For the pair [x_{2k}, x_{2k+1}]:
        # x'_{2k}   = x_{2k} * cos - x_{2k+1} * sin
        # x'_{2k+1} = x_{2k} * sin + x_{2k+1} * cos
        
        # Get cos and sin for even / odd positions
        cos_even = cos[..., 0::2] # (..., seq_len, d_k/2)
        cos_odd = cos[..., 1::2] # (..., seq_len, d_k/2)
        sin_even = sin[..., 0::2] # (..., seq_len, d_k/2)
        sin_odd = sin[..., 1::2] # (..., seq_len, d_k/2)
        
        # Compute rortated components
        x_even_rot = x_even * cos_even - x_odd * sin_even
        x_odd_rot = x_even * sin_odd + x_odd * cos_odd
        
        # Interleave the results back
        # Stack and reshape to interleave
        x_out = torch.stack([x_even_rot, x_odd_rot], dim=-1) # (..., seq_len, d_k/2, 2)
        x_out = x_out.flatten(-2) # (..., seq_len, d_k)
        
        return x_out
                

def softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax operation on a tensor along a specified dimension.
    Args:
        tensor: Input tensor of arbitrary shape
        dim: Dimension along which to apply softmax
    Returns:
        Tensor of the same shape with softmax applied along the specified dimension
    """
    # Find the maximum value along the specified dimension
    # keepdim=True to maintain the dimension for broadcasting
    max_vals = tensor.max(dim=dim, keepdim=True)[0]
    
    # Substract max from the original tensor for numerical stability
    shifted = tensor - max_vals
    
    # Compute exponentials
    exp_vals = torch.exp(shifted)
    
    # Sum the exponentials along the dimension
    sum_exp = exp_vals.sum(dim=dim, keepdim=True)
    
    # Normalize by dividing by the sum
    softmax_output = exp_vals / sum_exp 
    
    return softmax_output

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """ 
    Compute scaled dot-product attention using einsum.
    Args : 
        query: Query tensor of shape (batch_size, ..., seq_len_q, d_k)
        key: Key tensor of shape (batch_size, ..., seq_len_q, d_k)
        value: Value tensor of shape (batch_size, ..., seq_len_k, d_v)
        mask: Optional boolean mask of shape (seq_len_q, seq_len_k) 
              where True = attend, False = do not attend
              
    Returns: 
        Output tensor of shape (batch_size, ..., seq_len_q, d_v)
    """
    d_k = query.size(-1)
    
    # Step 1 : Compute attention scores using einsum
    # query : (..., n, d), key: (..., m, d)
    # We want to compute dot product between each query and key
    # Output : (..., n, m)
    scores = torch.einsum('...nd,...md->...nm', query, key)
    
    # Step 2: Scale by 1/sqrt(d_k)
    scores = scores / (d_k ** 0.5)
    
    # Step 3: Apply mask if provided
    if mask is not None:
        # mask shape: (n, m) or broadcastable to (..., n, m)
        scores = scores.masked_fill(~mask, float('-inf'))
        
    # Step 4: Apply softmax to get attention weights
    attention_weights = softmax(scores, dim=-1) # (..., n, m)
    
    # Handle the case where an entire row is -inf (no valid keys to attend to)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
    # Step 5: Multiply by values
    output = torch.einsum('...nm,...mv->...nv', attention_weights, value) # (..., n, d_v)
    
    return output
    
    
class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention without RoPE (for basic test)"""
    def __init__(self, 
        d_model: int,
        num_heads: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        # Linear projections for Q, K, V and output
        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head self-attention (without RoPE).
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional boolean mask of shape (seq_len, seq_len) where True = attend, False = do not attend
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
            
        *batch_dims, seq_len, d_model = x.shape
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
            
        # Reshape gto (batch_size, seq_len, d_model)
        x_flat = x.view(batch_size, seq_len, d_model)
        
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        if mask is None:
            mask = causal_mask
        
        # Step 1: project to Q, K, V
        Q = self.W_Q(x_flat) # (batch_size, seq_len, d_model
        K = self.W_K(x_flat) # (batch_size, seq_len, d_model)
        V = self.W_V(x_flat) # (batch_size, seq_len, d_model
        
        # Step 2: Reshape to separate heads
        # New shape: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 3 : Apply scaled dot-product attention
        # Output shape: (batch_size, num_heads, seq_len, d_v)
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        
        # Step 4 : Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        # Step 5: Final output projection
        output = self.W_O(attn_output) # (batch_size, seq_len, d_model)
        output = output.view(*batch_dims, seq_len, d_model)
        return output
    

class CausalMultiHeadSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention with RoPE"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # Linear projections
        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)
        
        
        self.rope = RotaryPositionalEmbedding(
            theta = rope_theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=device
        )
        
        # Register causal mask buffer
        casual_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)
        )
        self.register_buffer('casual_mask', casual_mask, persistent=False)
        
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply causal multi-head self-attention with RoPE.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional token positions of shape (..., seq_len)
                           If None, uses [0, 1, 2, ..., seq_len-1]
            mask: Optional additional mask. If None, uses causal mask.
        
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # print(f"Input x shape : {x.shape}")
        # print(f"Token positions: {token_positions.shape}")
        *batch_dims, seq_len, d_model = x.shape
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
            
        # Reshape to (batch_size, seq_len, d_model)
        x = x.view(batch_size, seq_len, d_model)
        
        # Step 1: Project to Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # Step 2 : Reshape to separate heads of shape (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # Step 3 : Apply RoPE to Q and K
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0) # shape (1, seq_len)
                       
        Q = self.rope(Q, token_positions) # (batch_size, num_heads, seq_len, d_k)
        K = self.rope(K, token_positions) # (batch_size, num_heads,
        
        # Step 4: Get mask
        if mask is None:
            # Use casual mask which is a lower triangular matrix
            mask = self.casual_mask[:seq_len, :seq_len]
            
        # Step 5 : Get attention output
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        
        # Step 6 : Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Step 7: Apply output projection
        output = self.W_O(attn_output)
        output = output.view(*batch_dims, seq_len, self.d_model)
            
        return output
        
        
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with:
    1. Multi-head self-attention sublayer
    2. Feed-forward network sublayer
    Each sublayer uses: x + Sublayer(RMSNorm(x))
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
        dtype=None
    ):
        """
        Initialize Transformer block.
        Args:
            d_model: Dimensionality of model embeddings
            num_heads: Number of attention heads
            d_ff: Dimensionality of feed-forward inner layer
            rope_theta: Theta parameter for RoPE
            max_seq_len: Maximum sequence length
            device: Device to create parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Pre-norm for attention sublayer
        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        # Pre-norm for feed-forward sublayer
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Multi-head self-attention
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )
        
        # Feed-forward network
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        

    def forward(
        self, 
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Optional token positions of shape (batch_size, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First sublayer: Multi-head self-attention with residual connection
        # y = x + MultiHeadSelfAttention(RMSNorm(x))
        attn_input = self.attn_norm(x)
        attn_output = self.attn(attn_input, token_positions)
        x = x + attn_output  # Residual connection
        
        # Second sublayer: Feed-forward network with residual connection
        # y = x + FFN(RMSNorm(x))
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output  # Residual connection
        
        return x
        

class TransformerLM(nn.Module):
    """
    Complete Transformer Language Model.
    
    Architecture:
    1. Token embedding
    2. Stack of Transformer blocks
    3. Final RMSNorm
    4. Output projection to vocabulary
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        """
        Initialize Transformer Language Model.
        
        Args:
            vocab_size: Size of the vocabulary
            context_length: Maximum context length
            d_model: Dimensionality of model embeddings
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads
            d_ff: Dimensionality of feed-forward inner layer
            rope_theta: Theta parameter for RoPE
            device: Device to create parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Token embedding layer
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Output projection to vocabulary (no bias)
        self.output_projection = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Transformer LM.
        
        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
        
        Returns:
            Logits over vocabulary of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Step 1: Token embedding
        x = self.token_embedding(token_ids)  # (batch_size, seq_len, d_model)
        
        # Step 2: Generate token positions for RoPE
        # Each token at position i in the sequence
        token_positions = torch.arange(seq_len, device=token_ids.device) # (seq_len,)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1) # (batch_size, seq_len)
        
        # Step 3: Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, token_positions=token_positions)
            
        # Step 4: Final normalization
        x = self.final_norm(x)  # (batch_size, seq_len, d_model)
        
        # Step 5: Output projection to vocabulary
        logits = self.output_projection(x)  # (batch_size, seq_len, vocab_size
        return logits
    
    
class AdamW(Optimizer):
    """
    Implements AdamW optimizer.
    
    AdamW is Adam with decoupled weight decay as described in:
    "Decoupled Weight Decay Regularization" - Loshchilov & Hutter (2019)
    
    Arguments:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        eps: term added to denominator for numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0.01)
    """ 
    def __init__(
        self, 
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure: A closure that reevaluates the model and returns the loss (optional)
                     When using LBFGS or other optimizers that require multiple evaluations.
                     closure is required as it needs to be called within a torch.enable_grad() context.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        # Iterate through parameter groups
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            # Iterate through parameters
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                # Get or initialize state for this parameter
                state = self.state[p] # state is a defaultdict(dict) in Optimizer
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # First moment vector
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Second moment vector
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                m = state['m']
                v = state['v']
                
                # Increment step counter
                state['step'] += 1
                t = state['step']
                
                # Update first moment estimate : m = β1 * m + (1 - β1) * grad
                m.mul_(beta1).add_(grad, alpha = 1 - beta1)
                
                # Update second moment estimate : v = β2 * v + (1 - β2) * (grad ⊙ grad)
                v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                
                # Compute bias-corrected learning rate
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                
                # Update parameters
                p.addcdiv_(m, v.sqrt().add(eps), value= -step_size)
                # Apply weight decay, note that this is decoupled from the lr udpate
                if weight_decay != 0:
                    p.add_(p, alpha = -lr * weight_decay)
                    
        return loss
                

    
def count_parameters(model: nn.Module) -> int:
    """ Count the number of trainable params in transformer"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def FLOPs(seq_len, d_model, num_heads, d_ff, num_layers):
    """
    Rough estimate of FLOPs for a single forward pass of Transformer LM.
    
    Args:
        seq_len: Sequence length
        d_model: Dimensionality of model embeddings
        num_heads: Number of attention heads
        d_ff: Dimensionality of feed-forward inner layer
        num_layers: Number of Transformer blocks
    """
    return seq_len * (
        4 * vocab_size * d_model + 
        num_layers * (
            8 * d_model**2 +
            4 * d_model * seq_len +
            6 * d_model * d_ff
        )
    )


def flops_components(seq_len, d_model, num_heads, d_ff, num_layers):
    """
    Return a dict of FLOPs broken down by Transformer LM components.
    """
    # Embedding + final logits projection (one-hot matmul implementation)
    embed_flops = 2 * seq_len * vocab_size * d_model  # embedding
    final_flops = 2 * seq_len * d_model * vocab_size  # final logits

    # Q, K, V projections per layer
    qkv_flops = num_layers * (6 * seq_len * d_model * d_model)

    # Attention score and weighted sum (quadratic in seq_len)
    attn_flops = num_layers * (4 * seq_len**2 * d_model)

    # Output projection after attention
    wo_flops = num_layers * (2 * seq_len * d_model * d_model)

    # Feed-forward (SwiGLU style)
    ffn_flops = num_layers * (6 * seq_len * d_model * d_ff)

    return {
        "Embedding+FinalVocab": embed_flops + final_flops,
        "QKV": qkv_flops,
        "Attention (QK, softmax·V)": attn_flops,
        "WO": wo_flops,
        "FFN": ffn_flops,
    }


def total_flops(seq_len, d_model, num_heads, d_ff, num_layers):
    comps = flops_components(seq_len, d_model, num_heads, d_ff, num_layers)
    return sum(comps.values())


def plot_flops_vs_seq(model_cfg, seq_lens):
    """
    Plot component-wise FLOPs as a proportion of total FLOPs vs sequence length.
    model_cfg: dict with d_model, num_heads, d_ff, num_layers, name
    seq_lens: list/array of sequence lengths
    """
    proportions = {key: [] for key in [
        "Embedding+FinalVocab", "QKV", "Attention (QK, softmax·V)", "WO", "FFN"
    ]}

    for L in seq_lens:
        comps = flops_components(L, model_cfg['d_model'], model_cfg['num_heads'], model_cfg['d_ff'], model_cfg['num_layers'])
        total = sum(comps.values())
        for k in proportions:
            proportions[k].append(comps[k] / total)

    # Plot
    plt.figure(figsize=(8,5))
    for k, vals in proportions.items():
        plt.plot(seq_lens, vals, label=k)
    plt.title(f"FLOPs Proportion vs Sequence Length ({model_cfg['name']})")
    plt.xlabel("Sequence length")
    plt.ylabel("Proportion of total FLOPs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def cross_entropy_loss(inputs: torch.Tensor, 
                       targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute the average cross-entropy loss across a batch.

    Args:
        inputs (Tensor): Unnormalized logits of shape (batch_size, vocab_size).
        targets (Tensor): Indices of the correct class (batch_size,).

    Returns:
        Tensor: Scalar tensor (float), the average cross-entropy loss.
    """
    # Step 1 : Shift logits for numerical stability
    shifted_logits = inputs - inputs.max(dim=-1, keepdim=True).values  # (batch_size, vocab_size)
    
    # Step 2 : Compute log-sum-exp
    log_sum_exp = shifted_logits.exp().sum(dim=-1).log()  # (batch_size,)
    
    # Step 3 : Gather the logits corresponding to the target classes
    target_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (batch_size,)
    
    # Step 4 : Compute per-sample loss: - (logit_correct - logsumexp)
    losses = -(target_logits - log_sum_exp) # (batch_size,)
    return losses.mean()  # Average over the batch
    

def cosine_learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    """
    Compute learning rate at iteration t using cosine annealing with linear warmup.
    
    Args:
        it: Current iteration number (t)
        max_learning_rate: α_max, maximum learning rate
        min_learning_rate: α_min, minimum/final learning rate
        warmup_iters: T_w, number of warmup iterations
        cosine_cycle_iters: T_c, total iterations for cosine annealing
    
    Returns:
        Learning rate at iteration t
    """  
    # Phase 1: Linear warmup
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    # Phase 2: Cosine annealing
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)
    
    # Phase 3: Post-annealing (t > T_c)
    else:
        return min_learning_rate
    

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    """
    Clip gradients to have L2 norm at most max_l2_norm.
    
    If ||g||_2 > max_l2_norm, scale g by max_l2_norm / (||g||_2 + eps)
    
    Args:
        parameters: Iterable of parameters with gradients
        max_l2_norm: Maximum allowed L2 norm
        eps: Small value for numerical stability (default: 1e-6)
    """
    # Step 1: Collect all gradients and compute total L2 norm
    # We need to compute the total L2 norm across all parameters
    total_norm = 0
    
    # Convert to list to allow multiple iterations
    params_with_grad = []
    for p in parameters:
        if p.grad is not None:
            params_with_grad.append(p)
            # Compute squared norm of this parameter's gradient
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    # Compute total L2 norm
    total_norm = total_norm ** 0.5
    # Compute the clipping coefficient
    clip_coef = max_l2_norm / (total_norm + eps)
    # Only clip if total_norm exceeds max_l2_norm
    if clip_coef < 1:
        for p in params_with_grad:
            p.grad.data.mul_(clip_coef)        

    
if __name__ == "__main__":
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 6400
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=100000.0,
        device='cpu',
        dtype=torch.float32
    )
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # In total 2,127,057,600 trainable parameters for GPT2
    # A FP32 number takes 4 bytes (32 bits), hence 8.5 GB of memory
    
    # Example usage: GPT-2 Small config
    gpt2_small = {"name": "GPT-2 Small", "d_model": 768, "num_heads": 12, "d_ff": 3072, "num_layers": 12}
    gpt2_medium = {"name": "GPT-2 Medium", "d_model": 1024, "num_heads": 16, "d_ff": 3072, "num_layers": 24}
    gpt2_large = {"name": "GPT-2 Large", "d_model": 1280, "num_heads": 20, "d_ff": 3072, "num_layers": 36}
    gpt2_xlarge = {"name": "GPT-2 XL", "d_model": 1600, "num_heads": 25, "d_ff": 6400, "num_layers": 48}
    
    seqs = np.arange(128, 4097, 256)
    plot_flops_vs_seq(gpt2_xlarge, seqs)

    
    
    

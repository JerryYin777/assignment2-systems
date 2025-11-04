"""Neural network module implementations for CSE599-O Assignment 1."""

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Embedding(nn.Module):
    """Embedding module for token embeddings.

    Maps integer token IDs to dense vectors of dimension embedding_dim.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ):
        """Construct an embedding module.

        Args:
            num_embeddings: int
                Size of the vocabulary
            embedding_dim: int
                Dimension of the embedding vectors, i.e., d_model
            device: torch.device | None = None
                Device to store the parameters on
            dtype: torch.dtype | None = None
                Data type of the parameters
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Create embedding matrix with shape (num_embeddings, embedding_dim)
        # Store with embedding_dim as the final dimension
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # Initialize weights using truncated normal distribution
        # N(μ=0, σ²=1) truncated at [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: Tensor of token IDs with shape (..., seq_len)

        Returns:
            Tensor of embeddings with shape (..., seq_len, embedding_dim)
        """
        # Use indexing to lookup embeddings for each token ID
        return self.weight[token_ids]


class Linear(nn.Module):
    """Linear transformation module without bias.

    Performs the transformation: y = W @ x
    where W has shape (out_features, in_features).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        """Construct a linear transformation module.

        Args:
            in_features: int
                Final dimension of the input
            out_features: int
                Final dimension of the output
            device: torch.device | None = None
                Device to store the parameters on
            dtype: torch.dtype | None = None
                Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Create weight parameter with shape (out_features, in_features)
        # We store W (not W^T) for memory ordering reasons
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Initialize weights using truncated normal distribution
        # N(μ=0, σ²=2/(din + dout)) truncated at [-3σ, 3σ]
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Apply the linear transformation to the input.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Use matrix multiplication: y = x @ W^T
        # This works because PyTorch uses row-major memory ordering
        return x @ self.weight.T


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes activations using RMS and applies learnable gain parameters.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        """Construct the RMSNorm module.

        Args:
            d_model: int
                Hidden dimension of the model
            eps: float = 1e-5
                Epsilon value for numerical stability
            device: torch.device | None
                Device to store the parameters on
            dtype: torch.dtype | None
                Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable gain parameters, one per dimension
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Process an input tensor and return normalized output.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of the same shape as input
        """
        # Save original dtype for later
        in_dtype = x.dtype

        # Upcast to float32 to prevent overflow when squaring
        x = x.to(torch.float32)

        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and apply learnable gain
        result = (x / rms) * self.weight

        # Return in original dtype
        return result.to(in_dtype)


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Combines SiLU activation with Gated Linear Units (GLU).
    FFN(x) = W2 * (SiLU(W1 * x) ⊙ W3 * x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        """Construct the SwiGLU module.

        Args:
            d_model: int
                Dimensionality of input and output
            d_ff: int
                Dimensionality of the hidden layer (typically 8/3 * d_model)
            device: torch.device | None
                Device to store the parameters on
            dtype: torch.dtype | None
                Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # Three linear transformations (without bias)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        # FFN(x) = W2 * (SiLU(W1 * x) ⊙ W3 * x)
        return self.w2(silu(self.w1(x)) * self.w3(x))


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """Apply softmax to a tensor along a specified dimension.

    Uses the numerical stability trick of subtracting the max value.

    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to apply softmax over

    Returns:
        Tensor of same shape with softmax applied along dim
    """
    # Subtract max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max

    # Compute softmax
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Applies rotations to query and key vectors based on their position.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        """Construct the RoPE module and create buffers.

        Args:
            theta: Theta value for RoPE
            d_k: Dimension of query/key vectors (should be even)
            max_seq_len: Maximum sequence length that will be inputted
            device: torch.device | None
                Device to store the buffers on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Precompute the rotation angles for all positions and dimensions
        # theta_i,k = i / Θ^((2k-1)/d) for k ∈ {1, ..., d/2}
        # We'll compute for positions 0 to max_seq_len-1

        # Compute frequency for each pair of dimensions
        # For k ∈ {1, ..., d/2}, we need Θ^((2k-1)/d) = Θ^((2k-2)/d)
        # Let's use k ∈ {0, ..., d/2-1} so it's Θ^(2k/d)
        k = torch.arange(0, d_k // 2, device=device)
        freqs = 1.0 / (theta ** (2 * k.float() / d_k))

        # Compute positions
        positions = torch.arange(0, max_seq_len, device=device)

        # Compute angles: outer product of positions and frequencies
        # Shape: (max_seq_len, d_k // 2)
        angles = torch.outer(positions, freqs)

        # Precompute cos and sin for all positions
        # Shape: (max_seq_len, d_k // 2)
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        # Register as buffers (not parameters, non-persistent)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        """Apply RoPE to an input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len) giving absolute positions

        Returns:
            Tensor of same shape with RoPE applied
        """
        # Get the cos and sin values for the given positions
        # Shape: (..., seq_len, d_k // 2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # Split x into pairs: (x0, x1, x2, x3, ...) -> ((x0, x1), (x2, x3), ...)
        # Reshape to (..., seq_len, d_k // 2, 2)
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)

        # Extract even and odd indices
        x_even = x_pairs[..., 0]  # (..., seq_len, d_k // 2)
        x_odd = x_pairs[..., 1]   # (..., seq_len, d_k // 2)

        # Apply rotation: [cos -sin] [x_even]
        #                 [sin  cos] [x_odd ]
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        # Stack back together
        x_rotated_pairs = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)

        # Reshape back to original shape
        return x_rotated_pairs.reshape(*x.shape)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape (..., keys, d_k)
        V: Value tensor of shape (..., keys, d_v)
        mask: Optional boolean mask of shape (..., queries, keys)
              True = attend, False = don't attend

    Returns:
        Output tensor of shape (..., queries, d_v)
    """
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # Q: (..., queries, d_k), K: (..., keys, d_k)
    # Result: (..., queries, keys)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # Apply mask if provided
    if mask is not None:
        # Where mask is False, set score to -inf
        scores = scores.masked_fill(~mask, float('-inf'))

    # Apply softmax
    attn_weights = softmax(scores, dim=-1)

    # Apply attention weights to values
    # attn_weights: (..., queries, keys), V: (..., keys, d_v)
    # Result: (..., queries, d_v)
    output = torch.matmul(attn_weights, V)

    return output


class MultiHeadSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention without RoPE.

    Implements multi-head attention with causal masking.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device=None,
        dtype=None,
    ):
        """Construct the multi-head self-attention module.

        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Head dimension
        self.d_v = d_model // num_heads

        # Query, key, value projections for all heads (batched)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Output projection
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Apply multi-head self-attention with causal masking.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)

        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        # Project to Q, K, V
        # Each has shape (..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to separate heads
        # (..., seq_len, d_model) -> (..., seq_len, num_heads, d_k)
        # -> (..., num_heads, seq_len, d_k)
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_v).transpose(-3, -2)

        # Create causal mask: token i can only attend to positions j <= i
        # Shape: (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Apply scaled dot-product attention
        # Q, K, V: (..., num_heads, seq_len, d_k/d_v)
        # Output: (..., num_heads, seq_len, d_v)
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Reshape back: (..., num_heads, seq_len, d_v) -> (..., seq_len, num_heads, d_v)
        # -> (..., seq_len, d_model)
        attn_output = attn_output.transpose(-3, -2).contiguous()
        attn_output = attn_output.view(*batch_shape, seq_len, self.d_model)

        # Apply output projection
        output = self.output_proj(attn_output)

        return output


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """Causal Multi-Head Self-Attention with RoPE.

    Implements multi-head attention with causal masking and rotary position embeddings.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        """Construct the multi-head self-attention module with RoPE.

        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Head dimension
        self.d_v = d_model // num_heads

        # Query, key, value projections for all heads (batched)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Output projection
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE module (applied to head dimension)
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Apply multi-head self-attention with RoPE and causal masking.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional positions tensor of shape (..., seq_len)

        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        # If token_positions not provided, use sequential positions
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
            # Broadcast to match batch dimensions
            for _ in range(len(batch_shape)):
                token_positions = token_positions.unsqueeze(0)
            token_positions = token_positions.expand(*batch_shape, seq_len)

        # Project to Q, K, V
        # Each has shape (..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to separate heads
        # (..., seq_len, d_model) -> (..., seq_len, num_heads, d_k)
        # -> (..., num_heads, seq_len, d_k)
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_v).transpose(-3, -2)

        # Apply RoPE to Q and K (not V)
        # Need to expand token_positions to include the num_heads dimension
        # token_positions: (..., seq_len) -> (..., num_heads, seq_len)
        token_positions_expanded = token_positions.unsqueeze(-2).expand(*batch_shape, self.num_heads, seq_len)

        # Apply RoPE: (..., num_heads, seq_len, d_k) with positions (..., num_heads, seq_len)
        Q = self.rope(Q, token_positions_expanded)
        K = self.rope(K, token_positions_expanded)

        # Create causal mask: token i can only attend to positions j <= i
        # Shape: (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Apply scaled dot-product attention
        # Q, K, V: (..., num_heads, seq_len, d_k/d_v)
        # Output: (..., num_heads, seq_len, d_v)
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Reshape back: (..., num_heads, seq_len, d_v) -> (..., seq_len, num_heads, d_v)
        # -> (..., seq_len, d_model)
        attn_output = attn_output.transpose(-3, -2).contiguous()
        attn_output = attn_output.view(*batch_shape, seq_len, self.d_model)

        # Apply output projection
        output = self.output_proj(attn_output)

        return output


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE.

    Implements: y = x + MHA(RMSNorm(x))
                output = y + FFN(RMSNorm(y))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        """Construct the Transformer block.

        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward inner layer
            max_seq_len: Maximum sequence length
            theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # First RMSNorm (before attention)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)

        # Multi-head self-attention with RoPE
        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

        # Second RMSNorm (before feed-forward)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Apply the Transformer block.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional positions tensor of shape (..., seq_len)

        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # First sublayer: y = x + MHA(RMSNorm(x))
        y = x + self.attn(self.ln1(x), token_positions=token_positions)

        # Second sublayer: output = y + FFN(RMSNorm(y))
        output = y + self.ffn(self.ln2(y))

        return output


class TransformerLM(nn.Module):
    """Full Transformer language model.

    Architecture:
    Token Embeddings -> num_layers Transformer Blocks -> RMSNorm -> LM Head
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
        dtype=None,
    ):
        """Construct the Transformer language model.

        Args:
            vocab_size: Size of the vocabulary
            context_length: Maximum context length
            d_model: Dimensionality of the model
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads per block
            d_ff: Dimensionality of the feed-forward inner layer
            rope_theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # Token embeddings
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Final RMSNorm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # Language model head (output projection to vocabulary)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: Int[Tensor, "... seq_len"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len vocab_size"]:
        """Forward pass through the Transformer language model.

        Args:
            token_ids: Input token IDs of shape (..., seq_len)
            token_positions: Optional positions tensor of shape (..., seq_len)

        Returns:
            Logits over vocabulary of shape (..., seq_len, vocab_size)
        """
        # Embed tokens
        x = self.token_embeddings(token_ids)

        # If token_positions not provided, use sequential positions
        if token_positions is None:
            seq_len = token_ids.shape[-1]
            batch_shape = token_ids.shape[:-1]
            token_positions = torch.arange(seq_len, device=token_ids.device)
            # Broadcast to match batch dimensions
            for _ in range(len(batch_shape)):
                token_positions = token_positions.unsqueeze(0)
            token_positions = token_positions.expand(*batch_shape, seq_len)

        # Apply Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # Apply final RMSNorm
        x = self.ln_final(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits

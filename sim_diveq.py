import torch
from torch import nn
import torch.nn.functional as F

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class SimDiVeQ(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        frozen_codebook_dim: int = None,
        channel_first: bool = False,
        eps: float = 1e-5
    ):
        """
        SimDiVeQ: Combining SimVQ (Linear Parametrization) and DiVeQ (Gradient Trick).
        
        Args:
            dim: Output dimension.
            codebook_size: Number of entries.
            frozen_codebook_dim: Dimension of frozen vectors.
            channel_first: If True, input is (B, C, ...). Mask should still be (B, ...).
            eps: Stability epsilon.
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.channel_first = channel_first
        self.eps = eps
        
        # SimVQ: Frozen Codebook + Trainable Linear Layer
        frozen_dim = default(frozen_codebook_dim, dim)
        
        # Frozen Gaussian initialization
        frozen_codes = torch.randn(codebook_size, frozen_dim) * (frozen_dim ** -0.5)
        self.register_buffer('frozen_codebook', frozen_codes)

        # The only trainable parameter: W
        self.code_transform = nn.Linear(frozen_dim, dim, bias=False)

    @property
    def codebook(self):
        # E_effective = E_frozen @ W
        return self.code_transform(self.frozen_codebook)

    def indices_to_codes(self, indices):
        frozen_selected = F.embedding(indices, self.frozen_codebook)
        quantized = self.code_transform(frozen_selected)
        if self.channel_first:
            quantized = quantized.movedim(-1, 1)
        return quantized

    def forward(self, x, mask = None):
        """
        Args:
            x: Input tensor (B, ..., D) or (B, D, ...) if channel_first.
            mask: Boolean mask (B, ...). True = Valid, False = Ignored/Padding.
                  Shape must match x's spatial dimensions.
        """
        # 1. Handle Channel Dimension
        # x: (B, C, ...) -> (B, ..., C)
        if self.channel_first:
            x = x.movedim(1, -1)
            
        input_shape = x.shape
        D = input_shape[-1]
        
        # Flatten x: (N, D)
        x_flat = x.reshape(-1, D)

        # 2. Handle Mask
        mask_flat = None
        if exists(mask):
            # mask shape should be (B, ...), same as x excluding last dim
            # Check consistency (optional, usually assumed correct)
            assert mask.shape == input_shape[:-1], f"Mask shape {mask.shape} must match input spatial shape {input_shape[:-1]}"
            
            # Flatten mask: (N,)
            mask_flat = mask.reshape(-1)

        # 3. SimVQ: Get Codebook
        implicit_codebook = self.codebook

        # 4. Nearest Neighbor
        with torch.no_grad():
            dist = torch.cdist(x_flat, implicit_codebook)
            indices = dist.argmin(dim=-1)

        # 5. Get Quantized Vectors (q)
        quantized = F.embedding(indices, implicit_codebook)

        # -----------------------------------------------------------
        # 6. DiVeQ Gradient Trick with Masking
        # -----------------------------------------------------------
        
        # Calculate diff: q - z
        diff = quantized - x_flat
        
        # Calculate Magnitude: ||q - z||
        # (N, 1)
        dist_magnitude = diff.norm(p=2, dim=-1, keepdim=True)
        
        # Apply Mask to Magnitude
        # If mask is False (Invalid/Pad), dist_magnitude becomes 0.
        # This kills the gradient flow for those tokens.
        if exists(mask_flat):
            dist_magnitude = dist_magnitude * mask_flat.unsqueeze(-1)
        
        # Calculate Direction: (q - z) / ||q - z||
        direction = diff / dist_magnitude.clamp(min=self.eps)
        
        # Apply Formula: z_q = z + ||q-z|| * sg[direction]
        # For masked tokens: dist_magnitude is 0, so output is z (Identity).
        # For valid tokens: dist_magnitude is ||q-z||, so output is q.
        quantized_out = x_flat + dist_magnitude * direction.detach()

        # -----------------------------------------------------------

        # 7. Restore Shapes
        quantized_out = quantized_out.view(input_shape)
        indices = indices.view(input_shape[:-1])

        if self.channel_first:
            quantized_out = quantized_out.movedim(-1, 1)

        # Loss is 0 (implicit in gradients)
        loss = torch.tensor(0., device=x.device, requires_grad=self.training)

        return quantized_out, indices, loss

import torch
import math

def jensenshannon_pt(p: torch.Tensor, q: torch.Tensor, base: float = None, dim: int = 0, keepdim: bool = False) -> torch.Tensor:
    """
    Computes the Jensen-Shannon distance on GPU using PyTorch.
    Perfectly replicates scipy.spatial.distance.jensenshannon.
    """
    # 1. Ensure inputs are floating point tensors
    if not p.is_floating_point():
        p = p.float()
    if not q.is_floating_point():
        q = q.float()
        
    # 2. SciPy automatically normalizes inputs to probability distributions
    p = p / torch.sum(p, dim=dim, keepdim=True)
    q = q / torch.sum(q, dim=dim, keepdim=True)
    
    # 3. Calculate the pointwise mean
    m = (p + q) / 2.0
    
    # 4. Numerically stable KL divergence: x * log(x / y)
    def kl_div(x, y):
        # Prevent 0/0 and log(0) without breaking gradients or causing NaNs.
        # When x = 0, x_safe = 1 and y_safe = 1. log(1/1) = 0.
        # Since x is 0, the final product (0 * 0) is correctly 0.
        x_safe = torch.where(x == 0, torch.ones_like(x), x)
        y_safe = torch.where(x == 0, torch.ones_like(y), y)
        return x * torch.log(x_safe / y_safe)
        
    # 5. Compute Divergences
    kl_p = torch.sum(kl_div(p, m), dim=dim, keepdim=keepdim)
    kl_q = torch.sum(kl_div(q, m), dim=dim, keepdim=keepdim)
    
    # Average the divergences
    js_div = (kl_p + kl_q) / 2.0
    
    # Apply logarithmic base correction if specified
    if base is not None:
        js_div = js_div / math.log(base)
        
    # 6. Distance is the square root of divergence.
    # Clamp to 0.0 to avoid NaNs from negative zeros caused by floating-point imprecision.
    return torch.sqrt(torch.clamp(js_div, min=0.0))
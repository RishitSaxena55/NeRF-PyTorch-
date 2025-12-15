import torch

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """Hierarchical sampling function to sample points based on a given PDF.
    
    Args:
        bins: Tensor of shape [batch_size, num_bins+1] representing the bin edges.
        weights: Tensor of shape [batch_size, num_bins] representing the weights for each bin.
        N_samples: Number of samples to draw.
        det: If True, use deterministic sampling.
        pytest: If True, use fixed random numbers for testing.
    
    Returns:
        samples: Tensor of shape [batch_size, N_samples] representing the sampled points.
    """
    # Add a small value to weights to prevent division by zero
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    target_shape = list(cdf.shape[:-1]) + [N_samples]
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(target_shape)

    else:
        u = torch.rand(target_shape)
    
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds-1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds.shape[0], inds.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze_(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze_(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

# # Example usage:
# bins = torch.linspace(0, 1, steps=11)
# weights = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.3, 0.25, 0.2, 0.1])
# weights = weights / torch.sum(weights)
# N_samples = 5
# det=False

# bins = bins.unsqueeze(0)
# weights = weights.unsqueeze(0)

# samples = sample_pdf(bins, weights, N_samples, det)
# print(samples)

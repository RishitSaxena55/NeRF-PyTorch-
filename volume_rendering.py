import torch
from torch import nn

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    
    Args:
        raw: [num_rays, num_samples, 4]. Prediction from model.
        z_vals: [num_rays, num_samples]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        raw_noise_std: Standard deviation of noise added to raw density.
        white_bkgd: If True, assume a white background.
        pytest: If True, uses fixed random numbers for testing.
    
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        depth_map: [num_rays]. Estimated distance to object along a ray.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])

    raw2alpha = lambda raw, dists, act_fn=nn.ReLU(): 1.-torch.exp(-act_fn(raw)*dists)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        if pytest:
            torch.manual_seed(0)
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)

    # Ti = cumprod(1 - alpha_j) from j= 1 to i-1
    # wi = Ti * alpha_i

    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)
    weights = alpha * T[..., :-1]

    rgb_map = torch.sum(weights[..., None]*rgb, -2)

    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map

# # Dimensions
# N_rays = 2      # Batch size (e.g., 2 rays)
# N_samples = 4   # Number of points sampled along each ray
# H, W = 100, 100 # Image dimensions (not strictly needed here but good for context)

# # A. Create 'raw' (Model Predictions)
# # Shape: [N_rays, N_samples, 4] -> (RGB + Density)
# # We use standard normal distribution to simulate logits
# raw = torch.randn(N_rays, N_samples, 4) 

# # B. Create 'z_vals' (Depths along the ray)
# # Shape: [N_rays, N_samples]
# # Note: Depths must be increasing! We use sorted random numbers between 2.0 and 6.0
# z_vals, _ = torch.sort(torch.rand(N_rays, N_samples) * 4.0 + 2.0, dim=-1)

# # C. Create 'rays_d' (Ray Directions)
# # Shape: [N_rays, 3]
# # Random viewing directions (normalized)
# rays_d = torch.randn(N_rays, 3)
# rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

# # --- 3. Run the Function ---
# print("Running raw2outputs...")
# rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
#     raw, 
#     z_vals, 
#     rays_d, 
#     raw_noise_std=0.0, 
#     white_bkgd=False
# )

# # --- 4. Inspect Results ---
# print("\n=== INPUTS ===")
# print(f"Raw predictions shape: {raw.shape}")
# print(f"Z values (depths): \n{z_vals}")

# print("\n=== OUTPUTS ===")
# print(f"1. RGB Map (Final Pixel Colors) [Shape {rgb_map.shape}]:\n{rgb_map}")
# print("-" * 30)
# print(f"2. Acc Map (Opacity) [Shape {acc_map.shape}]:\n{acc_map}")
# print("-" * 30)
# print(f"3. Depth Map (Distance to object) [Shape {depth_map.shape}]:\n{depth_map}")
# print("-" * 30)
# print(f"4. Weights (Contribution of each sample) [Shape {weights.shape}]:\n{weights}")

import numpy as np
import torch


def get_rays_np(H, W, K, c2w):
    """
    Get ray origins and directions in world coordinates.

    Parameters:
    H : int
        Image height.
    W : int
        Image width.
    K : ndarray of shape (3, 3)
        Camera intrinsic matrix.
    c2w : ndarray of shape (4, 4)
        Camera-to-world transformation matrix.

    Returns:
    rays_o : ndarray of shape (H*W, 3)
        Ray origins in world coordinates.
    rays_d : ndarray of shape (H*W, 3)
        Ray directions in world coordinates.
    """

    # K = [[fx, 0, cx],
    #      [0, fy, cy],
    #      [0,  0,  1]]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    dirs = np.stack(((i - cx) / fx, -(j - cy) / fy, np.ones_like(i)), axis=-1)
    # Rotate ray directions from camera frame to the world frame
    rot_matrix = c2w[:3, :3]  # (3, 3)
    rays_d = dirs @ rot_matrix.T  # (H, W, 3)

    # The origin of all rays is the camera origin in world coordinates
    trans_cam = c2w[:3, -1] # (3,)
    rays_o = np.broadcast_to(trans_cam, rays_d.shape)  # (H*W, 3 )

    return rays_o, rays_d

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                            torch.arange(H, dtype=torch.float32),
                            indexing='xy')
    
    dirs = torch.stack([(i - K[0,2]) / K[0,0],
                        -(j - K[1,2]) / K[1,1],
                        torch.ones_like(i)], -1)  # (H, W, 3)
    # Rotate ray directions from camera frame to the world frame
    rays_d = dirs.to(torch.float32) @ c2w[:3, :3].T.to(torch.float32) # (H, W, 3)
    # The origin of all rays is the camera origin in world coordinates
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # (H, W, 3 )
    return rays_o, rays_d


# # Example usage:

# W, H, focal = 3, 3, 2
     
# K = np.array([[focal, 0, W//2],
#                   [0, focal, H//2],
#                   [0, 0, 1]], dtype=np.float32)
     
# c2w = np.array([[ 1.0,  0.0,  0.0,  0.0],
#                     [ 0.0,  1.0,  0.0,  0.0],
#                     [ 0.0,  0.0,  1.0,  1.0]])

# K = torch.from_numpy(K)
# c2w = torch.from_numpy(c2w)
 
# rays_o, rays_d = get_rays(H, W, K, c2w)

# print(f"Shapes:\n\trays_o shape: {rays_o.shape}\n\trays_d shape: {rays_d.shape}")
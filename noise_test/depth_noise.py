import numpy as np
import cv2

def add_realistic_depth_noise(
    depth,
    max_depth=10.0,
    alpha=0.002,
    dropout_base=0.01,
    dropout_far=0.08,
    corr_scale=8,
    edge_dropout=0.15,
    edge_threshold=0.3,
):
    """
    depth: np.ndarray, shape [H, W], unit: meters
    invalid depth is assumed to be 0
    """
    depth = depth.astype(np.float32).copy()
    H, W = depth.shape

    valid = (depth > 0.0) & (depth < max_depth)

    # 1. depth-dependent Gaussian noise
    sigma = alpha * depth**2

    # 2. spatially correlated noise
    h_small = max(1, H // corr_scale)
    w_small = max(1, W // corr_scale)
    small_noise = np.random.randn(h_small, w_small).astype(np.float32)
    corr_noise = cv2.resize(small_noise, (W, H), interpolation=cv2.INTER_LINEAR)

    depth_noisy = depth.copy()
    depth_noisy[valid] += corr_noise[valid] * sigma[valid]

    # 3. independent small pixel noise, optional
    pixel_noise = np.random.randn(H, W).astype(np.float32) * (0.25 * sigma)
    depth_noisy[valid] += pixel_noise[valid]

    # 4. distance-dependent dropout
    normalized_depth = np.clip(depth / max_depth, 0.0, 1.0)
    drop_prob = dropout_base + dropout_far * normalized_depth**2
    dropout_mask = (np.random.rand(H, W) < drop_prob) & valid
    depth_noisy[dropout_mask] = 0.0

    # 5. edge dropout / flying pixel approximation
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx**2 + gy**2)
    edge_mask = (edge > edge_threshold) & valid

    edge_drop_mask = edge_mask & (np.random.rand(H, W) < edge_dropout)
    depth_noisy[edge_drop_mask] = 0.0

    # 6. clip
    depth_noisy = np.clip(depth_noisy, 0.0, max_depth)

    return depth_noisy
#!/usr/bin/env python3
import torch
import torch.nn.functional as F

def test_sampling():
    # 1. Create a dummy pyramid structure mocking the saved .pt file
    mock_dict = {
        "pyramid": [torch.ones(1, c, 32, 32, 32) for c in [48, 96, 192, 384, 768]],
        "volume_shape": (223, 193, 210)
    }

    D, H, W = mock_dict["volume_shape"]

    # 2. Define test coordinates: Origin, Center, Boundary
    xyz_phys = torch.tensor([
        [0.0, 0.0, 0.0],
        [D/2, H/2, W/2],
        [D-1.0, H-1.0, W-1.0]
    ])
    N = xyz_phys.shape[0]

    # 3. Normalize logic (matching train_inr.py)
    xyz_norm = torch.zeros_like(xyz_phys)
    xyz_norm[:, 0] = (xyz_phys[:, 0] / (D - 1)) * 2 - 1
    xyz_norm[:, 1] = (xyz_phys[:, 1] / (H - 1)) * 2 - 1
    xyz_norm[:, 2] = (xyz_phys[:, 2] / (W - 1)) * 2 - 1

    grid = torch.stack([xyz_norm[:, 2], xyz_norm[:, 1], xyz_norm[:, 0]], dim=-1).view(1, 1, 1, N, 3)

    print("Testing Grid Sample Core Logic...")
    for i, feature_grid in enumerate(mock_dict["pyramid"]):
        sampled = F.grid_sample(feature_grid, grid, mode='bilinear', align_corners=True)
        sampled = sampled.squeeze(0).squeeze(1).squeeze(1).permute(1, 0)

        # Check for NaNs or zeros at boundaries
        assert not torch.isnan(sampled).any(), f"NaN detected at scale {i}!"
        assert sampled.sum() > 0, f"All zeros detected at scale {i} boundary!"
        print(f"Scale {i} (Channels {sampled.shape[-1]}) passed. Sampled shape: {sampled.shape}")

    print("All spatial coordinate mapping tests passed successfully!")

if __name__ == "__main__":
    test_sampling()
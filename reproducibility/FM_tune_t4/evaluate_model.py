#!/usr/bin/env python3
"""
Model Performance Evaluation for FM_tune_t4 (MAE + Spatial STRD)
Computes per-channel metrics: MSE, PSNR, SSIM, MAE on all 16 samples.
Generates a summary table and saves as JSON + CSV.
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import SSIMLoss

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from train_lora_autoencoder import OpticalPropertyDataset, MAEDomainAutoencoder
import repro_config as RC

# ==========================================
# Paths
# ==========================================
CHECKPOINT_DIR = RC.FM_SAVE_DIR
DATA_DIR = RC.FM_TRAINING_DIR
MODEL_CKPT = RC.VISTA3D_CKPT
OUTPUT_DIR = os.path.join(RC.RESULTS_DIR, "fm_encoder")

CHANNEL_NAMES = [
    r'$\mu_a$ (Absorption)',
    r'$\mu_s$ (Scattering)',
    r'$g$ (Anisotropy Factor)',
    r'$n$ (Refractive Index)',
]
METRIC_NAMES = ['MSE', 'MAE', 'PSNR (dB)', 'SSIM']


def compute_metrics_per_channel(input_np, recon_np, ssim_fn, device):
    """Compute MSE, MAE, PSNR, SSIM for each channel."""
    results = {'MSE': [], 'MAE': [], 'PSNR (dB)': [], 'SSIM': []}

    for ch in range(4):
        inp_ch = torch.from_numpy(input_np[ch]).unsqueeze(0).unsqueeze(0).to(device)
        recon_ch = torch.from_numpy(recon_np[ch]).unsqueeze(0).unsqueeze(0).to(device)

        mse = float(((recon_np[ch] - input_np[ch]) ** 2).mean())
        mae = float(np.abs(recon_np[ch] - input_np[ch]).mean())
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
        ssim_val = float(ssim_fn(recon_ch, inp_ch).item())

        results['MSE'].append(mse)
        results['MAE'].append(mae)
        results['PSNR (dB)'].append(psnr)
        results['SSIM'].append(ssim_val)

    return results


def compute_global_metrics(input_np, recon_np, ssim_fn, device):
    """Compute global (volume-level) metrics across all channels."""
    diff = recon_np - input_np
    mse = float((diff ** 2).mean())
    mae = float(np.abs(diff).mean())
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0

    inp_t = torch.from_numpy(input_np).unsqueeze(0).to(device)
    recon_t = torch.from_numpy(recon_np).unsqueeze(0).to(device)
    ssim_val = float(ssim_fn(recon_t, inp_t).item())

    return {'MSE': mse, 'MAE': mae, 'PSNR (dB)': psnr, 'SSIM': ssim_val}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    dataset = OpticalPropertyDataset(DATA_DIR, crop_size=(64, 64, 64))
    model = MAEDomainAutoencoder(
        encoder_ckpt_path=MODEL_CKPT,
        lora_rank=32, lora_alpha=64,
        patch_size=(8, 8, 8), mask_ratio=0.75,
        strd_num_heads=8, strd_L_s=1,
    ).to(device)

    # Load the full checkpoint (decoder + STRD + patch-masker)
    dec_path = os.path.join(CHECKPOINT_DIR, "decoder.pt")
    dec_ckpt = torch.load(dec_path, map_location='cpu')
    model.decoder.load_state_dict(dec_ckpt['decoder_state_dict'], strict=True)
    if 'strd_attention_state' in dec_ckpt:
        model.strd_attention.load_state_dict(dec_ckpt['strd_attention_state'], strict=True)
    if 'patch_masker_state' in dec_ckpt:
        model.patch_masker.load_state_dict(dec_ckpt['patch_masker_state'], strict=True)
    print(f"Decoder + STRD + patch-masker loaded from {dec_path}\n")

    model.eval()
    ssim_fn = SSIMLoss(spatial_dims=3)

    all_sample_metrics = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            input_raw = dataset[idx]
            input_t = input_raw.unsqueeze(0).to(device)
            # MAEDomainAutoencoder returns (recon, patch_mask, orig_patches, grid_shape)
            recon_t, _patch_mask, _orig_patches, _grid = model(input_t)
            recon_np = recon_t.cpu().squeeze(0).numpy()
            input_np = input_raw.numpy()

            ch_metrics = compute_metrics_per_channel(input_np, recon_np, ssim_fn, device)
            global_metrics = compute_global_metrics(input_np, recon_np, ssim_fn, device)

            all_sample_metrics.append({
                'sample': dataset.file_list[idx],
                'channel': ch_metrics,
                'global': global_metrics,
            })

    n_samples = len(all_sample_metrics)
    n_channels = 4

    # Per-channel averages
    ch_avg = {name: [0.0] * n_channels for name in METRIC_NAMES}
    for name in METRIC_NAMES:
        for ch in range(n_channels):
            ch_avg[name][ch] = np.mean([all_sample_metrics[s]['channel'][name][ch] for s in range(n_samples)])

    # Global averages
    global_avg = {}
    for name in METRIC_NAMES:
        global_avg[name] = np.mean([all_sample_metrics[s]['global'][name] for s in range(n_samples)])

    # ==========================================
    # Print summary table
    # ==========================================
    print("=" * 80)
    print("FM_tune_t4 (MAE + Spatial STRD) — Model Performance Evaluation")
    print("=" * 80)
    print(f"\nPer-Channel Metrics (mean across {n_samples} samples):\n")
    header = f"{'Channel':<30}" + "".join(f"{name:>14}" for name in METRIC_NAMES)
    print(header)
    print("-" * 80)
    for ch in range(n_channels):
        row = f"{CHANNEL_NAMES[ch]:<30}"
        for name in METRIC_NAMES:
            row += f"{ch_avg[name][ch]:>14.6f}"
        print(row)

    print("-" * 80)
    row = f"{'Global (all channels)':<30}"
    for name in METRIC_NAMES:
        row += f"{global_avg[name]:>14.6f}"
    print(row)
    print()

    # Per-sample breakdown
    print(f"\nPer-Sample MSE:\n")
    sample_header = f"{'Sample':<45}" + "".join(f"{'Ch'+str(c):>10}" for c in range(n_channels))
    print(sample_header)
    for s in range(n_samples):
        row = f"{all_sample_metrics[s]['sample']:<45}"
        for c in range(n_channels):
            row += f"{all_sample_metrics[s]['channel']['MSE'][c]:>10.6f}"
        print(row)
    print()

    # ==========================================
    # Save JSON report
    # ==========================================
    per_channel_avg_dict = {}
    for ch in range(n_channels):
        per_channel_avg_dict[CHANNEL_NAMES[ch]] = {name: float(ch_avg[name][ch]) for name in METRIC_NAMES}

    per_sample_list = []
    for s in range(n_samples):
        per_sample_list.append({
            'sample': all_sample_metrics[s]['sample'],
            'channel_metrics': {name: all_sample_metrics[s]['channel'][name] for name in METRIC_NAMES},
            'global_metrics': all_sample_metrics[s]['global'],
        })

    report = {
        'model': 'FM_tune_t4 (MAE + Spatial STRD)',
        'n_samples': n_samples,
        'per_channel_avg': per_channel_avg_dict,
        'global_avg': {name: float(global_avg[name]) for name in METRIC_NAMES},
        'per_sample': per_sample_list,
    }

    json_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[Save] JSON report → {json_path}")

    # ==========================================
    # Save CSV table
    # ==========================================
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_per_channel.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Channel"] + METRIC_NAMES)
        for ch in range(n_channels):
            row_vals = [CHANNEL_NAMES[ch]]
            for name in METRIC_NAMES:
                row_vals.append(f"{ch_avg[name][ch]:.6f}")
            writer.writerow(row_vals)
        global_row = ["Global"]
        for name in METRIC_NAMES:
            global_row.append(f"{global_avg[name]:.6f}")
        writer.writerow(global_row)
    print(f"[Save] CSV table → {csv_path}")

    # ==========================================
    # Save text summary
    # ==========================================
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("FM_tune_t4 (MAE + Spatial STRD) — Model Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-Channel Averages:\n")
        for ch in range(n_channels):
            f.write(f"  {CHANNEL_NAMES[ch]}:\n")
            for name in METRIC_NAMES:
                f.write(f"    {name}: {ch_avg[name][ch]:.6f}\n")
        f.write("\nGlobal Averages:\n")
        for name in METRIC_NAMES:
            f.write(f"  {name}: {global_avg[name]:.6f}\n")
    print(f"[Save] Text summary → {summary_path}")

    print("\n[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()

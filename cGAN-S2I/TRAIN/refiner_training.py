# Standard library imports
import os
import sys
import csv

# Image processing and metrics
import cv2
import yaml
import lpips
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist
from sklearn.utils import resample

# PyTorch core
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torch.utils.tensorboard import SummaryWriter

# Evaluation metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Project-specific modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UTILS.loading_config import load_config
from UTILS.REFINE_binmask_realimage_genimage_loading import MaskImageDataset
from REFINER.def_ref_UNET_COORDATT import UNetBackgroundRefiner  # Using U-Net with CoordAttention

# Utility functions

def denorm_to_uint8(t):
    # Converts a tensor from [-1, 1] to uint8 [0, 255]
    t = (t.clamp(-1, 1) + 1) * 127.5
    return t.to(torch.uint8)

def total_variation_loss(img):
    # Encourages spatial smoothness by penalising abrupt pixel changes
    return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

def compute_ssim_psnr(fake_imgs, real_imgs):
    # Calculates SSIM and PSNR for a batch of images
    ssim_total = 0.0
    psnr_total = 0.0
    n = fake_imgs.size(0)
    for i in range(n):
        fake_np = np.array(to_pil_image(fake_imgs[i].cpu().clamp(0, 1)))
        real_np = np.array(to_pil_image(real_imgs[i].cpu().clamp(0, 1)))
        ssim = ssim_metric(real_np, fake_np, channel_axis=2, data_range=255)
        psnr = psnr_metric(real_np, fake_np, data_range=255)
        ssim_total += ssim
        psnr_total += psnr
    return ssim_total / n, psnr_total / n

def compute_ci(metric_list, confidence=0.95):
    # Computes confidence interval (CI) around the mean
    arr = np.array(metric_list)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, (np.nan, np.nan)
    mean = np.mean(arr)
    lower = np.percentile(arr, (1 - confidence) / 2 * 100)
    upper = np.percentile(arr, (1 + confidence) / 2 * 100)
    return mean, (lower, upper)

def denorm(img):
    return (img.clamp(-1, 1) + 1) / 2

def save_np_as_img(np_img, path):
    # Normalises and saves numpy array as image
    norm = cv2.normalize(np_img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(path, norm.astype(np.uint8))

def canny_edges(tensor):
    # Computes Canny edges from tensor
    gray = torch.mean(denorm(tensor), dim=1, keepdim=True).squeeze().cpu().numpy() * 255
    return [cv2.Canny(im.astype(np.uint8), 10, 20) for im in gray]

def dilate_edges(edges, dilation_size=3):
    # Dilates binary edge maps for IoU/Dice
    kernel = np.ones((dilation_size, dilation_size), dtype=np.uint8)
    return [binary_dilation(edge > 0, structure=kernel) for edge in edges]

def chamfer_distance(edge1, edge2):
    # Chamfer distance between two binary edge maps
    edge1_pts = np.column_stack(np.nonzero(edge1))
    edge2_pts = np.column_stack(np.nonzero(edge2))
    if len(edge1_pts) == 0 or len(edge2_pts) == 0:
        return np.nan
    dists_1_to_2 = cdist(edge1_pts, edge2_pts).min(axis=1)
    dists_2_to_1 = cdist(edge2_pts, edge1_pts).min(axis=1)
    return (dists_1_to_2.mean() + dists_2_to_1.mean()) / 2

def edge_metrics_dilated_and_chamfer(edges1, edges2):
    # Computes IoU, Dice, and Chamfer metrics on edge maps
    eps = 1e-6
    ious, dices, chamfers = [], [], []
    edges1_dilated = dilate_edges(edges1)
    edges2_dilated = dilate_edges(edges2)
    for e1, e2, ed1, ed2 in zip(edges1, edges2, edges1_dilated, edges2_dilated):
        inter = np.logical_and(ed1, ed2).sum()
        union = np.logical_or(ed1, ed2).sum()
        iou = inter / (union + eps)
        dice = 2 * inter / (ed1.sum() + ed2.sum() + eps)
        chamf = chamfer_distance(e1, e2)
        ious.append(iou)
        dices.append(dice)
        chamfers.append(chamf)
    return ious, dices, chamfers

def compute_gradient_map(tensor):
    # Computes gradient magnitude map using Sobel
    tensor = denorm(tensor).cpu().numpy()
    grad_maps = []
    for img in tensor:
        gray = np.mean(img, axis=0)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_maps.append(grad_mag)
    return grad_maps

def crop_by_rois(tensor_batch, rois_batch):
    # Crops each image in a batch using specified ROIs
    crops = []
    for i, img in enumerate(tensor_batch):
        for (x1, y1), (x2, y2) in rois_batch[i]:
            crops.append(img[:, y1:y2, x1:x2])
    return crops

def load_rois(yaml_path):
    # Loads region-of-interest annotations from YAML
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    rois = data.get("rois", [])
    return [[r] if isinstance(r[0][0], int) else r for r in rois]

def bootstrap_ci(data, n_bootstraps=300, ci=95):
    # Computes mean and confidence interval via bootstrapping
    data = [d for d in data if np.isfinite(d)]
    stats = [np.mean(resample(data)) for _ in range(n_bootstraps)]
    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper

def report_all_metrics(name_prefix, metrics_dict, output_file):
    # Reports all metrics to both console and CSV
    print(f"\n===== {name_prefix} Results (with 95% CI) =====")
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for key, values in metrics_dict.items():
            mean, low, high = bootstrap_ci(values)
            print(f"{key}: {mean:.4f} (95% CI: {low:.4f}–{high:.4f})")
            writer.writerow([f"{name_prefix}_{key}", f"{mean:.4f}", f"{low:.4f}", f"{high:.4f}"])

# Main training + evaluation loop
def train_and_evaluate_refiner(train_loader, val_loader, test_loader, roi_list, save_dir, log_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetBackgroundRefiner(in_channels=3, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    writer_tb = SummaryWriter(log_dir=log_dir)

    num_epochs = 20
    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for gen_img, real_img in tqdm(train_loader):
            gen_img, real_img = gen_img.to(device), real_img.to(device)
            out = model(gen_img)

            # Compute losses
            loss_l1 = criterion(out, real_img)
            loss_lpips_val = lpips_loss(out, real_img).mean()
            loss_tv = total_variation_loss(out)
            loss = loss_l1 + 5 * loss_lpips_val + 5 * loss_tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TensorBoard logging
            writer_tb.add_scalar("Train/L1", loss_l1.item(), global_step)
            writer_tb.add_scalar("Train/LPIPS", loss_lpips_val.item(), global_step)
            writer_tb.add_scalar("Train/TV", loss_tv.item(), global_step)
            writer_tb.add_scalar("Train/Total", loss.item(), global_step)
            global_step += 1

        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

    # Save trained model
    os.makedirs(save_dir, exist_ok=True)
    final_model_path = os.path.join(save_dir, "refiner_final.pth")
    if not os.path.exists(final_model_path):
        torch.save(model.state_dict(), final_model_path)

    # Evaluation loop (global + ROI metrics)
    model.eval()
    fid = FrechetInceptionDistance(feature=2048).to(device)
    metrics_global = {k: [] for k in ['LPIPS', 'Laplacian Var Diff', 'Gradient L1 Loss', 'PSNR', 'SSIM', 'IoU', 'Dice', 'Chamfer']}
    metrics_roi = {k: [] for k in ['IoU', 'Dice', 'Chamfer', 'Grad L1', 'LPIPS', 'PSNR', 'SSIM', 'Laplacian Var Diff']}

    for i, (gen_img, real_img) in enumerate(tqdm(test_loader)):
        gen_img, real_img = gen_img.to(device), real_img.to(device)
        refined = model(gen_img)

        # Compute global image metrics
        metrics_global['LPIPS'].extend(lpips_loss(refined, real_img).squeeze().cpu().numpy().tolist())

        fake_np = denorm_to_uint8(refined).permute(0, 2, 3, 1).cpu().numpy()
        real_np = denorm_to_uint8(real_img).permute(0, 2, 3, 1).cpu().numpy()
        for f, r in zip(fake_np, real_np):
            metrics_global['PSNR'].append(psnr_metric(r, f, data_range=255))
            metrics_global['SSIM'].append(ssim_metric(r, f, channel_axis=-1, data_range=255))
            metrics_global['Laplacian Var Diff'].append(abs(cv2.Laplacian(f.astype(np.float64), cv2.CV_64F).var() -
                                                            cv2.Laplacian(r.astype(np.float64), cv2.CV_64F).var()))

        edges_f = canny_edges(refined)
        edges_r = canny_edges(real_img)
        ious, dices, chamfers = edge_metrics_dilated_and_chamfer(edges_f, edges_r)
        metrics_global['IoU'].extend(ious)
        metrics_global['Dice'].extend(dices)
        metrics_global['Chamfer'].extend(chamfers)

        grad_f = compute_gradient_map(refined)
        grad_r = compute_gradient_map(real_img)
        for gf, gr in zip(grad_f, grad_r):
            metrics_global['Gradient L1 Loss'].append(np.abs(gf - gr).mean())

        # ROI-based metrics (if provided)
        if roi_list:
            roi_batch = roi_list[i]
            refined_crops = crop_by_rois(refined, [roi_batch])
            real_crops = crop_by_rois(real_img, [roi_batch])
            for fc, rc in zip(refined_crops, real_crops):
                f_np = denorm_to_uint8(fc.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
                r_np = denorm_to_uint8(rc.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
                edge_f = canny_edges(fc.unsqueeze(0))[0]
                edge_r = canny_edges(rc.unsqueeze(0))[0]
                iou, dice, chamf = edge_metrics_dilated_and_chamfer([edge_f], [edge_r])
                metrics_roi['IoU'].append(iou[0])
                metrics_roi['Dice'].append(dice[0])
                metrics_roi['Chamfer'].append(chamf[0])
                metrics_roi['Grad L1'].append(np.abs(compute_gradient_map(fc.unsqueeze(0))[0] - compute_gradient_map(rc.unsqueeze(0))[0]).mean())
                metrics_roi['LPIPS'].append(lpips_loss(fc.unsqueeze(0), rc.unsqueeze(0)).item())
                metrics_roi['PSNR'].append(psnr_metric(r_np, f_np, data_range=255))
                metrics_roi['SSIM'].append(ssim_metric(r_np, f_np, channel_axis=-1, data_range=255))
                metrics_roi['Laplacian Var Diff'].append(abs(cv2.Laplacian(np.mean(f_np, axis=2), cv2.CV_64F).var() -
                                                             cv2.Laplacian(np.mean(r_np, axis=2), cv2.CV_64F).var()))

        # Update FID score
        fid.update(denorm_to_uint8(real_img), real=True)
        fid.update(denorm_to_uint8(refined), real=False)

    # Write metrics to CSV
    fid_score = fid.compute().item()
    csv_path = os.path.join(save_dir, "combined_results_summary.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Mean", "CI Lower", "CI Upper"])
    report_all_metrics("Global", metrics_global, csv_path)
    report_all_metrics("ROI", metrics_roi, csv_path)
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["FID", f"{fid_score:.2f}", "", ""])

    writer_tb.close()
    print("Training + Evaluation complete. Metrics saved.")


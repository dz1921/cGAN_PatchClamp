import sys
import os
# Add parent directory to the path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard libraries and third-party packages
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
import csv
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation
from sklearn.utils import resample
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Custom imports
from UTILS.rightbinmaks_realimage_loading import MaskImageDataset
from GENERATOR.resnet_def_gen import ResNetGenerator
from GENERATOR.gen_def_RESNET_SPADE import SPADEResNetGenerator
from GENERATOR.gen_def_UNETPP import UNetPPGenerator
from GENERATOR.gen_def_RESNET_CBAM_SmoothUPsampling import CBAMResNetGenerator
from GENERATOR.generator_def_RGB_BIN import UNetGenerator
from GENERATOR.gen_def_UNET_CBAM import UNetCBAMGenerator
from GENERATOR.gen_def_UNET_SPADE_CBAM import SPADECBAMUNetGenerator

# Compute mean and confidence interval for a list of metric values
def compute_ci(metric_list, confidence=0.95):
    arr = np.array(metric_list)
    arr = arr[~np.isnan(arr)]  # Remove NaN values
    if len(arr) == 0:
        return np.nan, (np.nan, np.nan)
    mean = np.mean(arr)
    lower = np.percentile(arr, (1 - confidence) / 2 * 100)
    upper = np.percentile(arr, (1 + confidence) / 2 * 100)
    return mean, (lower, upper)

# Convert image tensor from [-1, 1] to [0, 1]
def denorm(img):
    return (img.clamp(-1, 1) + 1) / 2

# Convert image tensor from [-1, 1] to [0, 255] uint8 for saving
def denorm_to_uint8(img):
    return (img.clamp(-1, 1) * 127.5 + 127.5).to(torch.uint8)

# Save a numpy array as an image using OpenCV
def save_np_as_img(np_img, path):
    norm = cv2.normalize(np_img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(path, norm.astype(np.uint8))

# Apply Canny edge detection to batch of image tensors
def canny_edges(tensor):
    gray = torch.mean(denorm(tensor), dim=1, keepdim=True).squeeze().cpu().numpy() * 255
    return [cv2.Canny(im.astype(np.uint8), 10, 20) for im in gray]

# Apply binary dilation to a list of edge maps
def dilate_edges(edges, dilation_size=3):
    kernel = np.ones((dilation_size, dilation_size), dtype=np.uint8)
    return [binary_dilation(edge > 0, structure=kernel) for edge in edges]

# Compute average Chamfer distance between two edge maps
def chamfer_distance(edge1, edge2):
    edge1_pts = np.column_stack(np.nonzero(edge1))
    edge2_pts = np.column_stack(np.nonzero(edge2))
    if len(edge1_pts) == 0 or len(edge2_pts) == 0:
        return np.nan
    dists_1_to_2 = cdist(edge1_pts, edge2_pts).min(axis=1)
    dists_2_to_1 = cdist(edge2_pts, edge1_pts).min(axis=1)
    return (dists_1_to_2.mean() + dists_2_to_1.mean()) / 2

# Compute edge-based similarity metrics: IoU, Dice, Chamfer
def edge_metrics_dilated_and_chamfer(edges1, edges2):
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

# Compute gradient magnitude map using Sobel filter
def compute_gradient_map(tensor):
    tensor = denorm(tensor).cpu().numpy()
    grad_maps = []
    for img in tensor:
        gray = np.mean(img, axis=0)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_maps.append(grad_mag)
    return grad_maps

# Crop ROIs (regions of interest) from image batch based on YAML file
def crop_by_rois(tensor_batch, rois_batch):
    crops = []
    for i, img in enumerate(tensor_batch):
        for (x1, y1), (x2, y2) in rois_batch[i]:
            crops.append(img[:, y1:y2, x1:x2])
    return crops

# Load list of ROIs from a YAML file
def load_rois(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    rois = data.get("rois", [])
    return [[r] if isinstance(r[0][0], int) else r for r in rois]

# Bootstrap confidence interval for a given list of metric values
def bootstrap_ci(data, n_bootstraps=300, ci=95):
    data = [d for d in data if np.isfinite(d)]
    stats = []
    for _ in range(n_bootstraps):
        sample = resample(data)
        stats.append(np.mean(sample))
    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper

# Print and save all metrics to a CSV file
def report_all_metrics(name_prefix, metrics_dict, output_file):
    print(f"\n===== {name_prefix} Results (with 95% CI) =====")
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for key, values in metrics_dict.items():
            mean, low, high = bootstrap_ci(values)
            print(f"{key}: {mean:.4f} (95% CI: {low:.4f}–{high:.4f})")
            writer.writerow([f"{name_prefix}_{key}", f"{mean:.4f}", f"{low:.4f}", f"{high:.4f}"])


# Evaluate a trained generator model using global and ROI-based image similarity metrics

def test_generator(model_path, test_root, mask_subfolder, image_subfolder, roi_yaml_path,
                   batch_size=4, save_imgs=True, save_dir="EVAL_RESULTS"):

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # Initialise dictionary for region-based metrics
    metrics_roi = {
        'IoU': [], 'Dice': [], 'Chamfer': [], 'Grad L1': [], 'LPIPS': [], 'PSNR': [], 'SSIM': [], 'Laplacian Var Diff': []
    }

    # Load test dataset and ROI annotations
    dataset = MaskImageDataset(test_root, mask_subfolder=mask_subfolder, image_subfolder=image_subfolder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    roi_list = load_rois(roi_yaml_path)

    # Instantiate the generator model and load weights
    generator = SPADECBAMUNetGenerator(in_channels=1, out_channels=3).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Define perceptual and FID metrics
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Initialise dictionary for global image similarity metrics
    metrics_global = {k: [] for k in ['LPIPS', 'Laplacian Var Diff', 'Gradient L1 Loss', 'PSNR', 'SSIM', 'IoU', 'Dice', 'Chamfer']}

    # Create output folders
    gen_dir = os.path.join(save_dir, "generated")
    edge_dir = os.path.join(save_dir, "edges")
    grad_dir = os.path.join(save_dir, "gradients")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)
    os.makedirs(grad_dir, exist_ok=True)

    with torch.no_grad():
        for i, (mask, real) in enumerate(tqdm(loader)):
            mask, real = mask.to(device), real.to(device)
            fake = generator(mask, mask)  # Generate fake image conditioned on mask

            # Global LPIPS
            metrics_global['LPIPS'].extend(lpips_loss(fake, real).squeeze().cpu().numpy().tolist())

            # Global Laplacian variance difference
            fake_gray = torch.mean(denorm(fake), dim=1).cpu().numpy()
            real_gray = torch.mean(denorm(real), dim=1).cpu().numpy()
            for f, r in zip(fake_gray, real_gray):
                metrics_global['Laplacian Var Diff'].append(
                    abs(cv2.Laplacian(f.astype(np.float64), cv2.CV_64F).var() -
                        cv2.Laplacian(r.astype(np.float64), cv2.CV_64F).var())
                )

            # Global edge-based metrics
            canny_f = canny_edges(fake)
            canny_r = canny_edges(real)
            ious, dices, chamfers = edge_metrics_dilated_and_chamfer(canny_f, canny_r)
            metrics_global['IoU'].extend(ious)
            metrics_global['Dice'].extend(dices)
            metrics_global['Chamfer'].extend(chamfers)

            # Global gradient map L1 loss
            grad_f = compute_gradient_map(fake)
            grad_r = compute_gradient_map(real)
            for j, (gf, gr) in enumerate(zip(grad_f, grad_r)):
                idx = i * batch_size + j
                save_np_as_img(gr, os.path.join(grad_dir, f"real_grad_{idx:04d}.png"))
                save_np_as_img(gf, os.path.join(grad_dir, f"fake_grad_{idx:04d}.png"))
                metrics_global['Gradient L1 Loss'].append(np.abs(gf - gr).mean())

            # Global PSNR and SSIM
            fake_np = denorm_to_uint8(fake).permute(0, 2, 3, 1).cpu().numpy()
            real_np = denorm_to_uint8(real).permute(0, 2, 3, 1).cpu().numpy()
            for f, r in zip(fake_np, real_np):
                metrics_global['PSNR'].append(psnr_metric(r, f, data_range=255))
                metrics_global['SSIM'].append(ssim_metric(r, f, channel_axis=-1, data_range=255))

            # Region of interest (ROI) analysis
            roi_batch = roi_list[i * batch_size:(i + 1) * batch_size]
            fake_crops = crop_by_rois(fake, roi_batch)
            real_crops = crop_by_rois(real, roi_batch)

            if fake_crops and real_crops:
                for fake_crop, real_crop in zip(fake_crops, real_crops):
                    h, w = fake_crop.shape[1], fake_crop.shape[2]

                    # ROI edge metrics
                    canny_f_roi = canny_edges(fake_crop.unsqueeze(0))[0]
                    canny_r_roi = canny_edges(real_crop.unsqueeze(0))[0]

                    if np.sum(canny_f_roi) == 0 or np.sum(canny_r_roi) == 0:
                        print("[SKIP] Empty edge map in ROI")
                        metrics_roi['IoU'].append(np.nan)
                        metrics_roi['Dice'].append(np.nan)
                        metrics_roi['Chamfer'].append(np.nan)
                    else:
                        iou, dice, chamfer = edge_metrics_dilated_and_chamfer([canny_f_roi], [canny_r_roi])
                        metrics_roi['IoU'].append(iou[0])
                        metrics_roi['Dice'].append(dice[0])
                        metrics_roi['Chamfer'].append(chamfer[0])

                    # ROI gradient L1
                    grad_f_roi = compute_gradient_map(fake_crop.unsqueeze(0))[0]
                    grad_r_roi = compute_gradient_map(real_crop.unsqueeze(0))[0]
                    metrics_roi['Grad L1'].append(np.abs(grad_f_roi - grad_r_roi).mean())

                    # ROI LPIPS (only if sufficiently large)
                    if h >= 16 and w >= 16:
                        lpips_val = lpips_loss(fake_crop.unsqueeze(0).to(device), real_crop.unsqueeze(0).to(device)).item()
                        metrics_roi['LPIPS'].append(lpips_val)
                    else:
                        print(f"[SKIP] LPIPS: ROI too small ({h}, {w})")
                        metrics_roi['LPIPS'].append(np.nan)

                    # ROI PSNR + SSIM
                    f_img = denorm_to_uint8(fake_crop.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
                    r_img = denorm_to_uint8(real_crop.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
                    psnr_val = psnr_metric(r_img, f_img, data_range=255)
                    ssim_val = ssim_metric(r_img, f_img, channel_axis=-1, data_range=255)
                    metrics_roi['PSNR'].append(psnr_val)
                    metrics_roi['SSIM'].append(ssim_val)

                    # ROI Laplacian Var Diff
                    f_gray = np.mean(denorm(fake_crop.unsqueeze(0)).squeeze(0).cpu().numpy(), axis=0).astype(np.float64)
                    r_gray = np.mean(denorm(real_crop.unsqueeze(0)).squeeze(0).cpu().numpy(), axis=0).astype(np.float64)
                    lap_diff = abs(cv2.Laplacian(f_gray, cv2.CV_64F).var() - cv2.Laplacian(r_gray, cv2.CV_64F).var())
                    metrics_roi['Laplacian Var Diff'].append(lap_diff)

            # FID requires separate real/fake updates per batch
            fid.update(denorm_to_uint8(real), real=True)
            fid.update(denorm_to_uint8(fake), real=False)

            # Optionally save output images and edges
            if save_imgs:
                for j in range(fake.size(0)):
                    idx = i * batch_size + j
                    save_image(denorm(fake[j]), os.path.join(gen_dir, f"gen_{idx:04d}.png"))
                    save_np_as_img(canny_r[j], os.path.join(edge_dir, f"real_edge_{idx:04d}.png"))
                    save_np_as_img(canny_f[j], os.path.join(edge_dir, f"fake_edge_{idx:04d}.png"))

    # Compute final FID score
    fid_score = fid.compute().item()
    print(f"\nFID: {fid_score:.2f}")

    # Save and print summary metrics
    csv_path = os.path.join(save_dir, "combined_results_summary.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Mean", "CI Lower", "CI Upper"])

    report_all_metrics("Global", metrics_global, csv_path)
    report_all_metrics("ROI", metrics_roi, csv_path)

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["FID", f"{fid_score:.2f}", "", ""])

    print(f"Results saved to {csv_path}")



if __name__ == "__main__":
    test_generator(
        model_path=r"C:\\Users\\johan\\Downloads\\GENERATORS_TRAINED\\LIGHT\\UNET_CBAM_SPADE_LPIPSTV_l5_light_50\\generator_lr_0_0002.pth",
        test_root="DATA",
        mask_subfolder="LIGHT_MASKS/TEST_SET",
        image_subfolder="LIGHT_IMG/TEST_SET",
        roi_yaml_path="CONFIG/TEST_SET_LIGHT_annotated_rois.yaml",
        batch_size=4,
        save_imgs=True,
        save_dir="EVAL_RESULTS/LIGHT/UNET_CBAM_SPADE_LPIPSTV_l5_light_50_00002"
    )

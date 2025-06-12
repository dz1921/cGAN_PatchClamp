import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
from torch.utils.tensorboard import SummaryWriter

# Ensure the project root directory is on the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import local utility modules and network definitions
from UTILS.weight_init import init_weights
from UTILS.HEATMAPbinmask_realimage_loading import MaskImageDataset
from UTILS.loading_config import load_config
from DISCRIMINATOR.disc_def_MSP_HEATBIN import MultiScaleDiscriminator


# Converts normalised tensors in [-1, 1] to 8-bit images in [0, 255]
def denorm_to_uint8(t):
    t = (t.clamp(-1, 1) + 1) * 127.5
    return t.to(torch.uint8)

# Compute average SSIM and PSNR for a batch of generated vs real images
def compute_ssim_psnr(fake_imgs, real_imgs):
    ssim_total = 0.0
    psnr_total = 0.0
    n = fake_imgs.size(0)
    for i in range(n):
        fake_np = to_pil_image(fake_imgs[i].cpu().clamp(0, 1))
        real_np = to_pil_image(real_imgs[i].cpu().clamp(0, 1))
        fake_np = np.array(fake_np)
        real_np = np.array(real_np)
        ssim = ssim_metric(real_np, fake_np, channel_axis=2, data_range=255)
        psnr = psnr_metric(real_np, fake_np, data_range=255)
        ssim_total += ssim
        psnr_total += psnr
    return ssim_total / n, psnr_total / n

# Computes total variation loss to encourage smoothness in generated images
def total_variation_loss(img):
    return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

# Main training loop function
def train_lr_sweep(config_path="CONFIG/RES_SPADE_RGB_D_LR.yaml"):
    # Load hyperparameters and paths from config YAML
    config = load_config(config_path)
    data_root = config["data_root"]
    val_root = config["val_root"]
    batch_size = config["batch_size"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    lambda_l1 = config["lambda_l1"]
    lambda_tv = config.get("lambda_tv", 3)
    lambda_lpips = config.get("lambda_lpips", 5)
    lambda_gan = config.get("lambda_gan", 1.0)
    save_dir_gen = config["save_dir_gen"]
    save_dir_disc = config["save_dir_disc"]
    image_output_dir = config["image_output_dir"]
    log_csv_path = config["csv_log_path"]

    # Create output directories if they don't already exist
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    os.makedirs(save_dir_gen, exist_ok=True)
    os.makedirs(save_dir_disc, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare datasets and loaders
    train_set = MaskImageDataset(data_root, mask_subfolder="MASKS_DARK_200", heatmap_subfolder="HEATMAP", image_subfolder="DARK_IMG_200")
    val_set = MaskImageDataset(val_root, mask_subfolder="MASKS_DARK_200/VALIDATION_SET", heatmap_subfolder="HEATMAP/VALIDATION_SET", image_subfolder="DARK_IMG_200/VALIDATION_SET")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    # TensorBoard writer for visualisation
    writer_tb = SummaryWriter(log_dir="runs/CBAMRESNETheatmap_cgan")

    # Prepare CSV log for metrics
    with open(log_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["LR", "Epoch", "SSIM", "PSNR", "FID", "LPIPS"])

        learning_rates = [2e-4]  # Can be extended into a sweep
        num_epochs = 50

        for lr in learning_rates:
            print(f"\n--- Training with learning rate: {lr} ---")

            # Initialise generator and discriminator
            netG = CBAMResNetGenerator(input_nc=2, output_nc=3).to(device)
            netD = MultiScaleDiscriminator(input_nc=5).to(device)
            init_weights(netG, "normal", 0.02)
            init_weights(netD, "normal", 0.02)

            optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
            optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))

            # Define loss functions
            criterion_adv = nn.BCEWithLogitsLoss()
            criterion_l1 = nn.L1Loss()
            lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

            global_step = 0
            for epoch in range(num_epochs):
                for i, (mask, heatmap, real_img) in enumerate(train_loader):
                    mask = mask.to(device)
                    heatmap = heatmap.to(device)
                    real_img = real_img.to(device)

                    # Train Discriminator
                    with torch.no_grad():
                        fake_img = netG(segmask=mask, heatmap=heatmap)

                    real_pred = netD(mask, heatmap, real_img)
                    fake_pred = netD(mask, heatmap, fake_img)

                    d_loss = 0.0
                    for real_p, fake_p in zip(real_pred, fake_pred):
                        d_loss += 0.5 * (criterion_adv(real_p, torch.ones_like(real_p)) +
                                         criterion_adv(fake_p, torch.zeros_like(fake_p)))
                    d_loss /= len(real_pred)

                    optimizer_D.zero_grad()
                    d_loss.backward()
                    optimizer_D.step()

                    # Train Generator
                    fake_img = netG(segmask=mask, heatmap=heatmap)
                    fake_pred = netD(mask, heatmap, fake_img)

                    g_adv = sum([criterion_adv(pred, torch.ones_like(pred)) for pred in fake_pred]) / len(fake_pred)
                    g_adv *= lambda_gan
                    g_l1 = criterion_l1(fake_img, real_img) * lambda_l1
                    g_tv = total_variation_loss(fake_img) * lambda_tv
                    g_lpips = lpips_loss_fn(fake_img, real_img).mean() * lambda_lpips
                    g_loss = g_adv + g_l1 + g_tv + g_lpips

                    optimizer_G.zero_grad()
                    g_loss.backward()
                    optimizer_G.step()

                    # Log training metrics to TensorBoard
                    writer_tb.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
                    writer_tb.add_scalar("Loss/Generator", g_loss.item(), global_step)
                    writer_tb.add_scalar("Loss/G_adv", g_adv.item(), global_step)
                    writer_tb.add_scalar("Loss/G_L1", g_l1.item(), global_step)
                    writer_tb.add_scalar("Loss/G_TV", g_tv.item(), global_step)
                    writer_tb.add_scalar("Loss/G_LPIPS", g_lpips.item(), global_step)
                    global_step += 1

                # Validation every 5 epochs
                if (epoch + 1) % 5 == 0 or epoch + 1 == num_epochs:
                    netG.eval()
                    fid = FrechetInceptionDistance(feature=2048).to(device)
                    ssim_accum, psnr_accum, lpips_accum, count = 0.0, 0.0, 0.0, 0
                    with torch.no_grad():
                        for j, (val_mask, val_heatmap, val_real) in enumerate(val_loader):
                            val_mask = val_mask.to(device)
                            val_heatmap = val_heatmap.to(device)
                            val_real = val_real.to(device)

                            val_fake = netG(segmask=val_mask, heatmap=val_heatmap)

                            # FID updates
                            fid.update(denorm_to_uint8(val_real), real=True)
                            fid.update(denorm_to_uint8(val_fake), real=False)

                            # Save preview image
                            if j == 0:
                                save_path = os.path.join(image_output_dir, f"lr_{str(lr).replace('.', '_')}_epoch_{epoch + 1}.png")
                                save_image(val_fake, save_path, normalize=True)

                            ssim_batch, psnr_batch = compute_ssim_psnr(val_fake, val_real)
                            lpips_batch = lpips_loss_fn(val_fake, val_real).mean().item()
                            ssim_accum += ssim_batch
                            psnr_accum += psnr_batch
                            lpips_accum += lpips_batch
                            count += 1

                    avg_ssim = ssim_accum / count
                    avg_psnr = psnr_accum / count
                    avg_lpips = lpips_accum / count
                    fid_score = fid.compute().item()

                    # Print and record validation metrics
                    print(f"[LR {lr}] Epoch {epoch + 1} — SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f} dB, FID: {fid_score:.2f}, LPIPS: {avg_lpips:.4f}")
                    writer.writerow([lr, epoch + 1, avg_ssim, avg_psnr, fid_score, avg_lpips])
                    netG.train()

            # Save model weights after learning rate is done
            torch.save(netG.state_dict(), os.path.join(save_dir_gen, f"generator_lr_{str(lr).replace('.', '_')}.pth"))
            torch.save(netD.state_dict(), os.path.join(save_dir_disc, f"discriminator_lr_{str(lr).replace('.', '_')}.pth"))
            print(f"Saved models for LR {lr}.")

    writer_tb.close()

# Entry point
if __name__ == "__main__":
    train_lr_sweep()

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

# Add project root to path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Local project-specific modules
from UTILS.weight_init import init_weights
from DISCRIMINATOR.discriminator_def_RGB_BIN import PatchGANDiscriminator
from UTILS.rightbinmaks_realimage_loading import MaskImageDataset
from UTILS.loading_config import load_config
from GENERATOR.gen_def_UNET_SPADE import SPADEUNetGenerator
from GENERATOR.resnet_def_gen import ResNetGenerator
from GENERATOR.gen_def_UNETPP_CBAM_SmoothUPsampling import UNetPPGenerator
from DISCRIMINATOR.disc_def_MultiScalePATCHcGAN import MultiScaleDiscriminator
from GENERATOR.generator_def_RGB_BIN import UNetGenerator
from GENERATOR.gen_def_UNET_CBAM import UNetCBAMGenerator

# Converts a tensor from [-1, 1] to uint8 [0, 255]
def denorm_to_uint8(t):
    t = (t.clamp(-1, 1) + 1) * 127.5
    return t.to(torch.uint8)

# Computes average SSIM and PSNR over a batch of images
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

# Computes total variation loss to encourage smoothness
def total_variation_loss(img):
    return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

# Main training loop for running a learning rate sweep
def train_lr_sweep(config_path="CONFIGURATION/RGB_learning_rate_tuning_2.yaml"):
    config = load_config(config_path)

    # Extract parameters from configuration file
    data_root = config["data_root"]
    val_root = config["val_root"]
    batch_size = config["batch_size"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    lambda_l1 = config.get("lambda_l1", 100)
    lambda_tv = config.get("lambda_tv", 3)
    lambda_lpips = config.get("lambda_lpips", 5)
    lambda_gan = config.get("lambda_gan", 1.0)
    save_dir_gen = config["save_dir_gen"]
    save_dir_disc = config["save_dir_disc"]
    image_output_dir = config["image_output_dir"]
    log_csv_path = config["csv_log_path"]

    # Create necessary output directories
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    os.makedirs(save_dir_gen, exist_ok=True)
    os.makedirs(save_dir_disc, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    # Set up device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TensorBoard writer for visualising training progress
    writer_tb = SummaryWriter(log_dir="runs/UNETCBAM_LPIPSTV_RGB_DARK_l5")

    # Load training and validation datasets
    train_set = MaskImageDataset(data_root, mask_subfolder="MASKS_DARK_200", image_subfolder="DARK_IMG_200")
    val_set = MaskImageDataset(val_root, mask_subfolder="MASKS_DARK_200/VALIDATION_SET", image_subfolder="DARK_IMG_200/VALIDATION_SET")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    print(f"Training on {len(train_set)} samples, validating on {len(val_set)} samples.")

    # Define the learning rates to try
    learning_rates = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
    num_epochs = 10

    # Prepare CSV file to log results
    with open(log_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["LR", "Epoch", "SSIM", "PSNR", "FID", "LPIPS"])

        for lr in learning_rates:
            print(f"\n--- Training with learning rate: {lr} ---")

            # Initialise generator and discriminator models
            netG = UNetCBAMGenerator(in_channels=1, out_channels=3).to(device)
            netD = MultiScaleDiscriminator(input_nc=4).to(device)
            init_weights(netG, "normal", 0.02)
            init_weights(netD, "normal", 0.02)

            # Define optimisers and loss functions
            optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
            optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
            criterion_adv = nn.BCEWithLogitsLoss()
            criterion_l1 = nn.L1Loss()
            lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

            global_step = 0
            for epoch in range(num_epochs):
                for i, (mask, real_img) in enumerate(train_loader):
                    mask = mask.to(device)
                    real_img = real_img.to(device)

                    # Discriminator update
                    with torch.no_grad():
                        fake_img = netG(mask)
                    real_preds = netD(mask, real_img)
                    fake_preds = netD(mask, fake_img)

                    d_loss = 0.0
                    # Iterating through the different PatchGANs
                    for real_pred, fake_pred in zip(real_preds, fake_preds):
                        real_labels = torch.ones_like(real_pred)
                        fake_labels = torch.zeros_like(fake_pred)
                        d_loss += 0.5 * (
                            criterion_adv(real_pred, real_labels).mean() +
                            criterion_adv(fake_pred, fake_labels).mean()
                        )
                    d_loss /= len(real_preds)
                    optimizer_D.zero_grad()
                    d_loss.backward()
                    optimizer_D.step()

                    # Generator update
                    fake_img = netG(mask)
                    fake_preds = netD(mask, fake_img)
                    g_adv = sum([criterion_adv(pred, torch.ones_like(pred)).mean() for pred in fake_preds]) / len(fake_preds)
                    g_adv *= lambda_gan
                    g_l1 = criterion_l1(fake_img, real_img) * lambda_l1
                    g_tv = total_variation_loss(fake_img) * lambda_tv
                    g_lpips = lpips_loss_fn(fake_img, real_img).mean() * lambda_lpips
                    g_loss = g_adv + g_l1 + g_tv + g_lpips

                    optimizer_G.zero_grad()
                    g_loss.backward()
                    optimizer_G.step()

                    # Log training losses
                    writer_tb.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
                    writer_tb.add_scalar("Loss/Generator", g_loss.item(), global_step)
                    writer_tb.add_scalar("Loss/G_adv", g_adv.item(), global_step)
                    writer_tb.add_scalar("Loss/G_L1", g_l1.item(), global_step)
                    writer_tb.add_scalar("Loss/G_TV", g_tv.item(), global_step)
                    writer_tb.add_scalar("Loss/G_LPIPS", g_lpips.item(), global_step)
                    global_step += 1

                    if (i + 1) % 20 == 0:
                        print(f"Epoch [{epoch + 1}/{num_epochs}] Step [{i + 1}/{len(train_loader)}] "
                              f"D: {d_loss.item():.4f}, G: {g_loss.item():.4f}, "
                              f"G_adv: {g_adv.item():.4f}, G_L1: {g_l1.item():.4f}, "
                              f"G_TV: {g_tv.item():.4f}, G_LPIPS: {g_lpips.item():.4f}")

                # Validation pass on selected epochs
                if (epoch + 1) in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
                    netG.eval()
                    fid = FrechetInceptionDistance(feature=2048).to(device)
                    ssim_accum, psnr_accum, lpips_accum, count = 0.0, 0.0, 0.0, 0
                    with torch.no_grad():
                        for j, (val_mask, val_real) in enumerate(val_loader):
                            val_mask = val_mask.to(device)
                            val_real = val_real.to(device)
                            val_fake = netG(val_mask)

                            fid.update(denorm_to_uint8(val_real), real=True)
                            fid.update(denorm_to_uint8(val_fake), real=False)

                            if j == 0:
                                save_path = os.path.join(
                                    image_output_dir, f"lr_{str(lr).replace('.', '_')}_epoch_{epoch + 1}.png"
                                )
                                save_image(val_fake, save_path, normalize=True)

                            ssim_batch, psnr_batch = compute_ssim_psnr(val_fake, val_real)
                            lpips_batch = lpips_loss_fn(val_fake, val_real).mean().item()
                            ssim_accum += ssim_batch
                            lpips_accum += lpips_batch
                            psnr_accum += psnr_batch
                            count += 1

                    avg_ssim = ssim_accum / count
                    avg_psnr = psnr_accum / count
                    avg_lpips = lpips_accum / count
                    fid_score = fid.compute().item()
                    print(f"[LR {lr}] Epoch {epoch + 1} — SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f} dB, "
                          f"FID: {fid_score:.2f}, LPIPS: {avg_lpips:.4f}")
                    writer.writerow([lr, epoch + 1, avg_ssim, avg_psnr, fid_score, avg_lpips])
                    netG.train()

            # Save model checkpoints after each LR sweep
            torch.save(netG.state_dict(), os.path.join(save_dir_gen, f"generator_lr_{str(lr).replace('.', '_')}.pth"))
            torch.save(netD.state_dict(), os.path.join(save_dir_disc, f"discriminator_lr_{str(lr).replace('.', '_')}.pth"))
            print(f"Saved models for LR {lr}.")

    # Close TensorBoard logger
    writer_tb.close()

if __name__ == "__main__":
    train_lr_sweep()


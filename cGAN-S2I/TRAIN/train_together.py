import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from UTILS.weight_init import init_weights
from GENERATOR.generator_def_RGB_BIN import UNetGenerator
from DISCRIMINATOR.discriminator_def_RGB_BIN import PatchGANDiscriminator
from UTILS.binmask_realimage_loading import MaskImageDataset
from UTILS.loading_config import load_config

def train_pix2pix(config_path = r"CONFIG\RGB_D_cGAN.yaml"):
    """
    data_root: Path to data folder containing images & masks.
    epochs: Number of training epochs.
    batch_size: Batch size for DataLoader.
    lr: Learning rate for both generator & discriminator.
    beta1, beta2: Adam optimizer hyperparameters.
    lambda_l1: Weight for L1 loss.
    """
    # Load hyperparameters from config file
    config = load_config(config_path)

    # Extract values from config
    data_root = config["data_root"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    lambda_l1 = config["lambda_l1"]
    save_dir_gen = config["save_dir_gen"]
    save_dir_disc = config["save_dir_disc"]

    #DEV CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #DATA LOADING
    dataset = MaskImageDataset(data_root)  #FROM THE IMPORT DATASET LINE
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataset size: {len(dataset)} images")

    #INITIALISE GEN AND DISC - if the GEN and DISC have been pre-trained they can be loaded here, but skip weight init
    netG = UNetGenerator(in_channels=1, out_channels=3).to(device)
    netD = PatchGANDiscriminator(in_channels=4).to(device)

    #INITIALISE WEIGHTS
    init_weights(netG, init_type="normal", init_gain=0.02)
    init_weights(netD, init_type="normal", init_gain=0.02)

    #OPTIMISERS AND LOSS FUNCTIONS
    # - BCEWithLogitsLoss for the discriminator (no Sigmoid in netD final layer)
    # - L1Loss for image reconstruction
    optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))

    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    #TRAINING LOOP
    print("Everything loaded succesfully. Training is about to begin.")
    for epoch in range(epochs):
        for i, (mask, real_img) in enumerate(dataloader):
            # Move data to device
            mask = mask.to(device)       # (N, 1, 256, 256)
            real_img = real_img.to(device)  # (N, 3, 256, 256)

            #TRAIN DISCRIMINATOR
            # fake image generation
            with torch.no_grad():
                fake_img = netG(mask)  # (N, 3, 256, 256)

            # predictions
            real_pred = netD(mask, real_img)      # (N, 1, 30, 30)
            fake_pred = netD(mask, fake_img)      # (N, 1, 30, 30)

            # labels
            real_labels = torch.ones_like(real_pred)
            fake_labels = torch.zeros_like(fake_pred)

            # compute D losses
            d_real_loss = criterion_adv(real_pred, real_labels)
            d_fake_loss = criterion_adv(fake_pred, fake_labels)
            d_loss = 0.5 * (d_real_loss + d_fake_loss) #instead of 0.5 something we can assign weights to the losses (0.5 is default so that both losses add to the total loss equally + normalisation)
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()


            # TRAIN GENERATOR
            fake_img = netG(mask)          # Forward pass again (this time grad is tracked)
            fake_pred = netD(mask, fake_img)  # Evaluate fake

            # try to fool D (so label is real)
            g_adv_loss = criterion_adv(fake_pred, real_labels)
            g_l1_loss = criterion_l1(fake_img, real_img) * lambda_l1
            g_loss = g_adv_loss + g_l1_loss

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            #progress checking
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{i+1}/{len(dataloader)}]")
                print(f"  D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
                print(f"  G_adv: {g_adv_loss.item():.4f}, G_l1: {g_l1_loss.item():.4f}")
            
            # Save checkpoints after each epoch
        # torch.save(netG.state_dict(), f"checkpoints/generator_epoch_{epoch+1}.pth")
        # torch.save(netD.state_dict(), f"checkpoints/discriminator_epoch_{epoch+1}.pth")
    
    #FINAL SAVE 
    save_dir_gen = r"MODELS/TRAINED_TOGETHER/GENERATORS"
    save_dir_disc = r"MODELS/TRAINED_TOGETHER/DISCRIMINATORS"
    torch.save(netG.state_dict(), os.path.join(save_dir_gen, "generator_final.pth"))
    torch.save(netD.state_dict(), os.path.join(save_dir_disc, "discriminator_final.pth"))
    print("Training complete. Models saved.")

if __name__ == "__main__":
    train_pix2pix()




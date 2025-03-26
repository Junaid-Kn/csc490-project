import torch.nn as nn
from torchvision.models import vgg16
import torch.optim as optim
import torch
import torch.nn.functional as F  # For interpolation
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Perceptual Loss using a pre-trained VGG network
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:23].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.criterion(pred_features, target_features)

# Total Variation Loss to reduce artifacts
def total_variation_loss(img):
    return torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
           torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))

def train(model, dataloader, epochs=10, device=device):
    """
    Train an image inpainting model with optimized performance.
    """
    # Warm-up learning rate scheduler
    base_lr = 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr, total_steps=len(dataloader) * epochs)

    # Loss functions
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    criterion_perceptual = PerceptualLoss()

    model.to(device)
    model.train()

    epoch_losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0

        # Progress bar for epoch
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")

        for i, (original_image, mask) in progress_bar:
            original_image = original_image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Create masked input
            masked_image = original_image * (1 - mask)

            optimizer.zero_grad()
            predicted_images = model(masked_image)

            # Refined loss function weights
            l1_loss_masked = criterion_l1(predicted_images * mask, original_image * mask)
            mse_loss_full = criterion_mse(predicted_images, original_image)
            l1_loss_unmasked = criterion_l1(predicted_images * (1 - mask), original_image * (1 - mask))
            perceptual_loss = criterion_perceptual(predicted_images, original_image)
            tv_loss = total_variation_loss(predicted_images)

            total_loss = 6 * l1_loss_masked + 2 * mse_loss_full + 0.5 * l1_loss_unmasked + 0.2 * perceptual_loss + 0.1 * tv_loss
            total_loss.backward()

            # Improved gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            scheduler.step()

            running_loss += total_loss.item()

            # Update progress bar
            progress_bar.set_postfix(loss=total_loss.item())

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.8f} | Time: {epoch_time:.2f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "unet_optimized_model.pth")

        # Stop early if loss is very low
        if avg_loss < 0.0001:
            print(f"Target loss reached at epoch {epoch+1}. Stopping training.")
            break

    # Load best model
    model.load_state_dict(torch.load("unet_optimized_model.pth"))

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig("optimized_loss_plot.png")
    plt.show()

    return model, epoch_losses
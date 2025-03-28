import torch.nn as nn
import torch

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    total_loss = 0.0
    criterion = nn.L1Loss()

    with torch.no_grad():
        for mask, target_image in test_loader:
            mask, target_image = mask.to(device), target_image.to(device)

            # Apply mask to the target image (mask out the missing areas)
            mask_applied_image = target_image * (1 - mask)

            # Get the predicted images
            predicted_images = model(mask_applied_image)

            # Compute loss only for the masked regions (missing areas)
            loss = criterion(predicted_images * mask, target_image * mask)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss (L1): {avg_loss:.4f}")
    return avg_loss

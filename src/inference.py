import torch


def run_inference(model, image, mask, device):
    model.eval()

    # Move tensors to the correct device
    image, mask = image.to(device), mask.to(device)

    # Ensure mask is grayscale (1 channel)
    mask = mask[:, 0:1, :, :]  # Keep only the first channel if there are multiple channels

    # Ensure image has 3 channels for RGB
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)  # Convert grayscale to RGB if single channel

    # Apply mask to image (set masked areas to 0)
    masked_image = image * (1 - mask)

    # Run model inference
    with torch.no_grad():
        predicted_image = model(masked_image)

    # Debug: Check model output
    print("Predicted Image Shape:", predicted_image.shape)
    print("Predicted Image Sample:", predicted_image[0, 0, 0, 0])  # Sample value

    # Fill in the missing areas
    inpainted_image = masked_image + predicted_image * mask

    return inpainted_image.cpu(), masked_image.cpu()  # Return both inpainted and masked images
import torch.nn as nn
from torchvision.models import vgg16
import torch.optim as optim
import torch
import torch.nn.functional as F  # For interpolation
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from torchvision import models
import os 
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
import copy
import subprocess
from torchmetrics.image import StructuralSimilarityIndexMeasure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, use_ssim=True):
        """
        Knowledge Distillation Loss for Image Inpainting

        Args:
        - alpha: Weight for the ground truth reconstruction loss (higher = prioritize ground truth)
        - use_ssim: Whether to use Structural Similarity (SSIM) for perceptual quality
        """
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()  # Pixel-wise MSE for hard targets
        self.use_ssim = use_ssim
        if use_ssim:
            self.ssim_loss = SSIMLoss()  # Custom SSIM loss

    def forward(self, student_outputs, teacher_outputs, target_imgs, alpha):
        """
        Args:
        - student_outputs: Student model predictions (B, 3, H, W)
        - teacher_outputs: Teacher model outputs (B, 3, H, W)
        - target_imgs: Ground truth images (B, 3, H, W)

        Returns:
        - Combined loss (MSE + optional SSIM)
        """
        # 1. Hard Target Loss: Student vs. Ground Truth
        hard_loss = self.mse_loss(student_outputs, target_imgs)

        # 2. Soft Target Loss: Student vs. Teacher
        soft_loss = self.mse_loss(student_outputs, teacher_outputs)

        # 3. Structural Similarity Loss (optional)
        ssim_loss = self.ssim_loss(student_outputs, target_imgs) if self.use_ssim else 0

        # Final Loss Combination
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss + 0.1 * ssim_loss  # Small SSIM weight

class SSIMLoss(nn.Module):
    """Structural Similarity (SSIM) Loss for image quality."""
    def __init__(self, data_range=1.0):
        """
        Args:
        - data_range: The range of the input images (e.g., 1.0 for [0, 1] normalized images).
        """
        super().__init__()
        self.data_range = data_range
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)
    

def clear_dir(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Total Variation Loss to reduce artifacts
def total_variation_loss(img):
    return torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
           torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))

def train_model_distillation(
    student,
    teacher_model_path,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    alpha=0.5,
    save_path='student_distilled_model_30.pth'
):
    student.to(device)

    best_train_loss = float('inf')
    best_model_wts = copy.deepcopy(student.state_dict())

    training_history = {
        'train_loss': [],
        'val_loss': [],
    }

    temp_inputs = os.path.join(os.getcwd(), 'temp_inputs')
    temp_outputs = os.path.join(os.getcwd(), 'teacher_outputs')
    os.makedirs(temp_inputs, exist_ok=True)
    os.makedirs(temp_outputs, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        student.train()
        train_loss = 0.0

        for batch_idx, (target_img, mask) in enumerate(train_loader):
            target_img = target_img.to(device)  # (B, 3, H, W)
            mask = mask.to(device)              # (B, 1, H, W)

            # Create masked image
            masked_image = target_img * (1 - mask)  # (B, 3, H, W)

            # Clear temporary directories
            clear_dir(temp_inputs)
            clear_dir(temp_outputs)

            # Save masked images and masks for teacher model
            for i in range(target_img.size(0)):
                img_path = os.path.join(temp_inputs, f'input_{batch_idx}_{i}.png')
                mask_path = os.path.join(temp_inputs, f'input_{batch_idx}_{i}_mask.png')
                save_image(masked_image[i], img_path)
                save_image(mask[i], mask_path)

            # Get teacher predictions
            teacher_predictions = get_teacher_predictions(temp_inputs, teacher_model_path, target_img.size(0))
            teacher_outputs = torch.stack(teacher_predictions).to(device)  # (B, 3, H, W)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                student_outputs = student(masked_image)  # (B, 3, H, W)
                loss = criterion(student_outputs, teacher_outputs, target_img, alpha)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_epoch_loss = train_loss / len(train_loader)
        scheduler.step()

        training_history['train_loss'].append(train_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")

        if train_epoch_loss < best_train_loss:
            best_train_loss = train_epoch_loss
            best_model_wts = copy.deepcopy(student.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"Saved best model to {save_path}")

    student.load_state_dict(best_model_wts)
    return student, training_history

def get_teacher_predictions(temp_inputs, model_path, batch_size):
    temp_outputs = os.path.join(os.getcwd(), 'teacher_outputs')
    cmd = (
        f"PYTHONPATH=. TORCH_HOME={os.getcwd()} python3 ../../lama/bin/predict.py "
        f"model.path={model_path} "
        f"indir={os.path.abspath(temp_inputs)} "
        f"outdir={os.path.abspath(temp_outputs)} "
        f"dataset.img_suffix=.png"
    )
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("Subprocess STDOUT:", result.stdout)
        print("Subprocess STDERR:", result.stderr)
        output_files = sorted([f for f in os.listdir(temp_outputs) if f.endswith('.png')])
        print("Output files:", output_files)
        if len(output_files) != batch_size:
            raise RuntimeError(f"Expected {batch_size} outputs, got {len(output_files)}")
        output_images = []
        transform = transforms.ToTensor()
        for filename in output_files:
            img_path = os.path.join(temp_outputs, filename)
            img = Image.open(img_path).convert('RGB')
            output_images.append(transform(img))
        # Optionally remove files after loading
        for filename in output_files:
            os.remove(os.path.join(temp_outputs, filename))
        return output_images
    except subprocess.CalledProcessError as e:
        print(f"Error executing teacher model: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error in get_teacher_predictions: {e}")
        raise
import torch
import torch.nn as nn

from UnetModel import UNetModel
from DoubleConv import DoubleConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetModel()
#model.load_state_dict(torch.load("../unet_optimized_model.pth", map_location=device))
#model.to(device)
#model.eval()
groups = 0
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Layer {name} has non-trainable parameters.")
print(f"Total: {sum(p.numel() for p in model.parameters() if p.requires_grad)} Parameters")

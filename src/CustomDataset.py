import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision as transforms


class CustomDataset(Dataset):
  def __init__(self,sample_csv, transform=None):
    self.data = pd.read_csv(sample_csv)
    self.transform = transform

  def __len__(self):
    return len(self.data)


  def __getitem__(self, idx):
    target_img_path = self.data.iloc[idx, 0]
    masked_img_path = self.data.iloc[idx, 1]

    masked_img = Image.open(masked_img_path).convert("L")
    target_img = Image.open(target_img_path).convert("RGB")

    if self.transform:
      masked_img = self.transform(masked_img)
      target_img = self.transform(target_img)

    return target_img, masked_img
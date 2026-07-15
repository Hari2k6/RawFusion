import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from thop import profile
import sys

sys.path.append("./model_zoo")
from simple_burst_model import SimpleBurstModel


# -----------------------
# Safe Image Reader
# -----------------------
def read_image_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# -----------------------
# Dataset
# -----------------------
class ValidationDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=transforms.ToTensor()):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform

        # SUPPORT TIF FILES
        valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

        self.input_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(valid_ext)
        ])

        self.gt_files = sorted([
            f for f in os.listdir(gt_dir)
            if f.lower().endswith(valid_ext)
        ])

        print(f"Found {len(self.input_files)} input images")
        print(f"Found {len(self.gt_files)} GT images")

        assert len(self.input_files) % 9 == 0, \
            f"Input images not multiple of 9. Found {len(self.input_files)}"

        assert len(self.gt_files) == len(self.input_files) // 9, \
            f"Mismatch: {len(self.gt_files)} GT vs {len(self.input_files)//9} bursts"

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):

        burst_imgs = []

        for i in range(9):
            img_path = os.path.join(
                self.input_dir,
                self.input_files[idx * 9 + i]
            )

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                raise ValueError(f"Failed to load: {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            burst_imgs.append(img)

        burst = torch.stack(burst_imgs)

        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        if gt is None:
            raise ValueError(f"Failed to load GT: {gt_path}")

        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = self.transform(gt)

        return burst, gt


# -----------------------
# Config
# -----------------------
input_dir = "validation_input"
gt_dir = "validation_gt"
output_dir = "test_img_results"
checkpoint_path = "models/best_burst_model.pth"

os.makedirs(output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# DataLoader
# -----------------------
val_dataset = ValidationDataset(input_dir, gt_dir)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# -----------------------
# Load Model
# -----------------------
model = SimpleBurstModel().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print("Model loaded successfully!")


# -----------------------
# Parameter Count
# -----------------------
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")


# -----------------------
# FLOPs
# -----------------------
dummy_input = torch.randn(1, 9, 3, 256, 256).to(device)
flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
print(f"FLOPs: {flops/1e9:.3f} GFLOPs")


# -----------------------
# Inference + PSNR
# -----------------------
psnr_total = 0.0

with torch.no_grad():
    for idx, (burst, gt) in enumerate(val_loader):

        burst = burst.to(device)
        gt = gt.to(device)

        pred = model(burst)

        pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        gt_np = gt.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        pred_np = np.clip(pred_np * 255, 0, 255).astype(np.uint8)
        gt_np = np.clip(gt_np * 255, 0, 255).astype(np.uint8)

        psnr = compare_psnr(gt_np, pred_np, data_range=255)
        psnr_total += psnr

        print(f"Scene {idx:03d} PSNR: {psnr:.4f} dB")

        out_path = os.path.join(output_dir, f"Scene-{idx:03d}-pred.png")
        cv2.imwrite(out_path, cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))

avg_psnr = psnr_total / len(val_loader)
print(f"\nAverage PSNR: {avg_psnr:.4f} dB")
print("Evaluation completed successfully!")
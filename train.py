import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
import time

from model_zoo.simple_burst_model import SimpleBurstModel

# -----------------------
# Dataset Classes
# -----------------------
class BurstDatasetAuto(Dataset):
    def __init__(self, root_dirs, transform=transforms.ToTensor()):
        self.transform = transform
        self.bursts = []  # list of 9-frame input paths
        self.gt_files = {}  # scene -> gt path

        # collect GT images
        for d in root_dirs:
            for f in os.listdir(d):
                if "-gt" in f:
                    scene = f.split("-gt")[0]
                    self.gt_files[scene] = os.path.join(d, f)

        # collect input frames and group by scene
        temp_bursts = {}
        for d in root_dirs:
            for f in os.listdir(d):
                if "-in-" in f:
                    scene = f.split("-in-")[0]
                    temp_bursts.setdefault(scene, []).append(os.path.join(d, f))

        # keep only bursts with GT and full 9 frames
        for scene, frames in temp_bursts.items():
            if scene in self.gt_files and len(frames) == 9:
                frames_sorted = sorted(frames, key=lambda x: int(x.split("-in-")[1].split(".")[0]))
                self.bursts.append((frames_sorted, self.gt_files[scene]))

    def __len__(self):
        return len(self.bursts)

    def __getitem__(self, idx):
        frames_paths, gt_path = self.bursts[idx]
        imgs = []
        for p in frames_paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        burst = torch.stack(imgs)

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        if self.transform:
            gt = self.transform(gt)

        return burst, gt

class ValidationInputDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        self.files = sorted([f for f in os.listdir(root_dir) if "-in-" in f])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files) // 9

    def __getitem__(self, idx):
        imgs = []
        for i in range(9):
            img_path = os.path.join(self.root_dir, self.files[idx*9 + i])
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs)

class ValidationGTDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        self.files = sorted([f for f in os.listdir(root_dir) if "-gt" in f])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img

# -----------------------
# Utility Functions
# -----------------------
def tensor_to_numpy(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1,2,0)
    return np.clip(img*255, 0, 255).astype(np.uint8)

def validate(model, input_loader, gt_loader, device):
    model.eval()
    psnr_total, ssim_total = 0.0, 0.0
    with torch.no_grad():
        for burst, gt in zip(input_loader, gt_loader):
            burst = burst.to(device)
            gt = gt.to(device)
            pred = model(burst)

            pred_np = tensor_to_numpy(pred.squeeze(0))
            gt_np = tensor_to_numpy(gt.squeeze(0))

            psnr_total += compare_psnr(gt_np, pred_np, data_range=255)
            ssim_total += compare_ssim(gt_np, pred_np, data_range=255, channel_axis=2)
    n = len(input_loader)
    return psnr_total/n, ssim_total/n

# -----------------------
# Main Training
# -----------------------
if __name__ == "__main__":
    # Config
    train_dirs = ["train1", "train2"]
    val_input_dir = "validation_input"
    val_gt_dir = "validation_gt"
    batch_size = 4
    num_epochs = 20
    learning_rate = 1e-4
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DataLoaders
    train_dataset = BurstDatasetAuto(train_dirs, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_input_dataset = ValidationInputDataset(val_input_dir, transform=transforms.ToTensor())
    val_gt_dataset = ValidationGTDataset(val_gt_dir, transform=transforms.ToTensor())
    val_input_loader = DataLoader(val_input_dataset, batch_size=1, shuffle=False)
    val_gt_loader = DataLoader(val_gt_dataset, batch_size=1, shuffle=False)

    # Model, Loss, Optimizer
    model = SimpleBurstModel().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop with progress bar
    best_psnr = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (burst, gt) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            burst = burst.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            output = model(burst)
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        remaining_epochs = num_epochs - (epoch + 1)
        est_remaining_time = remaining_epochs * epoch_time
        est_min, est_sec = divmod(est_remaining_time, 60)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.6f} | "
              f"Epoch Time: {epoch_time:.1f}s | Estimated Remaining: {int(est_min)}m {int(est_sec)}s")

        # Validation
        psnr, ssim = validate(model, val_input_loader, val_gt_loader, device)
        print(f"Validation PSNR: {psnr:.4f} dB | SSIM: {ssim:.4f}")

        # Save best model
        if psnr > best_psnr:
            best_psnr = psnr
            best_path = os.path.join(save_dir, "best_burst_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved BEST model: {best_path}")

    print("Training completed!")
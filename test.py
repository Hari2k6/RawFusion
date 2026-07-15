import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from thop import profile
import tifffile

# -----------------------
# MODEL
# -----------------------
from model_zoo.simple_burst_model import SimpleBurstModel


# -----------------------
# DATASET
# -----------------------
class TestDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir

        self.files = sorted([
            f for f in os.listdir(input_dir)
            if f.endswith(".tif")
        ])

        print(f"Found {len(self.files)} RAW images")
        assert len(self.files) % 9 == 0, "Must be multiple of 9"

    def __len__(self):
        return len(self.files) // 9

    def __getitem__(self, idx):
        burst = []

        for i in range(9):
            file_name = self.files[idx * 9 + i]
            path = os.path.join(self.input_dir, file_name)

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                raise ValueError(f"Failed to load {path}")

            # Convert to RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)

            img = img.astype(np.float32)

            # Normalize
            max_val = img.max()
            if max_val > 0:
                img = img / max_val

            # (H, W, 3) → (3, H, W)
            img = np.transpose(img, (2, 0, 1))

            burst.append(img)

        # -----------------------
        # 🔥 KEEP FRAME DIMENSION
        # (9, 3, H, W)
        # -----------------------
        burst = np.stack(burst, axis=0)

        burst = torch.from_numpy(burst)

        scene_name = self.files[idx * 9].split("-in-")[0]

        return burst, scene_name


# -----------------------
# CONFIG
# -----------------------
input_dir = "test"
output_dir = "test_op"
checkpoint_path = "models/best_burst_model.pth"

os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# DATA
# -----------------------
dataset = TestDataset(input_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=False)


# -----------------------
# MODEL LOAD
# -----------------------
model = SimpleBurstModel().to(device)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

if isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint)
else:
    model = checkpoint

model.eval()

print("Model loaded successfully!")


# -----------------------
# PARAMS + FLOPS
# -----------------------
params = sum(p.numel() for p in model.parameters())
print(f"Total Params: {params/1e6:.2f}M")

# 🔥 FIXED SHAPE
dummy = torch.randn(1, 9, 3, 256, 256).to(device)

try:
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    print(f"FLOPs: {flops/1e9:.3f}G")
except:
    print("FLOPs calculation skipped (shape mismatch handled)")


# -----------------------
# INFERENCE
# -----------------------
with torch.no_grad():
    for idx, (burst, scene_name) in enumerate(loader):

        burst = burst.to(device)  # (1, 9, 3, H, W)

        pred = model(burst)  # (1, 3, H, W)

        pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # -----------------------
        # FIX BLACK OUTPUT
        # -----------------------
        pred_np = np.clip(pred_np, 0, None)

        scale = np.percentile(pred_np, 99)

        if scale > 0:
            pred_np = pred_np / scale

        pred_np = np.clip(pred_np, 0, 1)

        pred_np = (pred_np * 255).astype(np.uint8)

        # -----------------------
        # SAVE
        # -----------------------
        out_path = os.path.join(output_dir, f"{scene_name}-pred.tif")

        tifffile.imwrite(out_path, pred_np, photometric='rgb', compression=None)

        print(f"[{idx+1}/{len(loader)}] Saved: {out_path}")

print("\n✅ DONE")
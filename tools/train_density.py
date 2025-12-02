# tools/train_density.py
import os, json, cv2, math, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from core.models.csrnet_lite import CSRNetLite

# -------------------------- I/O utils --------------------------

def read_points(jp: Path):
    with open(jp, "r", encoding="utf-8") as f:
        d = json.load(f)
    pts = np.array(d.get("points", []), dtype=np.float32)
    # شکل مورد انتظار: [[x,y], ...]
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return pts

# ---------------------- Density map builder --------------------

def _gaussian_kernel2d(ksize: int, sigma: float) -> np.ndarray:
    """ هسته گاوسی 2بعدی نرمالیزه (sum=1) """
    g1 = cv2.getGaussianKernel(ksize, sigma)
    g2 = g1 @ g1.T
    g2 /= (g2.sum() + 1e-8)
    return g2.astype(np.float32)

def make_density(h: int, w: int, pts: np.ndarray, sigma: float) -> np.ndarray:
    """
    Density با جمع برابر تعداد نقاط. برای هر نقطه، کرنل گاوسی را
    در پنجره‌ی بریده‌شده به نقشه اضافه می‌کند (sum-preserving).
    """
    dm = np.zeros((h, w), np.float32)
    k = int(max(3, round(sigma * 6))) | 1  # ~3*sigma در هر طرف
    G = _gaussian_kernel2d(k, sigma)

    r = k // 2
    for (x, y) in pts:
        cx = int(round(x))
        cy = int(round(y))
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        # محدوده در تصویر
        x0 = max(0, cx - r); x1 = min(w, cx + r + 1)
        y0 = max(0, cy - r); y1 = min(h, cy + r + 1)
        # محدوده در کرنل
        gx0 = r - (cx - x0); gx1 = gx0 + (x1 - x0)
        gy0 = r - (cy - y0); gy1 = gy0 + (y1 - y0)
        dm[y0:y1, x0:x1] += G[gy0:gy1, gx0:gx1]
    return dm

# --------------------------- Dataset ---------------------------

class PointDataset(Dataset):
    def __init__(self, root, long_side=1280, aug=True):
        self.root = Path(root)
        self.imgs = sorted([p for p in (self.root / "images").glob("*.*")
                            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.long = int(long_side)

        # Augmentation هم‌زمان روی تصویر و نقاط (keypoints=xy)
        if aug:
            self.aug = A.Compose(
                [
                    A.RandomBrightnessContrast(0.12, 0.12, p=0.5),
                    A.GaussNoise(var_limit=(5, 20), p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
        else:
            self.aug = None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        img = cv2.imread(str(ip))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]

        # مقیاس‌گذاری یکنواخت به long_side
        s = self.long / max(H0, W0)
        if s != 1.0:
            img = cv2.resize(img, (int(W0 * s), int(H0 * s)), interpolation=cv2.INTER_CUBIC)
        H, W = img.shape[:2]

        # خواندن نقاط و اعمال مقیاس
        jp = (self.root / "points" / (ip.stem + ".json"))
        pts = read_points(jp)
        if pts.size > 0 and s != 1.0:
            pts = pts * s

        # Augment تصویر و نقاط باهم
        if self.aug:
            pts_list = [tuple(map(float, p)) for p in pts]
            auged = self.aug(image=img, keypoints=pts_list)
            img = auged["image"]
            pts = np.array(auged["keypoints"], dtype=np.float32) if len(auged["keypoints"]) else np.zeros((0, 2), np.float32)

        # ساخت density map (sigma نسبی به ابعاد)
        sigma = max(2.5, 0.0075 * max(H, W))
        dm = make_density(H, W, pts, sigma)

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        dm_t = torch.from_numpy(dm[None, ...]).float()  # 1xHxW
        return img_t, dm_t

# --------------------------- Training --------------------------

def train(
    train_root="data/train",
    val_root="data/val",
    out="weights/density_csrnet.pt",
    epochs=80,
    batch=2,
    lr=1e-4,
    long_side=1280,
    lambda_count=0.01,  # وزن بسیار ملایم برای count loss
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    tr = PointDataset(train_root, long_side=long_side, aug=True)
    va = PointDataset(val_root, long_side=long_side, aug=False)

    num_workers = max(1, (os.cpu_count() or 2) // 2)
    pin_mem = device == "cuda"

    tl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=pin_mem, drop_last=False)
    vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=1, pin_memory=pin_mem)

    net = CSRNetLite().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    mse = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val = float("inf")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    for ep in range(1, epochs + 1):
        net.train()
        tbar = tqdm(tl, desc=f"Epoch {ep}/{epochs}")
        running = 0.0

        for img, dm in tbar:
            img = img.to(device, non_blocking=True)
            dm = dm.to(device, non_blocking=True)  # 1xHxW

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                pred = net(img)  # Bx1xhxw
                # GT را با average pooling به اندازه‌ی pred بیاوریم تا مجموع حفظ شود
                gt_small = F.adaptive_avg_pool2d(dm, pred.shape[-2:])  # Bx1xhxw (sum-preserving)
                loss_mse = mse(pred, gt_small)
                # count loss (L1 بین جمع pred و GT)
                pred_sum = pred.sum(dim=[1, 2, 3])
                gt_sum = gt_small.sum(dim=[1, 2, 3])
                loss_cnt = F.l1_loss(pred_sum, gt_sum)
                loss = loss_mse + lambda_count * loss_cnt

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            tbar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{loss_mse.item():.4f}", cnt=f"{loss_cnt.item():.4f}")

        # ---------- Validation ----------
        net.eval()
        val_mse = 0.0
        val_mae_count = 0.0
        with torch.inference_mode():
            for img, dm in vl:
                img = img.to(device, non_blocking=True)
                dm = dm.to(device, non_blocking=True)
                pred = net(img)
                gt_small = F.adaptive_avg_pool2d(dm, pred.shape[-2:])
                val_mse += mse(pred, gt_small).item()
                # count MAE
                pred_sum = pred.sum().item()
                gt_sum = gt_small.sum().item()
                val_mae_count += abs(pred_sum - gt_sum)

        n_val = max(1, len(vl))
        val_mse /= n_val
        val_mae_count /= n_val
        print(f"[Val] MSE: {val_mse:.4f} | Count MAE: {val_mae_count:.3f}")

        # معیار بهتر: ترکیب MSE و MAE
        score = val_mse + 0.001 * val_mae_count
        if score < best_val:
            best_val = score
            torch.save(net.state_dict(), out)
            print(f"Saved best to: {out}")

    print("Done.")

# --------------------------- CLI -------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="data/train")
    ap.add_argument("--val_root", default="data/val")
    ap.add_argument("--out", default="weights/density_csrnet.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--long_side", type=int, default=1280)
    ap.add_argument("--lambda_count", type=float, default=0.01)
    args = ap.parse_args()
    train(args.train_root, args.val_root, args.out, args.epochs, args.batch, args.lr, args.long_side, args.lambda_count)

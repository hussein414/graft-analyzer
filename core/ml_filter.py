# core/ml_filter.py
# فیلتر سبک روی پچ‌های 32×32 برای حذف پرتی‌ها (glare/rim/انعکاس)
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

MODEL_PATH = os.getenv("GA_FILTER_WEIGHTS", "models/graft_filter.pt")

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 4 * 4, 64)   # 32→16→8→4
        self.fc2   = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x  # احتمال «گرافت»

class LazyPatchFilter:
    _model = None
    _device = "cpu"

    @classmethod
    def is_ready(cls) -> bool:
        return _TORCH_OK and os.path.isfile(MODEL_PATH)

    @classmethod
    def _ensure_model(cls):
        if not cls.is_ready():
            raise RuntimeError("Model not available")
        if cls._model is None:
            state = torch.load(MODEL_PATH, map_location="cpu")
            model = TinyCNN()
            model.load_state_dict(state)
            model.eval()
            cls._model = model

    @classmethod
    def predict(cls, patches01: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        patches01: [N,32,32] float در بازه [0,1]
        خروجی: mask بولی (True=قبول)
        """
        if patches01.size == 0:
            return np.array([], dtype=bool)
        cls._ensure_model()
        x = torch.from_numpy(patches01).float().unsqueeze(1)
        with torch.no_grad():
            y = cls._model(x).squeeze(1).cpu().numpy()
        return (y >= float(threshold))

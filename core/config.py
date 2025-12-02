# core/config.py  (نسخه‌ی مینیمال برای سازگاری)
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    mode: str = "cv"
    density_ckpt: str = "weights/density_csrnet.pt"
    density_long_side: int = 1280

    @classmethod
    def from_dict(cls, cfg: dict) -> "PipelineConfig":
        fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in (cfg or {}).items() if k in fields}
        return cls(**filtered)

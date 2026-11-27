from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDAT_MAGIC = 0x54414453
SDAT_VERSION = 1

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "sram_192_v16.pth"
DEFAULT_TRAIN_DATA = ROOT / "artifacts" / "data" / "sram_192_train_1024.sdat"
DEFAULT_PRED_DIR = ROOT / "artifacts" / "predictions"
DEFAULT_LOG_DIR = ROOT / "artifacts" / "logs"

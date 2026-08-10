"""Shared filesystem paths for the PhotoRevive project."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASETS_DIR = DATA_DIR / "datasets"
TEXTURES_DIR = DATA_DIR / "textures"

MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODELS_DIR = MODELS_DIR / "base"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
WEIGHTS_DIR = MODELS_DIR / "weights"
CLASSIFIERS_DIR = MODELS_DIR / "classifiers"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
API_DATA_DIR = ARTIFACTS_DIR / "api"
CACHE_DIR = ARTIFACTS_DIR / "cache"
EVALUATIONS_DIR = ARTIFACTS_DIR / "evaluations"
LOGS_DIR = ARTIFACTS_DIR / "logs"

THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"

"""
config.py — Configuration centrale RAF-DB Pipeline
"""
import os

# ─────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────
DATASET_ROOT   = "data/RAF-DB/basic"
IMAGE_DIR      = os.path.join(DATASET_ROOT, "Image/aligned")
LABEL_FILE     = os.path.join(DATASET_ROOT, "EmoLabel/list_patition_label.txt")
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
LOGS_DIR       = "logs"

# ─────────────────────────────────────────────────────
# CLASSES
# ─────────────────────────────────────────────────────
CLASS_NAMES = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral"
}
CLASS_EMOJIS = {
    "Surprise":  "😮",
    "Fear":      "😨",
    "Disgust":   "🤢",
    "Happiness": "😊",
    "Sadness":   "😢",
    "Anger":     "😠",
    "Neutral":   "😐",
}
NUM_CLASSES = 7

# ─────────────────────────────────────────────────────
# HYPERPARAMÈTRES
# ─────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 64
NUM_EPOCHS   = 30
LR_BACKBONE  = 5e-4  # plus grand car from scratch
LR_HEAD      = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.4
VAL_SPLIT    = 0.15      # 15% du train → validation

# Phase 2 fine-tuning
NUM_EPOCHS_P2 = 10
LR_P2         = 1e-5

# ─────────────────────────────────────────────────────
# MODÈLE
# ─────────────────────────────────────────────────────
MODEL_NAME      = "FaceEmotionCNN"
PRETRAINED      = False  # from scratch
FREEZE_BACKBONE = False  # from scratch

# ─────────────────────────────────────────────────────
# MLFLOW
# ─────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "RAF-DB-Pipeline"
MLFLOW_RUN_NAME   = "resnet18-full-pipeline"

# ─────────────────────────────────────────────────────
# DIVERS
# ─────────────────────────────────────────────────────
SEED        = 42
NUM_WORKERS = 4
PIN_MEMORY  = True
BEST_CKPT   = os.path.join(CHECKPOINT_DIR, "best_model.pth")
FINAL_CKPT  = os.path.join(CHECKPOINT_DIR, "final_model.pth")

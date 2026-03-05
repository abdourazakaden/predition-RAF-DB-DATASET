"""
dataset.py
ÉTAPE 1 — Chargement du dataset
ÉTAPE 2 — Prétraitement des images
ÉTAPE 3 — Data augmentation
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config


# ═══════════════════════════════════════════════════
# ÉTAPE 2 & 3 — PRÉTRAITEMENT + DATA AUGMENTATION
# ═══════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(split: str):
    """
    Retourne les transformations adaptées à chaque split.

    Train  → augmentation forte (flip, rotation, color jitter, affine, erasing)
    Val    → redimensionnement + normalisation uniquement
    Test   → redimensionnement + normalisation uniquement
    """
    if split == "train":
        return transforms.Compose([
            # Redimensionnement
            transforms.Resize((config.IMG_SIZE + 20, config.IMG_SIZE + 20)),
            transforms.RandomCrop(config.IMG_SIZE),
            # Géométrique
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1), shear=5),
            # Couleur
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            # Normalisation
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # Random erasing (simule occlusions)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

def get_tta_transforms():
    """Test-Time Augmentation — 5 crops pour une meilleure inférence."""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE + 20, config.IMG_SIZE + 20)),
        transforms.FiveCrop(config.IMG_SIZE),
        transforms.Lambda(lambda crops: [
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])(c) for c in crops
        ]),
    ])


# ═══════════════════════════════════════════════════
# ÉTAPE 1 — CHARGEMENT DU DATASET
# ═══════════════════════════════════════════════════

class RAFDBDataset(Dataset):
    """
    Dataset RAF-DB Basic (7 expressions faciales).

    Paramètres
    ----------
    df         : DataFrame [filename, label, split]
    image_dir  : Dossier des images alignées
    split      : 'train' | 'val' | 'test'
    transform  : Pipeline de transformations
    """
    def __init__(self, df, image_dir, split="train", transform=None):
        self.data      = df[df["split"] == split].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform or get_transforms(split)
        self.split     = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        img_name = row["filename"].replace(".jpg", "_aligned.jpg")
        img_path = os.path.join(self.image_dir, img_name)
        image    = Image.open(img_path).convert("RGB")
        label    = int(row["label"]) - 1   # 0-indexé (0..6)
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_labels(self):
        return (self.data["label"].values - 1).tolist()


def load_dataframe(label_file: str) -> pd.DataFrame:
    """
    Étape 1a — Lit list_patition_label.txt et crée le DataFrame.
    Ajoute une colonne 'split' : train / test (selon le préfixe RAF-DB).
    """
    df = pd.read_csv(label_file, sep=" ", header=None,
                     names=["filename", "label"])
    df["split"]      = df["filename"].apply(
        lambda x: "train" if "train" in x else "test")
    df["class_name"] = df["label"].map(config.CLASS_NAMES)
    return df


def create_val_split(df: pd.DataFrame, val_ratio=config.VAL_SPLIT,
                     seed=config.SEED) -> pd.DataFrame:
    """
    Étape 1b — Extrait une validation stratifiée depuis le train.
    Les images de validation gardent le split 'val'.
    """
    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()

    tr_idx, val_idx = train_test_split(
        train_df.index,
        test_size=val_ratio,
        stratify=train_df["label"],
        random_state=seed,
    )
    df_new = df.copy()
    df_new.loc[val_idx, "split"] = "val"
    return df_new


def compute_class_weights(df: pd.DataFrame) -> np.ndarray:
    """Calcule les poids inversement proportionnels aux effectifs (train)."""
    labels  = df[df["split"] == "train"]["label"].values - 1
    counts  = np.bincount(labels, minlength=config.NUM_CLASSES).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * config.NUM_CLASSES
    return weights


def get_dataloaders(df: pd.DataFrame):
    """
    Construit et retourne les DataLoaders train / val / test.
    """
    datasets = {
        split: RAFDBDataset(df, config.IMAGE_DIR, split=split)
        for split in ["train", "val", "test"]
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=config.BATCH_SIZE,
                            shuffle=True,  num_workers=config.NUM_WORKERS,
                            pin_memory=config.PIN_MEMORY),
        "val":   DataLoader(datasets["val"],   batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=config.PIN_MEMORY),
        "test":  DataLoader(datasets["test"],  batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=config.PIN_MEMORY),
    }
    for split, ds in datasets.items():
        print(f"  {split:5s} : {len(ds):5d} images | "
              f"{len(loaders[split]):3d} batches")
    return loaders, datasets


def print_dataset_info(df: pd.DataFrame):
    """Affiche les statistiques complètes du dataset."""
    print("\n" + "═" * 55)
    print("  📊 RAF-DB — Statistiques du dataset")
    print("═" * 55)
    total = len(df)
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        print(f"\n  [{split.upper()}] — {len(sub)} images "
              f"({100*len(sub)/total:.1f}%)")
        counts = sub["class_name"].value_counts()
        for name, cnt in counts.items():
            pct = 100 * cnt / len(sub)
            bar = "▓" * int(pct / 3)
            emoji = config.CLASS_EMOJIS.get(name, "")
            print(f"    {emoji} {name:12s} : {cnt:5d}  ({pct:5.1f}%)  {bar}")
    print("═" * 55)

"""
utils.py — Visualisations, early stopping, sauvegarde, seed
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import config


# ─────────────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────────────

def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────

class EarlyStopping:
    """
    Arrête l'entraînement si la métrique de validation ne s'améliore pas
    pendant `patience` époques consécutives.
    """
    def __init__(self, patience=7, delta=1e-4, mode="max", verbose=True):
        self.patience  = patience
        self.delta     = delta
        self.mode      = mode
        self.verbose   = verbose
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, value):
        if self.best is None:
            self.best = value
            return False

        improved = (value > self.best + self.delta) if self.mode == "max" \
                   else (value < self.best - self.delta)

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ⏳ EarlyStopping : {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
                if self.verbose:
                    print("  🛑 Early stopping déclenché !")
        return self.stop


# ─────────────────────────────────────────────────────
# SAUVEGARDE / CHARGEMENT
# ─────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics":     metrics,
    }, path)

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt["epoch"], ckpt["metrics"]


# ─────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────

def plot_class_distribution(df, save_dir=config.RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    splits = [s for s in ["train", "val", "test"] if s in df["split"].values]
    fig, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 5))
    if len(splits) == 1:
        axes = [axes]
    palette = sns.color_palette("Set3", config.NUM_CLASSES)

    for ax, split in zip(axes, splits):
        sub    = df[df["split"] == split]
        counts = sub["class_name"].value_counts()
        bars   = ax.bar(counts.index, counts.values,
                        color=palette, edgecolor="white", linewidth=1.2)
        ax.set_title(f"{split.upper()} ({len(sub)} images)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Expression"); ax.set_ylabel("Nombre d'images")
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 10,
                    str(v), ha="center", fontsize=9)

    plt.suptitle("RAF-DB — Distribution des classes", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(save_dir, "1_class_distribution.png")


def plot_augmentation_samples(dataset, n_samples=8, save_dir=config.RESULTS_DIR):
    """Montre des exemples avant/après augmentation."""
    os.makedirs(save_dir, exist_ok=True)
    from torchvision import transforms as T
    inv = T.Compose([
        T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ])
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 5))
    indices   = np.random.choice(len(dataset), n_samples, replace=False)

    for col, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img_np = inv(img_tensor).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0][col].imshow(img_np)
        axes[0][col].axis("off")
        if col == 0:
            axes[0][col].set_ylabel("Augmenté", fontsize=10, fontweight="bold")
        axes[0][col].set_title(config.CLASS_NAMES[label + 1], fontsize=8)

        # Image originale (sans augmentation)
        from dataset import get_transforms
        orig_tf   = get_transforms("test")
        row       = dataset.data.iloc[idx]
        img_name  = row["filename"].replace(".jpg", "_aligned.jpg")
        img_path  = os.path.join(dataset.image_dir, img_name)
        orig_img  = orig_tf(
            __import__("PIL").Image.open(img_path).convert("RGB")
        )
        orig_np = inv(orig_img).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[1][col].imshow(orig_np)
        axes[1][col].axis("off")
        if col == 0:
            axes[1][col].set_ylabel("Original", fontsize=10, fontweight="bold")

    plt.suptitle("Data Augmentation — Exemples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(save_dir, "2_augmentation_samples.png")


def plot_training_curves(history, save_dir=config.RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train",
                 color="steelblue", lw=2, ms=3)
    axes[0].plot(epochs, history["val_loss"],   "o-", label="Val",
                 color="tomato",    lw=2, ms=3)
    axes[0].set_title("Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Époque"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train",
                 color="steelblue", lw=2, ms=3)
    axes[1].plot(epochs, history["val_acc"],   "o-", label="Val",
                 color="tomato",    lw=2, ms=3)
    axes[1].set_title("Accuracy (%)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Époque"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # LR
    if "lr" in history and history["lr"]:
        axes[2].plot(epochs, history["lr"], "o-", color="purple", lw=2, ms=3)
        axes[2].set_title("Learning Rate", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("Époque"); axes[2].set_ylabel("LR")
        axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)
    else:
        axes[2].axis("off")

    plt.suptitle("Courbes d'entraînement", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(save_dir, "5_training_curves.png")


def plot_confusion_matrix(labels, preds, save_dir=config.RESULTS_DIR,
                          phase="test"):
    os.makedirs(save_dir, exist_ok=True)
    class_names = list(config.CLASS_NAMES.values())
    cm      = __import__("sklearn.metrics", fromlist=["confusion_matrix"])\
                .confusion_matrix(labels, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0],
                linewidths=0.5)
    axes[0].set_title("Matrice de Confusion (valeurs)",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Vrai label"); axes[0].set_xlabel("Prédit")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                linewidths=0.5)
    axes[1].set_title("Matrice de Confusion (normalisée)",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Vrai label"); axes[1].set_xlabel("Prédit")

    plt.suptitle(f"Matrice de Confusion — {phase.upper()}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(save_dir, f"7_confusion_matrix_{phase}.png")


def plot_per_class_metrics(report_dict, save_dir=config.RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    class_names = list(config.CLASS_NAMES.values())
    metrics     = ["precision", "recall", "f1-score"]
    data        = {m: [report_dict[c][m] for c in class_names]
                   for m in metrics}

    x    = np.arange(len(class_names))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    colors  = ["steelblue", "tomato", "seagreen"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * w, data[metric], w, label=metric.capitalize(),
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x + w)
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Métriques par classe — Test Set",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(save_dir, "8_per_class_metrics.png")


def _save(directory, filename):
    path = os.path.join(directory, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def plot_feature_maps(model, image_tensor, save_dir=config.RESULTS_DIR):
    """Visualise les feature maps des différents stages du CNN."""
    import torch
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        maps = model.get_feature_maps(image_tensor.unsqueeze(0).to(device))

    fig, axes = plt.subplots(len(maps), 8, figsize=(20, len(maps) * 2.5))
    fig.patch.set_facecolor("#1a1a2e")

    for row, (stage_name, fmap) in enumerate(maps.items()):
        fmap_np = fmap.squeeze(0).cpu().numpy()
        n_show  = min(8, fmap_np.shape[0])
        for col in range(n_show):
            ax = axes[row][col]
            ax.imshow(fmap_np[col], cmap="viridis")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(stage_name, color="white",
                              fontsize=9, fontweight="bold")
        # Cacher les colonnes vides
        for col in range(n_show, 8):
            axes[row][col].axis("off")

    plt.suptitle("Feature Maps — FaceEmotionCNN",
                 fontsize=14, fontweight="bold", color="white")
    plt.tight_layout()
    _save(save_dir, "4_feature_maps.png")


def plot_model_architecture(save_dir=config.RESULTS_DIR):
    """Dessine un schéma simple de l'architecture CNN."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    stages = [
        ("Input\n3×224×224",   "#e74c3c", 0.5),
        ("STEM\n64×56×56",     "#e67e22", 1.5),
        ("Stage 1\n128×28×28", "#f1c40f", 2.5),
        ("Stage 2\n256×14×14", "#2ecc71", 3.5),
        ("Stage 3\n512×7×7",   "#3498db", 4.5),
        ("Stage 4\n512×7×7",   "#9b59b6", 5.5),
        ("GAP\n512",           "#1abc9c", 6.5),
        ("Head\n7 classes",    "#e74c3c", 7.5),
    ]
    for label, color, x in stages:
        ax.add_patch(plt.FancyBboxPatch(
            (x - 0.42, 0.2), 0.84, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85
        ))
        ax.text(x, 0.5, label, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white")
        if x < 7.5:
            ax.annotate("", xy=(x + 0.5, 0.5), xytext=(x + 0.42, 0.5),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

    # Légende SE + Residual
    ax.text(4.5, 0.1,
            "Chaque stage contient : ConvBlock + SE Block + ResidualBlock",
            ha="center", color="rgba(255,255,255,0.6)" if False else "lightgray",
            fontsize=9, style="italic")
    plt.title("Architecture FaceEmotionCNN — From Scratch",
              color="white", fontsize=13, fontweight="bold", pad=15)
    plt.xlim(0, 8)
    plt.ylim(0, 1)
    _save(save_dir, "4_cnn_architecture.png")

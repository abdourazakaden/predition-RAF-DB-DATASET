"""
predict.py
ÉTAPE 8 — Prédiction de l'émotion (image unique ou dossier)

Usage :
    python predict.py --image photo.jpg --checkpoint checkpoints/best_model.pth
    python predict.py --folder images/  --checkpoint checkpoints/best_model.pth --tta
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms

import config
from model import load_model


# ═══════════════════════════════════════════════════
# TRANSFORMATION D'INFÉRENCE
# ═══════════════════════════════════════════════════

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ═══════════════════════════════════════════════════
# ÉTAPE 8 — PRÉDICTION
# ═══════════════════════════════════════════════════

class EmotionPredictor:
    """
    Prédit l'expression faciale d'une image.

    Paramètres
    ----------
    ckpt_path : chemin vers le checkpoint .pth
    device    : 'cuda' ou 'cpu'
    """
    def __init__(self, ckpt_path: str, device=None):
        self.device    = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model     = load_model(ckpt_path, self.device)
        self.transform = get_inference_transform()
        self.class_names = list(config.CLASS_NAMES.values())
        print(f"  ✅ Modèle chargé — device : {self.device}")

    def predict(self, img_path: str) -> dict:
        """
        Prédit l'émotion d'une image.
        Retourne un dict {class, confidence, probabilities, image}.
        """
        img    = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)
            probs   = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

        pred_idx = probs.argmax()
        return {
            "class":         self.class_names[pred_idx],
            "class_id":      int(pred_idx),
            "confidence":    float(probs[pred_idx]),
            "probabilities": probs,
            "image":         img,
            "top3": sorted(
                zip(self.class_names, probs),
                key=lambda x: -x[1]
            )[:3],
        }

    def predict_batch(self, img_paths: list) -> list:
        """Prédit plusieurs images en batch."""
        return [self.predict(p) for p in img_paths]

    def visualize(self, result: dict, save_path: str = None):
        """Affiche l'image avec les probabilités par classe."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#1a1a2e")

        # Image
        axes[0].imshow(result["image"])
        axes[0].axis("off")
        emoji = config.CLASS_EMOJIS.get(result["class"], "")
        conf  = result["confidence"] * 100
        color = "lime" if conf >= 70 else "orange" if conf >= 50 else "red"
        axes[0].set_title(
            f"{emoji}  {result['class']}\nConfiance : {conf:.1f}%",
            fontsize=14, fontweight="bold", color=color,
            pad=12
        )
        axes[0].set_facecolor("#1a1a2e")

        # Barres de probabilité
        names  = list(config.CLASS_NAMES.values())
        probs  = result["probabilities"] * 100
        colors = ["#00d4aa" if n == result["class"] else "#3d5af1"
                  for n in names]
        bars   = axes[1].barh(names, probs, color=colors,
                               edgecolor="none", height=0.6)
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel("Probabilité (%)", color="white", fontsize=11)
        axes[1].set_title("Distribution des probabilités",
                          color="white", fontsize=12, fontweight="bold")
        axes[1].tick_params(colors="white")
        axes[1].set_facecolor("#16213e")
        for spine in axes[1].spines.values():
            spine.set_edgecolor("#3d5af1")
        for bar, p in zip(bars, probs):
            axes[1].text(p + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{p:.1f}%", va="center", fontsize=9, color="white")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="#1a1a2e")
            print(f"  ✅ Sauvegardé : {save_path}")
        else:
            plt.show()
        plt.close()


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="RAF-DB — Prédiction")
    p.add_argument("--image",      type=str, help="Chemin vers une image")
    p.add_argument("--folder",     type=str, help="Dossier d'images")
    p.add_argument("--checkpoint", type=str, default=config.BEST_CKPT)
    p.add_argument("--output",     type=str, default="results/predictions")
    return p.parse_args()


def main():
    args      = parse_args()
    predictor = EmotionPredictor(args.checkpoint)
    os.makedirs(args.output, exist_ok=True)

    if args.image:
        result = predictor.predict(args.image)
        print(f"\n  🎭 Expression : {result['class']}  "
              f"({result['confidence']*100:.1f}%)")
        print(f"  Top 3 :")
        for name, prob in result["top3"]:
            emoji = config.CLASS_EMOJIS.get(name, "")
            print(f"    {emoji} {name:12s} : {prob*100:.1f}%")
        save = os.path.join(args.output,
               os.path.splitext(os.path.basename(args.image))[0] + "_pred.png")
        predictor.visualize(result, save)

    elif args.folder:
        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted([
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in exts
        ])
        print(f"\n  📁 {len(files)} images trouvées")
        for path in files:
            result = predictor.predict(path)
            conf   = result["confidence"] * 100
            emoji  = config.CLASS_EMOJIS.get(result["class"], "")
            print(f"  {os.path.basename(path):35s} "
                  f"{emoji} {result['class']:12s}  ({conf:.1f}%)")
            save = os.path.join(args.output,
                   os.path.splitext(os.path.basename(path))[0] + "_pred.png")
            predictor.visualize(result, save)
    else:
        print("  ❌ Spécifiez --image ou --folder")


if __name__ == "__main__":
    main()

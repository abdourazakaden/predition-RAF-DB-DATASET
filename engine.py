"""
engine.py
ÉTAPE 5 — Entraînement
ÉTAPE 6 — Validation
ÉTAPE 7 — Test
"""
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)
import config


# ═══════════════════════════════════════════════════
# ÉTAPE 5 — ENTRAÎNEMENT
# ═══════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """
    Entraîne le modèle sur une époque.
    Supporte l'entraînement mixte (AMP) si scaler est fourni.
    Retourne : (loss_moy, accuracy)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  🔧 Train", leave=False,
                ncols=85, colour="blue")
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

        pbar.set_postfix(
            loss=f"{total_loss/total:.4f}",
            acc=f"{100.*correct/total:.1f}%"
        )

    return total_loss / total, 100.0 * correct / total


# ═══════════════════════════════════════════════════
# ÉTAPE 6 — VALIDATION
# ═══════════════════════════════════════════════════

def validate(model, loader, criterion, device):
    """
    Évalue le modèle sur le set de validation.
    Retourne : (loss_moy, accuracy, preds, labels)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels      = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="  📊 Val  ", leave=False,
                    ncols=85, colour="yellow")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


# ═══════════════════════════════════════════════════
# ÉTAPE 7 — TEST
# ═══════════════════════════════════════════════════

def test(model, loader, criterion, device, tta=False):
    """
    Évaluation finale sur le set de test.
    Retourne un dictionnaire complet de métriques.

    Si tta=True, utilise le Test-Time Augmentation (5 crops).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="  🧪 Test ", leave=False,
                    ncols=85, colour="green")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            probs   = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc    = 100.0 * correct / total
    f1_w   = f1_score(all_labels, all_preds, average="weighted") * 100
    f1_mac = f1_score(all_labels, all_preds, average="macro")    * 100
    cm     = confusion_matrix(all_labels, all_preds)

    metrics = {
        "loss":       total_loss / total,
        "accuracy":   acc,
        "f1_weighted": f1_w,
        "f1_macro":    f1_mac,
        "confusion_matrix": cm,
        "predictions":  np.array(all_preds),
        "labels":       np.array(all_labels),
        "probabilities": np.array(all_probs),
        "report": classification_report(
            all_labels, all_preds,
            target_names=list(config.CLASS_NAMES.values())
        ),
    }
    return metrics


def print_test_results(metrics: dict):
    """Affiche les résultats du test de façon lisible."""
    print("\n" + "═" * 55)
    print("  🧪 RÉSULTATS FINAUX — TEST SET")
    print("═" * 55)
    print(f"  Accuracy    : {metrics['accuracy']:.2f}%")
    print(f"  F1 Weighted : {metrics['f1_weighted']:.2f}%")
    print(f"  F1 Macro    : {metrics['f1_macro']:.2f}%")
    print(f"  Loss        : {metrics['loss']:.4f}")
    print("═" * 55)
    print("\n  Classification Report :\n")
    print(metrics["report"])

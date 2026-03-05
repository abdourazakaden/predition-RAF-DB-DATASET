"""
train.py — Script principal du pipeline RAF-DB avec MLflow
ÉTAPES 5 (Entraînement) + 6 (Validation) + 7 (Test)

Usage :
    python train.py
    python train.py --epochs 40 --batch_size 32 --phase2
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report
import json

import config
from dataset  import (load_dataframe, create_val_split, compute_class_weights,
                       get_dataloaders, print_dataset_info, RAFDBDataset)
from model    import build_model
from engine   import train_one_epoch, validate, test, print_test_results
from utils    import (set_seed, EarlyStopping, save_checkpoint, load_checkpoint,
                       plot_class_distribution, plot_augmentation_samples,
                       plot_training_curves, plot_confusion_matrix,
                       plot_per_class_metrics)


def parse_args():
    p = argparse.ArgumentParser(description="RAF-DB Training Pipeline")
    p.add_argument("--epochs",      type=int,   default=config.NUM_EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=config.LR_BACKBONE)
    p.add_argument("--lr_head",     type=float, default=config.LR_HEAD)
    p.add_argument("--patience",    type=int,   default=7)
    p.add_argument("--phase2",      action="store_true")
    p.add_argument("--epochs2",     type=int,   default=config.NUM_EPOCHS_P2)
    p.add_argument("--amp",         action="store_true",
                   help="Activer Mixed Precision Training")
    p.add_argument("--resume",      type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device : {device}")
    if torch.cuda.is_available():
        print(f"   GPU    : {torch.cuda.get_device_name(0)}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR,    exist_ok=True)
    os.makedirs(config.LOGS_DIR,       exist_ok=True)

    # ── ÉTAPE 1 — Chargement ────────────────────────────
    print("\n" + "═"*55)
    print("  ÉTAPE 1 — Chargement du dataset")
    print("═"*55)
    df = load_dataframe(config.LABEL_FILE)
    df = create_val_split(df)
    print_dataset_info(df)

    # ── ÉTAPE 3 — Augmentation + DataLoaders ────────────
    print("\n" + "═"*55)
    print("  ÉTAPES 2 & 3 — Prétraitement + Data Augmentation")
    print("═"*55)
    loaders, datasets = get_dataloaders(df)
    plot_class_distribution(df, config.RESULTS_DIR)
    try:
        plot_augmentation_samples(datasets["train"], save_dir=config.RESULTS_DIR)
    except Exception:
        pass

    # Class weights
    cw       = compute_class_weights(df)
    cw_t     = torch.FloatTensor(cw).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_t)
    print("\n  ⚖️  Class Weights :")
    for name, w in zip(config.CLASS_NAMES.values(), cw):
        print(f"    {name:12s} : {w:.4f}")

    # ── ÉTAPE 4 — Modèle ────────────────────────────────
    print("\n" + "═"*55)
    print("  ÉTAPE 4 — Construction du modèle CNN")
    print("═"*55)
    model = build_model().to(device)
    model.summary()

    # Optimizer & Scheduler
    param_groups = model.get_param_groups(args.lr, args.lr_head)
    optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, verbose=True)
    early_stop = EarlyStopping(patience=args.patience, mode="max")

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" \
             else None
    if scaler:
        print("  ⚡ Mixed Precision Training activé")

    # Resume
    start_epoch = 1
    best_acc    = 0.0
    if args.resume:
        start_epoch, ckpt_metrics = load_checkpoint(
            model, optimizer, args.resume, device)
        best_acc = ckpt_metrics.get("val_acc", 0.0)
        print(f"  ▶️  Reprise depuis époque {start_epoch}")

    # ── ÉTAPES 5 & 6 — Entraînement + Validation ────────
    print("\n" + "═"*55)
    print(f"  ÉTAPES 5 & 6 — Entraînement + Validation ({args.epochs} époques)")
    print("═"*55)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   [], "lr": []}

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=config.MLFLOW_RUN_NAME) as run:
        print(f"\n  🔬 MLflow Run : {run.info.run_id}")

        mlflow.log_params({
            "model": config.MODEL_NAME, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr_backbone": args.lr,
            "lr_head": args.lr_head, "img_size": config.IMG_SIZE,
            "val_split": config.VAL_SPLIT, "early_stop_patience": args.patience,
            "amp": args.amp, "phase2": args.phase2,
        })

        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\n  Époque [{epoch:02d}/{args.epochs}]")

            tr_loss, tr_acc = train_one_epoch(
                model, loaders["train"], criterion, optimizer, device, scaler)
            vl_loss, vl_acc, vl_preds, vl_labels = validate(
                model, loaders["val"], criterion, device)

            cur_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(vl_acc)

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)
            history["lr"].append(cur_lr)

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": vl_loss,   "val_acc": vl_acc, "lr": cur_lr,
            }, step=epoch)

            flag = ""
            if vl_acc > best_acc:
                best_acc = vl_acc
                save_checkpoint(model, optimizer, epoch,
                                {"val_acc": vl_acc}, config.BEST_CKPT)
                flag = "  ⭐ BEST"

            print(f"    Train  → Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%")
            print(f"    Val    → Loss: {vl_loss:.4f}  Acc: {vl_acc:.2f}%{flag}")

            if early_stop(vl_acc):
                break

        print(f"\n  ✅ Meilleure Val Accuracy (Phase 1) : {best_acc:.2f}%")
        mlflow.log_metric("best_val_acc_phase1", best_acc)

        # Visualisations Phase 1
        plot_training_curves(history, config.RESULTS_DIR)

        # ── ÉTAPE 6 Validation finale ─────────────────
        print("\n" + "═"*55)
        print("  ÉTAPE 6 — Validation finale (meilleur modèle)")
        print("═"*55)
        load_checkpoint(model, None, config.BEST_CKPT, device)
        _, val_acc_f, vp, vl = validate(model, loaders["val"], criterion, device)
        print(f"  Val Accuracy : {val_acc_f:.2f}%")
        plot_confusion_matrix(vl, vp, config.RESULTS_DIR, phase="val")

        # ── Phase 2 (optionnel) ───────────────────────
        if args.phase2:
            print("\n" + "═"*55)
            print(f"  🔓 Phase 2 — Fine-tuning complet ({args.epochs2} époques)")
            print("═"*55)
            model.unfreeze_all()
            model.summary()

            opt2 = optim.AdamW(model.parameters(),
                               lr=config.LR_P2, weight_decay=config.WEIGHT_DECAY)
            sch2 = optim.lr_scheduler.CosineAnnealingLR(
                opt2, T_max=args.epochs2, eta_min=1e-7)
            es2  = EarlyStopping(patience=5, mode="max")
            best2 = best_acc
            hist2 = {"train_loss": [], "train_acc": [],
                     "val_loss": [], "val_acc": [], "lr": []}

            for epoch in range(1, args.epochs2 + 1):
                print(f"\n  [P2] Époque [{epoch:02d}/{args.epochs2}]")
                tr_loss, tr_acc = train_one_epoch(
                    model, loaders["train"], criterion, opt2, device, scaler)
                vl_loss, vl_acc, _, _ = validate(
                    model, loaders["val"], criterion, device)
                sch2.step()

                hist2["train_loss"].append(tr_loss)
                hist2["train_acc"].append(tr_acc)
                hist2["val_loss"].append(vl_loss)
                hist2["val_acc"].append(vl_acc)
                hist2["lr"].append(opt2.param_groups[0]["lr"])

                mlflow.log_metrics({
                    "p2_train_loss": tr_loss, "p2_train_acc": tr_acc,
                    "p2_val_loss": vl_loss,   "p2_val_acc": vl_acc,
                }, step=epoch)

                flag = ""
                if vl_acc > best2:
                    best2 = vl_acc
                    save_checkpoint(model, opt2, epoch,
                                    {"val_acc": vl_acc}, config.FINAL_CKPT)
                    flag = "  ⭐ BEST"
                print(f"    Train  → Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%")
                print(f"    Val    → Loss: {vl_loss:.4f}  Acc: {vl_acc:.2f}%{flag}")
                if es2(vl_acc): break

            print(f"\n  ✅ Meilleure Val Accuracy (Phase 2) : {best2:.2f}%")
            mlflow.log_metric("best_val_acc_phase2", best2)
            results_p2 = os.path.join(config.RESULTS_DIR, "phase2")
            plot_training_curves(hist2, results_p2)

        # ── ÉTAPE 7 — TEST ────────────────────────────
        print("\n" + "═"*55)
        print("  ÉTAPE 7 — Test final")
        print("═"*55)
        final_ckpt = config.FINAL_CKPT if args.phase2 else config.BEST_CKPT
        load_checkpoint(model, None, final_ckpt, device)
        test_metrics = test(model, loaders["test"], criterion, device)
        print_test_results(test_metrics)

        # Sauvegarder le rapport
        report_path = os.path.join(config.RESULTS_DIR, "test_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Accuracy    : {test_metrics['accuracy']:.4f}%\n")
            f.write(f"F1 Weighted : {test_metrics['f1_weighted']:.4f}%\n")
            f.write(f"F1 Macro    : {test_metrics['f1_macro']:.4f}%\n\n")
            f.write(test_metrics["report"])

        # Visualisations Test
        plot_confusion_matrix(test_metrics["labels"],
                              test_metrics["predictions"],
                              config.RESULTS_DIR, phase="test")

        # Métriques par classe
        from sklearn.metrics import classification_report as cr
        rdict = cr(test_metrics["labels"], test_metrics["predictions"],
                   target_names=list(config.CLASS_NAMES.values()),
                   output_dict=True)
        plot_per_class_metrics(rdict, config.RESULTS_DIR)

        # Log dans MLflow
        mlflow.log_metrics({
            "test_accuracy":    test_metrics["accuracy"],
            "test_f1_weighted": test_metrics["f1_weighted"],
            "test_f1_macro":    test_metrics["f1_macro"],
        })
        mlflow.log_artifacts(config.RESULTS_DIR, artifact_path="plots")
        mlflow.pytorch.log_model(model, "model")

    print("\n" + "═"*55)
    print("  🎉 Pipeline terminé avec succès !")
    print(f"  📊 Lancez : mlflow ui  (http://localhost:5000)")
    print("═"*55)


if __name__ == "__main__":
    main()

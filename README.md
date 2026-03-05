# 🎭 RAF-DB — Pipeline Complet de Reconnaissance d'Émotions
### ResNet-18 · PyTorch · MLflow · Streamlit

---

## 📁 Structure du projet

```
rafdb_full/
├── config.py        ← Hyperparamètres et chemins
├── dataset.py       ← Étapes 1, 2, 3 : Chargement, prétraitement, augmentation
├── model.py         ← Étape 4 : Construction du modèle CNN
├── engine.py        ← Étapes 5, 6, 7 : Entraînement, validation, test
├── utils.py         ← Visualisations, EarlyStopping, sauvegarde
├── train.py         ← Script principal (pipeline complet + MLflow)
├── predict.py       ← Étape 8 : Prédiction sur image(s)
├── app.py           ← Étape 9 : Application web Streamlit
└── requirements.txt
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 📂 Structure des données attendue

```
data/
└── RAF-DB/
    └── basic/
        ├── Image/
        │   └── aligned/       ← images 100×100 alignées
        └── EmoLabel/
            └── list_patition_label.txt
```

---

## 🚀 Lancer le pipeline complet

### Phase 1 uniquement (fine-tuning backbone gelé)
```bash
python train.py
```

### Phase 1 + Phase 2 (fine-tuning complet)
```bash
python train.py --phase2
```

### Avec Mixed Precision (GPU)
```bash
python train.py --phase2 --amp
```

### Options disponibles
| Argument       | Défaut | Description                        |
|---------------|--------|------------------------------------|
| `--epochs`    | 30     | Nombre d'époques Phase 1           |
| `--batch_size`| 64     | Taille de batch                    |
| `--lr`        | 1e-4   | LR backbone                        |
| `--lr_head`   | 1e-3   | LR tête FC                         |
| `--patience`  | 7      | Patience early stopping            |
| `--phase2`    | False  | Activer Phase 2                    |
| `--epochs2`   | 10     | Époques Phase 2                    |
| `--amp`       | False  | Mixed Precision Training           |
| `--resume`    | None   | Reprendre depuis un checkpoint     |

---

## 📊 Visualiser les expériences MLflow

```bash
mlflow ui
# → http://localhost:5000
```

---

## 🔮 Prédire une émotion

```bash
# Sur une image
python predict.py --image ma_photo.jpg --checkpoint checkpoints/best_model.pth

# Sur un dossier
python predict.py --folder images/ --checkpoint checkpoints/best_model.pth
```

---

## 🌐 Lancer l'application web

```bash
streamlit run app.py
# → http://localhost:8501
```

---

## 🎯 Résultats attendus

| Phase       | Val Accuracy | Test Accuracy |
|-------------|-------------|---------------|
| Phase 1     | ~82–84%     | ~82–84%       |
| Phase 2     | ~85–87%     | ~85–87%       |

---

## 🔬 Pipeline en détail

| Étape | Fichier      | Description                              |
|-------|-------------|------------------------------------------|
| 1     | dataset.py  | Chargement RAF-DB + split val stratifié |
| 2     | dataset.py  | Resize, normalisation ImageNet           |
| 3     | dataset.py  | Flip, rotation, color jitter, erasing    |
| 4     | model.py    | ResNet-18 + tête FC personnalisée        |
| 5     | engine.py   | Entraînement + gradient clipping + AMP  |
| 6     | engine.py   | Validation avec early stopping           |
| 7     | engine.py   | Test avec métriques complètes            |
| 8     | predict.py  | Inférence image unique ou batch          |
| 9     | app.py      | Interface Streamlit avec upload/caméra  |

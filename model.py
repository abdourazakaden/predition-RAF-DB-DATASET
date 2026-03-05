"""
model.py
ÉTAPE 4 — CNN from scratch custom pour RAF-DB
Architecture : FaceEmotionCNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ═══════════════════════════════════════════════════
# BLOCS DE BASE
# ═══════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """
    Bloc Conv standard :
        Conv2d → BatchNorm → ReLU → (MaxPool optionnel)
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1,
                 padding=1, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Bloc résiduel léger (skip connection) :
        Conv → BN → ReLU → Conv → BN → (+skip) → ReLU
    Permet des gradients plus stables en profondeur.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block :
    Recalibre les feature maps en pondérant les canaux
    selon leur importance — très efficace pour les visages.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


# ═══════════════════════════════════════════════════
# ÉTAPE 4 — ARCHITECTURE CNN FROM SCRATCH
# ═══════════════════════════════════════════════════

class FaceEmotionCNN(nn.Module):
    """
    CNN custom from scratch pour la reconnaissance d'émotions RAF-DB.

    Architecture (input: 3 × 224 × 224)
    ─────────────────────────────────────
    STEM     : Conv(3→32) + Conv(32→64) + MaxPool        → 64 × 56 × 56
    STAGE 1  : Conv(64→128) + SE + ResBlock + MaxPool    → 128 × 28 × 28
    STAGE 2  : Conv(128→256) + SE + ResBlock + MaxPool   → 256 × 14 × 14
    STAGE 3  : Conv(256→512) + SE + ResBlock + MaxPool   → 512 × 7 × 7
    STAGE 4  : Conv(512→512) + SE + ResBlock             → 512 × 7 × 7
    HEAD     : GAP → Dropout → FC(512→256) → BN → ReLU
                             → Dropout → FC(256→7)
    ─────────────────────────────────────
    Paramètres totaux : ~4.5M  (léger et entraînable)
    """

    def __init__(self, num_classes=config.NUM_CLASSES,
                 dropout=config.DROPOUT):
        super().__init__()

        # ── STEM ─────────────────────────────────────
        # Capture les features de bas niveau (bords, textures)
        self.stem = nn.Sequential(
            ConvBlock(3,  32, kernel=3, padding=1),          # 224→224
            ConvBlock(32, 64, kernel=3, padding=1, pool=True), # 224→112
            ConvBlock(64, 64, kernel=3, padding=1, pool=True), # 112→56
        )

        # ── STAGE 1 — Features locales ───────────────
        self.stage1 = nn.Sequential(
            ConvBlock(64, 128, pool=False),
            SEBlock(128),
            ResidualBlock(128),
            nn.MaxPool2d(2, 2),                              # 56→28
            nn.Dropout2d(0.1),
        )

        # ── STAGE 2 — Features moyennes ──────────────
        self.stage2 = nn.Sequential(
            ConvBlock(128, 256, pool=False),
            SEBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.MaxPool2d(2, 2),                              # 28→14
            nn.Dropout2d(0.15),
        )

        # ── STAGE 3 — Features abstraites ────────────
        self.stage3 = nn.Sequential(
            ConvBlock(256, 512, pool=False),
            SEBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.MaxPool2d(2, 2),                              # 14→7
            nn.Dropout2d(0.2),
        )

        # ── STAGE 4 — Features hauts niveau ──────────
        self.stage4 = nn.Sequential(
            ConvBlock(512, 512, pool=False),
            SEBlock(512),
            ResidualBlock(512),
        )

        # ── GLOBAL AVERAGE POOLING ────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)   # 512 × 7 × 7 → 512 × 1 × 1

        # ── TÊTE DE CLASSIFICATION ────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """Initialisation He pour Conv, Xavier pour Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        """Retourne les feature maps intermédiaires (pour visualisation)."""
        maps = {}
        maps["stem"]   = self.stem(x)
        maps["stage1"] = self.stage1(maps["stem"])
        maps["stage2"] = self.stage2(maps["stage1"])
        maps["stage3"] = self.stage3(maps["stage2"])
        maps["stage4"] = self.stage4(maps["stage3"])
        return maps

    def summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"\n  🏗️  Modèle : FaceEmotionCNN (from scratch)")
        print(f"  ┌─ STEM    : Conv(3→64) × 2 + MaxPool × 2")
        print(f"  ├─ Stage 1 : Conv(64→128)  + SE + ResBlock + MaxPool")
        print(f"  ├─ Stage 2 : Conv(128→256) + SE + ResBlock × 2 + MaxPool")
        print(f"  ├─ Stage 3 : Conv(256→512) + SE + ResBlock × 2 + MaxPool")
        print(f"  ├─ Stage 4 : Conv(512→512) + SE + ResBlock")
        print(f"  └─ Head    : GAP → FC(512→256) → BN → ReLU → FC(256→7)")
        print(f"  Paramètres total    : {total:>10,}")
        print(f"  Paramètres entraîn. : {trainable:>10,}")

    # Compatibilité avec train.py (pas de backbone/head séparés)
    def get_param_groups(self, lr_backbone, lr_head):
        """Retourne 2 groupes : early stages / late stages + head."""
        early = list(self.stem.parameters()) + \
                list(self.stage1.parameters())
        late  = list(self.stage2.parameters()) + \
                list(self.stage3.parameters()) + \
                list(self.stage4.parameters()) + \
                list(self.classifier.parameters())
        return [
            {"params": early, "lr": lr_backbone * 0.5},
            {"params": late,  "lr": lr_head},
        ]

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ═══════════════════════════════════════════════════
# FACTORY & UTILS
# ═══════════════════════════════════════════════════

def build_model(**kwargs) -> FaceEmotionCNN:
    return FaceEmotionCNN(**kwargs)


def load_model(ckpt_path: str, device) -> FaceEmotionCNN:
    model = FaceEmotionCNN()
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model

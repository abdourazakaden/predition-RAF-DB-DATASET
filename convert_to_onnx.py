"""
convert_to_onnx.py
Exécutez ce script EN LOCAL pour convertir best_model.pth → best_model.onnx
Usage : python convert_to_onnx.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Modèle identique à l'entraînement ────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(ch//r, ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        r = x
        x = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(self.b2(self.c2(x)) + r, inplace=True)

class FaceEmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(
            ConvBlock(3,32), ConvBlock(32,64,pool=True), ConvBlock(64,64,pool=True))
        self.stage1 = nn.Sequential(
            ConvBlock(64,128), SEBlock(128), ResBlock(128),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.1))
        self.stage2 = nn.Sequential(
            ConvBlock(128,256), SEBlock(256), ResBlock(256),
            ResBlock(256), nn.MaxPool2d(2,2), nn.Dropout2d(0.15))
        self.stage3 = nn.Sequential(
            ConvBlock(256,512), SEBlock(512), ResBlock(512),
            ResBlock(512), nn.MaxPool2d(2,2), nn.Dropout2d(0.2))
        self.stage4 = nn.Sequential(
            ConvBlock(512,512), SEBlock(512), ResBlock(512))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(512,256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 7))
    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        return self.head(self.gap(x))

# ── Chargement ────────────────────────────────────────────────
print("📦 Chargement de best_model.pth ...")
model = FaceEmotionCNN()
ckpt  = torch.load("checkpoints/best_model.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── Export ONNX ───────────────────────────────────────────────
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "best_model.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

import os
size = os.path.getsize("best_model.onnx") / 1024 / 1024
print(f"✅ Converti : best_model.onnx ({size:.1f} MB)")
print("   → Uploadez ce fichier dans la sidebar de l'app Streamlit")

"""
app.py — RAF-DB Emotion Detection (Streamlit Cloud)
"""
import sys
import subprocess

# Installation forcée des packages si absents
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

try:
    import matplotlib
except ImportError:
    install("matplotlib")
    import matplotlib

try:
    import torch
except ImportError:
    install("torch")
    install("torchvision")
    import torch

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np

try:
    from PIL import Image
except ImportError:
    install("Pillow")
    from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Surprise", 1: "Fear",    2: "Disgust",
    3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral"
}
CLASS_EMOJIS = {
    "Surprise": "😮", "Fear": "😨", "Disgust": "🤢",
    "Happiness": "😊", "Sadness": "😢", "Anger": "😠", "Neutral": "😐",
}
IMG_SIZE    = 224
NUM_CLASSES = 7

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAF-DB — Détection d'émotions",
    page_icon="🎭",
    layout="wide",
)

st.markdown("""
<style>
    .title-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px; padding: 1.5rem;
    }
    .top3-item {
        background: rgba(255,255,255,0.08);
        border-radius: 8px; padding: 0.6rem 1rem; margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# MODÈLE CNN FROM SCRATCH
# ─────────────────────────────────────────────────────
@st.cache_resource
def build_and_load_model(ckpt_path):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import transforms

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
            def __init__(self, num_classes=7):
                super().__init__()
                self.stem   = nn.Sequential(
                    ConvBlock(3,32), ConvBlock(32,64,pool=True),
                    ConvBlock(64,64,pool=True))
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
                    nn.Linear(256, num_classes))
            def forward(self, x):
                x = self.stem(x); x = self.stage1(x); x = self.stage2(x)
                x = self.stage3(x); x = self.stage4(x)
                return self.head(self.gap(x))

        device = torch.device("cpu")
        model  = FaceEmotionCNN(NUM_CLASSES).to(device)
        ckpt   = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        return model, device, tf, None

    except Exception as e:
        return None, None, None, str(e)


def predict(image, model, device, tf):
    import torch
    tensor = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    idx   = probs.argmax()
    names = list(CLASS_NAMES.values())
    return {
        "class":      names[idx],
        "confidence": float(probs[idx]),
        "probs":      probs,
        "names":      names,
        "top3": sorted(zip(names, probs), key=lambda x: -x[1])[:3],
    }


# ─────────────────────────────────────────────────────
# UI — SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    ckpt_path = st.text_input("Checkpoint .pth", value="checkpoints/best_model.pth")
    st.markdown("---")
    st.markdown("### 🎭 Classes")
    for name, emoji in CLASS_EMOJIS.items():
        st.write(f"{emoji} {name}")

# ─────────────────────────────────────────────────────
# UI — TITRE
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="title-card">
    <h1 style="color:white;margin:0;font-size:2.2rem">
        🎭 Détection d'Émotions Faciales
    </h1>
    <p style="color:rgba(255,255,255,0.8);margin:0.5rem 0 0 0">
        RAF-DB · CNN from scratch · PyTorch
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────────────
model, device, tf, model_error = build_and_load_model(ckpt_path)

if model_error:
    st.warning(f"⚠️ Modèle non chargé : {model_error}")
    st.info("Entraînez le modèle avec `python train.py` puis déposez "
            "`best_model.pth` dans `checkpoints/`.")
    model = None
else:
    st.success("✅ Modèle chargé !")

# ─────────────────────────────────────────────────────
# UPLOAD & PRÉDICTION
# ─────────────────────────────────────────────────────
st.markdown("## 📤 Chargez une image")
tab1, tab2 = st.tabs(["📁 Upload", "📷 Caméra"])
uploaded = None
with tab1:
    uploaded = st.file_uploader("Image JPG / PNG",
                                type=["jpg","jpeg","png","bmp","webp"])
with tab2:
    cam = st.camera_input("Photo")
    if cam: uploaded = cam

if uploaded and model:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        st.markdown("### 🖼️ Image")
        st.image(image, use_column_width=True)

    with col2:
        with st.spinner("🔍 Analyse..."):
            result = predict(image, model, device, tf)

        emoji = CLASS_EMOJIS.get(result["class"], "")
        conf  = result["confidence"] * 100

        st.markdown("### 🎯 Résultat")
        st.markdown(f"""
        <div class="emotion-card" style="text-align:center">
            <div style="font-size:3rem">{emoji}</div>
            <h2 style="color:#00d4aa;margin:0">{result['class']}</h2>
            <p style="color:rgba(255,255,255,0.7)">
                Confiance : <strong style="color:#38ef7d">{conf:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(conf))

        st.markdown("### 🏆 Top 3")
        for medal, (name, prob) in zip(["🥇","🥈","🥉"], result["top3"]):
            em = CLASS_EMOJIS.get(name, "")
            p  = prob * 100
            st.markdown(f"""
            <div class="top3-item">
                <b>{medal} {em} {name}</b>
                <span style="float:right;color:#38ef7d"><b>{p:.1f}%</b></span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(p))

    st.markdown("---")
    st.markdown("### 📊 Distribution complète")
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#16213e")
    ax.set_facecolor("#16213e")
    names  = result["names"]
    probs  = result["probs"] * 100
    colors = ["#00d4aa" if n == result["class"] else "#3d5af1" for n in names]
    labels = [f"{CLASS_EMOJIS.get(n,'')} {n}" for n in names]
    bars   = ax.barh(labels, probs, color=colors, edgecolor="none", height=0.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Probabilité (%)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3d5af1")
    for bar, p in zip(bars, probs):
        ax.text(p + 0.5, bar.get_y() + bar.get_height()/2,
                f"{p:.1f}%", va="center", color="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

elif not model:
    st.markdown("---")
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, [
        ("🧠", "CNN from Scratch", "4 stages + SE Blocks + ResBlocks"),
        ("📸", "RAF-DB",           "15 000+ images, 7 expressions"),
        ("⚡", "Temps réel",       "Inférence CPU rapide"),
    ]):
        with col:
            st.markdown(f"""
            <div class="emotion-card" style="text-align:center;padding:1.5rem">
                <div style="font-size:2rem">{icon}</div>
                <b style="color:#00d4aa">{title}</b>
                <p style="font-size:0.85rem;color:rgba(255,255,255,0.6);margin:0.3rem 0 0">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

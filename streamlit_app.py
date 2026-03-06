"""
streamlit_app.py — RAF-DB Emotion Detection
Chargement du .pth directement via la sidebar (pas de chemin fixe)
"""
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="🎭 Prédiction Émotion RAF-DB",
    page_icon="🎭",
    layout="centered"
)

# ============================================================
# CLASSES
# ============================================================
CLASS_NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
CLASS_EMOJIS = {
    "Surprise": "😮", "Fear": "😨", "Disgust": "🤢",
    "Happiness": "😊", "Sadness": "😢", "Anger": "😠", "Neutral": "😐",
}
CLASS_DESC = {
    "Surprise":  "Événement inattendu, étonnement",
    "Fear":      "Peur, menace perçue",
    "Disgust":   "Répulsion, dégoût",
    "Happiness": "Joie, bonheur, sourire",
    "Sadness":   "Tristesse, douleur émotionnelle",
    "Anger":     "Colère, frustration",
    "Neutral":   "Expression neutre, calme",
}

# ============================================================
# CHARGEMENT MODÈLE depuis fichier uploadé
# ============================================================
@st.cache_resource
def load_model(file_bytes):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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

    # Charger depuis les bytes
    buffer = io.BytesIO(file_bytes)
    ckpt   = torch.load(buffer, map_location=torch.device("cpu"))
    model  = FaceEmotionCNN(7)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict(image, model):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tensor = tf(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()
    idx = int(probs.argmax())
    return {
        "class":      CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probs":      probs,
        "top3": sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])[:3],
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📁 Modèle CNN")
    pth_file = st.file_uploader("Charger best_model.pth", type=["pth", "pt"])
    st.markdown("---")
    st.header("ℹ️ À propos")
    st.write("Prédit l'**émotion faciale** à partir d'une photo.")
    st.markdown("---")
    st.write("**Dataset :** RAF-DB")
    st.write("**Modèle :** CNN from scratch")
    st.write("**Images :** 15 339")
    st.write("**Classes :** 7 émotions")
    st.markdown("---")
    st.header("🎭 Émotions supportées")
    for name in CLASS_NAMES:
        st.write(f"{CLASS_EMOJIS[name]} **{name}** — {CLASS_DESC[name]}")

# ============================================================
# TITRE
# ============================================================
st.title("🎭 Prédiction d'Émotion Faciale")
st.write("Chargez une **photo de visage** pour détecter automatiquement l'émotion exprimée.")
st.markdown("---")

# Vérification modèle
if pth_file is None:
    st.info("👈 Chargez d'abord votre fichier **best_model.pth** dans la barre latérale.")
    st.stop()

try:
    with st.spinner("⏳ Chargement du modèle..."):
        model = load_model(pth_file.read())
    st.sidebar.success("✅ Modèle chargé !")
except Exception as e:
    st.error(f"❌ Erreur chargement modèle : {e}")
    st.stop()

# ============================================================
# ÉTAPE 1 — CHARGER UNE IMAGE
# ============================================================
st.subheader("📋 Étape 1 — Chargez une image de visage")
st.write("Sélectionnez une photo contenant un visage humain :")
st.markdown("---")

tab1, tab2 = st.tabs(["📁 Upload fichier", "📷 Caméra"])
uploaded = None
with tab1:
    uploaded = st.file_uploader(
        "Image JPG / PNG / WEBP",
        type=["jpg","jpeg","png","bmp","webp"],
        label_visibility="collapsed")
with tab2:
    cam = st.camera_input("Prendre une photo")
    if cam: uploaded = cam

if uploaded is None:
    st.warning("⚠️ Aucune image chargée. Uploadez une photo pour continuer.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.markdown("---")

# ============================================================
# ÉTAPE 2 — PRÉVISUALISATION
# ============================================================
st.subheader("📋 Étape 2 — Prévisualisation de l'image")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(image, caption=f"{image.size[0]}×{image.size[1]} px",
             use_column_width=True)
st.markdown("---")

# ============================================================
# ÉTAPE 3 — INFOS IMAGE
# ============================================================
st.subheader("📋 Étape 3 — Informations sur l'image")
recap = pd.DataFrame({
    "Paramètre": ["Fichier", "Taille originale", "Mode couleur", "Taille après resize"],
    "Valeur":    [
        getattr(uploaded, "name", "caméra"),
        f"{image.size[0]} × {image.size[1]} px",
        image.mode,
        "224 × 224 px"
    ],
})
st.dataframe(recap, use_container_width=True, hide_index=True)
st.markdown("---")

# ============================================================
# BOUTON PRÉDICTION
# ============================================================
if st.button("🔍 Prédire — Quelle est l'émotion ?", use_container_width=True):

    with st.spinner("🔍 Analyse en cours..."):
        result = predict(image, model)

    pred_class = result["class"]
    pred_emoji = CLASS_EMOJIS[pred_class]
    pred_conf  = result["confidence"] * 100

    st.markdown("---")
    st.subheader("🎯 Résultat de la Prédiction")

    color_map = {"Happiness": "success", "Neutral": "success",
                 "Surprise": "info", "Sadness": "warning",
                 "Fear": "warning", "Disgust": "error", "Anger": "error"}
    level = color_map.get(pred_class, "info")
    msg   = f"## {pred_emoji} Émotion détectée : {pred_class.upper()}"

    if level == "success":   st.success(msg)
    elif level == "info":    st.info(msg)
    elif level == "warning": st.warning(msg)
    else:                    st.error(msg)

    st.write(f"💬 *{CLASS_DESC[pred_class]}*")
    st.markdown("---")

    # Top 3
    st.subheader("📊 Probabilités :")
    for medal, (name, prob) in zip(["🥇","🥈","🥉"], result["top3"]):
        emoji = CLASS_EMOJIS[name]
        p     = float(prob) * 100
        st.write(f"{medal} **{emoji} {name}** — {p:.1f}%")
        st.progress(float(prob))

    st.markdown("---")

    # Toutes les émotions
    st.subheader("📊 Toutes les émotions :")
    sorted_idx = np.argsort(result["probs"])[::-1]
    for idx in sorted_idx:
        name  = CLASS_NAMES[idx]
        emoji = CLASS_EMOJIS[name]
        prob  = float(result["probs"][idx])
        badge = " 🏆" if name == pred_class else ""
        st.write(f"**{emoji} {name}**{badge} — {prob*100:.1f}%")
        st.progress(prob)

    st.markdown("---")

    # Tableau récapitulatif
    st.subheader("📋 Récapitulatif :")
    result_df = pd.DataFrame({
        "Émotion":     [CLASS_NAMES[i] for i in sorted_idx],
        "Emoji":       [CLASS_EMOJIS[CLASS_NAMES[i]] for i in sorted_idx],
        "Probabilité": [f"{float(result['probs'][i])*100:.2f}%" for i in sorted_idx],
        "Détectée":    ["✅ OUI" if CLASS_NAMES[i] == pred_class else "" for i in sorted_idx],
    })
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("⚠️ Ce résultat est **indicatif uniquement**. Consultez un spécialiste pour une analyse approfondie.")

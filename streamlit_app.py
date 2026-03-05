import streamlit as st
import pickle
import numpy as np
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
# CLASSES & EMOJIS
# ============================================================
CLASS_NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
CLASS_EMOJIS = {
    "Surprise":  "😮",
    "Fear":      "😨",
    "Disgust":   "🤢",
    "Happiness": "😊",
    "Sadness":   "😢",
    "Anger":     "😠",
    "Neutral":   "😐",
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
# CHARGEMENT DU MODÈLE (.pkl)
# ============================================================
@st.cache_resource
def load_model(fichier_pkl):
    data = pickle.load(fichier_pkl)
    return data['model'], data['transform'], data.get('accuracy', None)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📁 Modèle CNN")
    pkl_file = st.file_uploader("Charger modele_rafdb.pkl", type="pkl")
    st.markdown("---")
    st.header("ℹ️ À propos")
    st.write("Cette application prédit l'**émotion faciale** d'une personne à partir d'une photo.")
    st.markdown("---")
    st.write("**Dataset :** RAF-DB")
    st.write("**Modèle :** CNN from scratch")
    st.write("**Images :** 15 339")
    st.write("**Classes :** 7 émotions")
    st.markdown("---")
    st.header("🎭 Émotions supportées")
    for name in CLASS_NAMES:
        emoji = CLASS_EMOJIS[name]
        st.write(f"{emoji} **{name}** — {CLASS_DESC[name]}")

# ============================================================
# TITRE
# ============================================================
st.title("🎭 Prédiction d'Émotion Faciale")
st.write("Chargez une **photo de visage** pour détecter automatiquement l'émotion exprimée.")
st.markdown("---")

# Vérification modèle
if pkl_file is None:
    st.info("👈 Chargez d'abord **modele_rafdb.pkl** dans la barre latérale.")
    st.stop()

try:
    model, transform, accuracy = load_model(pkl_file)
    acc_text = f" — Accuracy : **{accuracy:.2f}%**" if accuracy else ""
    st.sidebar.success(f"✅ Modèle chargé !{acc_text}")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
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
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

with tab2:
    cam = st.camera_input("Prendre une photo avec la caméra")
    if cam:
        uploaded = cam

if uploaded is None:
    st.warning("⚠️ Aucune image chargée. Uploadez une photo pour continuer.")
    st.stop()

# Afficher l'image
image = Image.open(uploaded).convert("RGB")
st.markdown("---")

# ============================================================
# ÉTAPE 2 — PRÉVISUALISATION
# ============================================================
st.subheader("📋 Étape 2 — Prévisualisation de l'image")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(image, caption=f"Image chargée — {image.size[0]}×{image.size[1]} px",
             use_column_width=True)

st.markdown("---")

# ============================================================
# ÉTAPE 3 — RÉCAPITULATIF IMAGE
# ============================================================
st.subheader("📋 Étape 3 — Informations sur l'image")

import pandas as pd
recap = pd.DataFrame({
    "Paramètre": ["Format",      "Taille originale",              "Mode couleur", "Taille après resize"],
    "Valeur":    [uploaded.name if hasattr(uploaded, 'name') else "caméra",
                  f"{image.size[0]} × {image.size[1]} px",
                  image.mode,
                  "224 × 224 px"],
})
st.dataframe(recap, use_container_width=True, hide_index=True)
st.markdown("---")

# ============================================================
# BOUTON PRÉDICTION
# ============================================================
if st.button("🔍 Prédire — Quelle est l'émotion ?", use_container_width=True):

    import torch

    # Prétraitement
    tensor = transform(image).unsqueeze(0)

    # Prédiction
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).squeeze().numpy()

    pred_idx   = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    pred_emoji = CLASS_EMOJIS[pred_class]
    pred_conf  = float(probs[pred_idx]) * 100

    st.markdown("---")
    st.subheader("🎯 Résultat de la Prédiction")

    # Résultat principal
    color_map = {
        "Happiness": "success",
        "Neutral":   "success",
        "Surprise":  "info",
        "Sadness":   "warning",
        "Fear":      "warning",
        "Disgust":   "error",
        "Anger":     "error",
    }
    level = color_map.get(pred_class, "info")

    if level == "success":
        st.success(f"## {pred_emoji} Émotion détectée : {pred_class.upper()}")
    elif level == "info":
        st.info(f"## {pred_emoji} Émotion détectée : {pred_class.upper()}")
    elif level == "warning":
        st.warning(f"## {pred_emoji} Émotion détectée : {pred_class.upper()}")
    else:
        st.error(f"## {pred_emoji} Émotion détectée : {pred_class.upper()}")

    st.write(f"💬 *{CLASS_DESC[pred_class]}*")
    st.markdown("---")

    # Probabilités — Top 3
    st.subheader("📊 Probabilités :")

    top3_idx = np.argsort(probs)[::-1][:3]
    medals   = ["🥇", "🥈", "🥉"]

    for i, idx in enumerate(top3_idx):
        name  = CLASS_NAMES[idx]
        emoji = CLASS_EMOJIS[name]
        prob  = float(probs[idx]) * 100
        st.write(f"{medals[i]} **{emoji} {name}** — {prob:.1f}%")
        st.progress(float(probs[idx]))

    st.markdown("---")

    # Toutes les émotions
    st.subheader("📊 Toutes les émotions :")

    sorted_idx = np.argsort(probs)[::-1]
    for idx in sorted_idx:
        name  = CLASS_NAMES[idx]
        emoji = CLASS_EMOJIS[name]
        prob  = float(probs[idx]) * 100
        badge = " 🏆" if idx == pred_idx else ""
        st.write(f"**{emoji} {name}**{badge} — {prob:.1f}%")
        st.progress(float(probs[idx]))

    st.markdown("---")

    # Récapitulatif final
    st.subheader("📋 Récapitulatif :")
    result_df = pd.DataFrame({
        "Émotion":     [CLASS_NAMES[i] for i in sorted_idx],
        "Emoji":       [CLASS_EMOJIS[CLASS_NAMES[i]] for i in sorted_idx],
        "Probabilité": [f"{float(probs[i])*100:.2f}%" for i in sorted_idx],
        "Détectée":    ["✅ OUI" if i == pred_idx else "" for i in sorted_idx],
    })
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("⚠️ Ce résultat est **indicatif uniquement**. La détection automatique d'émotions peut être imprécise selon la qualité de la photo.")

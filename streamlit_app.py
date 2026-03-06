"""
streamlit_app.py — RAF-DB Emotion Detection
Utilise ONNX Runtime — léger et compatible Streamlit Cloud gratuit
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
# CHARGEMENT MODÈLE ONNX
# ============================================================
@st.cache_resource
def load_model(file_bytes):
    import onnxruntime as ort
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    session = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    os.unlink(tmp)
    return session

def preprocess(image):
    image = image.resize((224, 224))
    arr   = np.array(image).astype(np.float32) / 255.0
    mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr   = (arr - mean) / std
    arr   = arr.transpose(2, 0, 1)[np.newaxis]
    return arr.astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(image, session):
    tensor  = preprocess(image)
    output  = session.run(None, {"input": tensor})[0][0]
    probs   = softmax(output)
    idx     = int(probs.argmax())
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
    st.header("📁 Modèle ONNX")
    onnx_file = st.file_uploader("Charger best_model.onnx", type=["onnx"])
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

if onnx_file is None:
    st.info("👈 Chargez d'abord **best_model.onnx** dans la barre latérale.")
    st.stop()

try:
    with st.spinner("⏳ Chargement du modèle..."):
        session = load_model(onnx_file.read())
    st.sidebar.success("✅ Modèle chargé !")
except Exception as e:
    st.error(f"❌ Erreur : {e}")
    st.stop()

# ============================================================
# ÉTAPE 1 — IMAGE
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
    "Valeur": [
        getattr(uploaded, "name", "caméra"),
        f"{image.size[0]} × {image.size[1]} px",
        image.mode, "224 × 224 px"
    ],
})
st.dataframe(recap, use_container_width=True, hide_index=True)
st.markdown("---")

# ============================================================
# BOUTON PRÉDICTION
# ============================================================
if st.button("🔍 Prédire — Quelle est l'émotion ?", use_container_width=True):

    with st.spinner("🔍 Analyse en cours..."):
        result = predict(image, session)

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
        p = float(prob) * 100
        st.write(f"{medal} **{CLASS_EMOJIS[name]} {name}** — {p:.1f}%")
        st.progress(float(prob))

    st.markdown("---")

    # Toutes les émotions
    st.subheader("📊 Toutes les émotions :")
    sorted_idx = np.argsort(result["probs"])[::-1]
    for idx in sorted_idx:
        name  = CLASS_NAMES[idx]
        prob  = float(result["probs"][idx])
        badge = " 🏆" if name == pred_class else ""
        st.write(f"**{CLASS_EMOJIS[name]} {name}**{badge} — {prob*100:.1f}%")
        st.progress(prob)

    st.markdown("---")

    # Tableau
    st.subheader("📋 Récapitulatif :")
    st.dataframe(pd.DataFrame({
        "Émotion":     [CLASS_NAMES[i] for i in sorted_idx],
        "Emoji":       [CLASS_EMOJIS[CLASS_NAMES[i]] for i in sorted_idx],
        "Probabilité": [f"{float(result['probs'][i])*100:.2f}%" for i in sorted_idx],
        "Détectée":    ["✅ OUI" if CLASS_NAMES[i] == pred_class else "" for i in sorted_idx],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("⚠️ Ce résultat est **indicatif uniquement**.")

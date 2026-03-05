"""
app.py
ÉTAPE 9 — Application Web avec Streamlit

Usage :
    streamlit run app.py
"""
import os
import io
import numpy as np
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

import config
from model import load_model

# ─────────────────────────────────────────────────────
# CONFIGURATION STREAMLIT
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAF-DB — Détection d'émotions",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown("""
<style>
    .main { background-color: #0f0f23; }
    .stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); }
    .title-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .emotion-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 10px; padding: 1rem;
        text-align: center; color: white;
    }
    .top3-item {
        background: rgba(255,255,255,0.08);
        border-radius: 8px; padding: 0.6rem 1rem;
        margin: 0.3rem 0; display: flex;
        justify-content: space-between; align-items: center;
    }
    .stProgress > div > div { background: linear-gradient(90deg,#667eea,#764ba2); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE (mis en cache)
# ─────────────────────────────────────────────────────
@st.cache_resource
def load_emotion_model(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = load_model(ckpt_path, device)
        return model, device, None
    except Exception as e:
        return None, device, str(e)


def get_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────────────
def predict(image: Image.Image, model, device) -> dict:
    transform = get_transform()
    tensor    = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    pred_idx = probs.argmax()
    class_names = list(config.CLASS_NAMES.values())
    return {
        "class":         class_names[pred_idx],
        "confidence":    float(probs[pred_idx]),
        "probabilities": probs,
        "class_names":   class_names,
        "top3": sorted(zip(class_names, probs), key=lambda x: -x[1])[:3],
    }


# ─────────────────────────────────────────────────────
# UI — SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    ckpt_path = st.text_input(
        "Chemin du checkpoint",
        value=config.BEST_CKPT,
        help="Chemin vers votre fichier .pth",
    )
    st.markdown("---")
    st.markdown("### 📊 Informations modèle")
    st.info(f"""
    **Architecture** : ResNet-18  
    **Classes** : {config.NUM_CLASSES} expressions  
    **Input** : {config.IMG_SIZE}×{config.IMG_SIZE} px  
    **Dataset** : RAF-DB
    """)
    st.markdown("---")
    st.markdown("### 🎭 Émotions supportées")
    for label, name in config.CLASS_NAMES.items():
        emoji = config.CLASS_EMOJIS.get(name, "•")
        st.write(f"{emoji} {name}")

    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.caption(
        "Application de reconnaissance d'expressions faciales "
        "basée sur RAF-DB et ResNet-18."
    )


# ─────────────────────────────────────────────────────
# UI — TITRE
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="title-card">
    <h1 style="color:white;margin:0;font-size:2.5rem">
        🎭 Détection d'Émotions Faciales
    </h1>
    <p style="color:rgba(255,255,255,0.8);margin:0.5rem 0 0 0;font-size:1.1rem">
        RAF-DB · ResNet-18 · PyTorch
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# UI — CHARGEMENT MODÈLE
# ─────────────────────────────────────────────────────
model, device, error = load_emotion_model(ckpt_path)
if error:
    st.error(f"❌ Impossible de charger le modèle : {error}")
    st.info("Entraînez d'abord le modèle avec `python train.py` "
            "puis vérifiez le chemin du checkpoint.")
    model = None
else:
    st.success(f"✅ Modèle chargé — Device : **{device}**")


# ─────────────────────────────────────────────────────
# UI — UPLOAD & PRÉDICTION
# ─────────────────────────────────────────────────────
st.markdown("## 📤 Chargez une image")

tab1, tab2 = st.tabs(["📁 Upload fichier", "📷 Caméra"])

uploaded = None
with tab1:
    uploaded = st.file_uploader(
        "Choisissez une image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
with tab2:
    camera_img = st.camera_input("Prendre une photo")
    if camera_img is not None:
        uploaded = camera_img

if uploaded is not None and model is not None:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1.4], gap="large")

    # ── Colonne gauche : image ──────────────────────
    with col1:
        st.markdown("### 🖼️ Image analysée")
        st.image(image, use_column_width=True,
                 caption=f"Taille originale : {image.size[0]}×{image.size[1]} px")

    # ── Colonne droite : résultats ──────────────────
    with col2:
        with st.spinner("🔍 Analyse en cours..."):
            result = predict(image, model, device)

        emoji = config.CLASS_EMOJIS.get(result["class"], "")
        conf  = result["confidence"] * 100

        # Résultat principal
        st.markdown("### 🎯 Résultat principal")
        st.markdown(f"""
        <div class="emotion-card">
            <h2 style="text-align:center;color:#00d4aa;margin:0">
                {emoji} {result['class']}
            </h2>
            <p style="text-align:center;color:rgba(255,255,255,0.7);margin:0.3rem 0">
                Confiance : <strong style="color:#38ef7d">{conf:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Jauge de confiance
        st.markdown("**Niveau de confiance**")
        color = "normal" if conf >= 70 else "off"
        st.progress(int(conf), text=f"{conf:.1f}%")

        # Top 3
        st.markdown("### 🏆 Top 3 prédictions")
        medals = ["🥇", "🥈", "🥉"]
        for medal, (name, prob) in zip(medals, result["top3"]):
            em = config.CLASS_EMOJIS.get(name, "")
            p  = prob * 100
            st.markdown(f"""
            <div class="top3-item">
                <span>{medal} {em} <strong style="color:white">{name}</strong></span>
                <span style="color:#38ef7d;font-weight:bold">{p:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(p))

    # ── Distribution complète ──────────────────────
    st.markdown("---")
    st.markdown("### 📊 Distribution complète des probabilités")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#16213e")
    ax.set_facecolor("#16213e")

    names  = result["class_names"]
    probs  = result["probabilities"] * 100
    colors = ["#00d4aa" if n == result["class"] else "#3d5af1" for n in names]
    emojis = [config.CLASS_EMOJIS.get(n, "") for n in names]
    labels = [f"{e} {n}" for e, n in zip(emojis, names)]

    bars = ax.barh(labels, probs, color=colors, edgecolor="none", height=0.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Probabilité (%)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3d5af1")
    for bar, p in zip(bars, probs):
        ax.text(p + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", color="white", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Téléchargement du résultat ──────────────────
    st.markdown("---")
    result_text = (
        f"Expression détectée : {result['class']}\n"
        f"Confiance : {conf:.2f}%\n\n"
        f"Toutes les probabilités :\n"
    )
    for name, prob in zip(names, probs):
        result_text += f"  {name}: {prob:.2f}%\n"

    st.download_button(
        "⬇️ Télécharger les résultats (.txt)",
        data=result_text,
        file_name="emotion_result.txt",
        mime="text/plain",
    )

elif uploaded is not None and model is None:
    st.warning("⚠️ Veuillez d'abord charger un modèle valide.")

else:
    # Page d'accueil sans image
    st.markdown("---")
    cols = st.columns(4)
    infos = [
        ("🎭", "7 émotions", "Surprise, Peur, Dégoût, Bonheur, Tristesse, Colère, Neutre"),
        ("🧠", "ResNet-18", "Fine-tuning sur ImageNet + RAF-DB"),
        ("📸", "Images réelles", "Dataset RAF-DB — 15 000+ images"),
        ("⚡", "Temps réel", "Inférence rapide sur CPU/GPU"),
    ]
    for col, (icon, title, desc) in zip(cols, infos):
        with col:
            st.markdown(f"""
            <div class="emotion-card" style="text-align:center;height:140px">
                <div style="font-size:2rem">{icon}</div>
                <strong style="color:#00d4aa">{title}</strong>
                <p style="font-size:0.8rem;color:rgba(255,255,255,0.6);margin:0.3rem 0 0 0">
                    {desc}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style="text-align:center;color:rgba(255,255,255,0.5);font-size:0.9rem">
        👆 Uploadez une image de visage pour détecter l'émotion
    </div>
    """, unsafe_allow_html=True)

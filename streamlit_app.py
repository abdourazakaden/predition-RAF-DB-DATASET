import streamlit as st
import pickle
import numpy as np
import pandas as pd 

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="🩺 Prédiction Diabète", 
    page_icon="🩺",
    layout="centered"
)

# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================
@st.cache_resource
def load_model(fichier_pkl): 
    data = pickle.load(fichier_pkl)
    return (
        data['models'],
        data['imputer'],
        data['scaler'],
        data['features'],
        data['best']
    )

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📁 Modèle ML")
    pkl_file = st.file_uploader("Charger modele_diabete.pkl", type="pkl")
    st.markdown("---")
    st.header("ℹ️ À propos")
    st.write("Cette application prédit si une personne est **diabétique ou non** à partir de ses données médicales.")
    st.markdown("---")
    st.write("**Dataset :** Pima Indians Diabetes")
    st.write("**Source :** Kaggle")
    st.write("**Patients :** 768")
    st.write("**Variables :** 8")

# ============================================================
# TITRE
# ============================================================
st.title("🩺 Prédiction du Diabète")
st.write("Remplissez les critères médicaux ci-dessous pour savoir si vous êtes à risque de diabète.")
st.markdown("---")

# Vérification modèle
if pkl_file is None:
    st.info("👈 Chargez d'abord **modele_diabete.pkl** dans la barre latérale.")
    st.stop()

try:
    models, imputer, scaler, features, best_name = load_model(pkl_file)
    st.sidebar.success(f"✅ Modèle chargé !\n🥇 **{best_name}**")
except Exception as e:
    st.error(f"❌ Erreur : {e}")
    st.stop()

# ============================================================
# ÉTAPE 1 — CRITÈRES MÉDICAUX
# ============================================================
st.subheader("📋 Étape 1 — Renseignez vos critères médicaux")
st.write("Répondez à chaque question en déplaçant le curseur :")
st.markdown("---")

# Question 1
st.write("**1. Quel est votre âge ?**")
age = st.slider("Âge (ans)", 18, 90, 30, label_visibility="collapsed")
st.caption(f"➜ Âge sélectionné : **{age} ans**")
st.markdown(" ")

# Question 2
st.write("**2. Combien de grossesses avez-vous eu ?** *(0 si aucune ou homme)*")
pregnancies = st.slider("Grossesses", 0, 20, 0, label_visibility="collapsed")
st.caption(f"➜ Nombre de grossesses : **{pregnancies}**")
st.markdown(" ")

# Question 3
st.write("**3. Quel est votre taux de glucose dans le sang ? (mg/dL)**")
st.caption("💡 Valeur normale à jeun : entre 70 et 100 mg/dL")
glucose = st.slider("Glucose (mg/dL)", 50, 250, 100, label_visibility="collapsed")
st.caption(f"➜ Glucose : **{glucose} mg/dL**")
st.markdown(" ")

# Question 4
st.write("**4. Quelle est votre tension artérielle diastolique ? (mmHg)**")
st.caption("💡 Valeur normale : entre 60 et 80 mmHg")
bp = st.slider("Tension (mmHg)", 30, 130, 70, label_visibility="collapsed")
st.caption(f"➜ Tension : **{bp} mmHg**")
st.markdown(" ")

# Question 5 
st.write("**5. Quelle est l'épaisseur de votre pli cutané ? (mm)**")
st.caption("💡 Mesure au niveau du triceps — valeur normale : 10 à 30 mm")
skin = st.slider("Épaisseur peau (mm)", 0, 100, 20, label_visibility="collapsed")
st.caption(f"➜ Épaisseur peau : **{skin} mm**")
st.markdown(" ")

# Question 6
st.write("**6. Quel est votre taux d'insuline sérique ? (µU/ml)**")
st.caption("💡 Valeur normale : entre 16 et 166 µU/ml — mettez 0 si inconnu")
insulin = st.slider("Insuline (µU/ml)", 0, 900, 80, label_visibility="collapsed")
st.caption(f"➜ Insuline : **{insulin} µU/ml**")
st.markdown(" ")

# Question 7
st.write("**7. Quel est votre IMC (Indice de Masse Corporelle) ?**")
st.caption("💡 IMC = Poids(kg) / Taille²(m) — Normal : 18.5 à 24.9 — Obèse : > 30")
bmi = st.slider("IMC (kg/m²)", 10.0, 70.0, 25.0, 0.1, label_visibility="collapsed")
st.caption(f"➜ IMC : **{bmi} kg/m²**")
st.markdown(" ")

# Question 8
st.write("**8. Quel est votre score de pedigree diabétique ?**")
st.caption("💡 Ce score mesure le risque génétique selon les antécédents familiaux (0.05 = faible, 2.5 = élevé)")
dpf = st.slider("Score Pedigree", 0.05, 2.50, 0.50, 0.01, label_visibility="collapsed")
st.caption(f"➜ Score pedigree : **{dpf}**")

st.markdown("---")

# ============================================================
# ÉTAPE 2 — CHOIX DU MODÈLE
# ============================================================
st.subheader("📋 Étape 2 — Choisissez le modèle de prédiction")
model_choice = st.selectbox(
    "Modèle ML :",
    list(models.keys()),
    index=list(models.keys()).index(best_name),
    help=f"Modèle recommandé : {best_name}"
)
st.caption(f"🥇 Modèle recommandé : **{best_name}** (meilleur AUC)")

st.markdown("---")

# ============================================================
# ÉTAPE 3 — RÉCAPITULATIF
# ============================================================
st.subheader("📋 Étape 3 — Récapitulatif de vos critères")

recap = pd.DataFrame({
    "Critère"  : ["Âge", "Grossesses", "Glucose", "Tension", "Épaisseur peau", "Insuline", "IMC", "Pedigree"],
    "Valeur"   : [age, pregnancies, glucose, bp, skin, insulin, bmi, dpf],
    "Unité"    : ["ans", "nb", "mg/dL", "mmHg", "mm", "µU/ml", "kg/m²", "score"],
    "Référence": ["18–80", "0–17", "70–100 (normal)", "60–80 (normal)", "10–30 (normal)", "16–166 (normal)", "18.5–24.9 (normal)", "0.05–0.5 (faible)"]
})
st.dataframe(recap, use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================
# BOUTON PRÉDICTION
# ============================================================
if st.button("🔍 Prédire — Suis-je diabétique ?", use_container_width=True):

    # Construction du vecteur
    patient     = pd.DataFrame(
        [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=features
    )
    patient_imp = imputer.transform(patient)
    patient_sc  = scaler.transform(patient_imp)

    # Prédiction
    model      = models[model_choice]
    prediction = model.predict(patient_sc)[0]
    probas     = model.predict_proba(patient_sc)[0]

    st.markdown("---")
    st.subheader("🎯 Résultat de la Prédiction")

    # Résultat principal
    if prediction == 1:
        st.error("## ⚠️ Résultat : DIABÉTIQUE")
        st.write("Ce patient présente un **risque élevé** de diabète selon le modèle.")
    else:
        st.success("## ✅ Résultat : NON DIABÉTIQUE")
        st.write("Ce patient **ne présente pas** de signe de diabète selon le modèle.")

    st.markdown("---")

    # Probabilités
    st.subheader("📊 Probabilités :")

    labels = ["✅ Non Diabétique", "⚠️ Diabétique"]
    top2   = np.argsort(probas)[::-1][:2]

    for i, idx in enumerate(top2):
        emoji = "🥇" if i == 0 else "🥈"
        st.write(f"{emoji} **{labels[idx]}** — {probas[idx]*100:.1f}%")
        st.progress(float(probas[idx]))

    st.markdown("---")

    # Avis des 3 modèles
    st.subheader("🤖 Avis des 3 modèles :")

    for nom, mod in models.items():
        p  = mod.predict(patient_sc)[0]
        pr = mod.predict_proba(patient_sc)[0]
        badge  = "🥇 " if nom == best_name else ""
        result = "⚠️ Diabétique" if p == 1 else "✅ Non Diabétique"
        st.write(f"{badge}**{nom}** → {result} ({pr[1]*100:.1f}% diabète)")
        st.progress(float(pr[1]))

    st.markdown("---")
    st.info("⚠️ Ce résultat est **indicatif uniquement**. Consultez toujours un médecin pour un diagnostic officiel.")

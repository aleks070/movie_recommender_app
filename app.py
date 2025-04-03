### --- app.py (version finale avec logo, style et placement propre) ---

import streamlit as st
import pandas as pd
from PIL import Image
from src.utils import load_data, get_user_profile
from src.backend import (
    get_item_user_recommendations,
    get_user_user_recommendations,
    get_content_based_recommendations,
    get_model_based_recommendations,
    get_content_based_on_best_rated
)
from src.sauvegarde import save_user_profile, append_user_history_as_text


# Configuration page
st.set_page_config(page_title="RecoFilms 🎬", layout="wide")


# CSS personnalisé (olive + beige + selectbox body #31333F)
st.markdown("""
    <style>
    html, body, .main {
        background-color: #515744;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f5f5dc !important;
    }
    .stButton>button {
        background-color: #f5f5dc;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-size: 1.05em;
    }
    section[data-testid="stSidebar"] {
        background-color: #202020;
    }
    section[data-testid="stSidebar"] .stButton>button {
        background-color: #f5f5dc;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        margin-top: 6px;
    }
    .stRadio>div>label {
        font-weight: bold;
        color: #f5f5dc;
    }
    .stSelectbox > div > div {
        background-color: #31333F;
        color: white;
        font-weight: bold;
        border-radius: 6px;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #f5f5dc;
        color: black;
        font-weight: bold;
    }
    .stDataFrame {
        background-color: #fff !important;
        color: #000;
    }
    .stTextInput>div>input {
        background-color: #31333F;
        color: white;
        font-weight: bold;
        border-radius: 6px;
    }
    div[data-baseweb="slider"] {
        background-color: #31333F;
        padding: 0.2em;
        border-radius: 8px;
    }
    div[data-baseweb="slider"] .css-1n76uvr {
        background-color: #f5f5dc;
    }
    div[data-baseweb="slider"] .css-14xtw13 {
        background-color: black;
    }
    img {
        pointer-events: none;
    }
    </style>
""", unsafe_allow_html=True)


# Logo à gauche, titre à droite
col1, col2 = st.columns([1, 4])
with col1:
    st.image("data/logo_recofilms.png", width=175)
with col2:
    st.markdown("""
        <h1 style='color:#f5f5dc; margin-top: 30px;'>RecoFilms</h1>
        <h3 style='color:#f5f5dc;'>Votre assistant cinéphile intelligent</h3>
    """, unsafe_allow_html=True)


# Entrée utilisateur
user_name = st.text_input("👤 Entrez votre prénom ou pseudo")


# Chargement des données
df = load_data("data/user_ratings_genres_mov.csv")


# Saisie de 3 films + notes
all_titles = df['title'].unique()
selected_titles, ratings = [], []

cols = st.columns(3)
for i in range(3):
    with cols[i]:
        title = st.selectbox(f"Film {i+1}", options=all_titles, key=f"title_{i}")
        rating = st.slider(f"Note pour {title}", 0.0, 5.0, step=0.5, key=f"rating_{i}")
        selected_titles.append(title)
        ratings.append(rating)

if len(set(selected_titles)) < 3:
    st.error("🚫 Films en double. Veuillez choisir 3 films différents.")
else:
    genres = [df[df['title'] == title]['genres'].iloc[0] for title in selected_titles]
    user_profile = get_user_profile(selected_titles, ratings, genres)

    st.markdown("### 🎯 Votre profil utilisateur")
    st.dataframe(user_profile)

    st.subheader("🔍 Méthode de recommandation")
    main_method = st.radio("Choisissez une catégorie :", ["Collaborative", "Content-Based"])

    if main_method == "Collaborative":
        sub_method = st.selectbox("Méthode collaborative :", ["Item-User", "User-User", "SVD", "KNN", "NMF"])
    else:
        sub_method = st.selectbox("Méthode content-based :", ["Moyenne des genres", "Film préféré"])

    if st.button("📽️ Générer mes recommandations"):
        if not user_name:
            st.error("❌ Le prénom est requis pour continuer.")
        else:
            with st.spinner("🔄 Génération en cours..."):
                if main_method == "Collaborative":
                    if sub_method == "Item-User":
                        recs = get_item_user_recommendations(df, user_profile)
                    elif sub_method == "User-User":
                        recs = get_user_user_recommendations(df, user_profile)
                    else:
                        recs = get_model_based_recommendations(df, user_profile, algo_name=sub_method)
                else:
                    if sub_method == "Moyenne des genres":
                        recs = get_content_based_recommendations(df, user_profile)
                    else:
                        recs = get_content_based_on_best_rated(df, user_profile)

            save_user_profile(user_profile, method=sub_method)
            append_user_history_as_text(user_name, user_profile, sub_method, recs)
            st.success("✅ Profil utilisateur sauvegardé !")

            recs.index = range(1, len(recs)+1)
            st.markdown("### ⭐ Recommandations")
            st.dataframe(recs)


# Sidebar - Historique
st.sidebar.markdown("## 🕓 Historique")
if st.sidebar.button("📜 Voir l'historique"):
    try:
        with open("data/history.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
        sessions = full_text.strip().split("-" * 50)
        for session in sessions:
            if session.strip():
                lines = session.strip().splitlines()
                user_line = next((l for l in lines if l.startswith("👤")), "Utilisateur inconnu")
                method_line = next((l for l in lines if l.startswith("⚙️")), "")
                expander_title = f"📌 {user_line[2:].strip()} ({method_line[10:].strip()})"
                with st.expander(expander_title):
                    st.markdown(f"```\n{session.strip()}\n```")
    except FileNotFoundError:
        st.warning("⚠️ Aucun historique trouvé pour le moment.")

if st.sidebar.button("🗑️ Réinitialiser l'historique"):
    import os
    if os.path.exists("data/history.txt"):
        os.remove("data/history.txt")
    if os.path.exists("data/history.csv"):
        os.remove("data/history.csv")
    st.sidebar.success("🧹 Historique réinitialisé avec succès !")
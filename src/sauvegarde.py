
import pandas as pd
import os

def save_user_profile(user_profile: pd.DataFrame, method: str, filepath: str = "data/history.csv") -> None:
    """
    Sauvegarde le profil utilisateur avec la méthode utilisée dans un fichier CSV.
    """

    # Ajout de la méthode comme colonne
    user_profile = user_profile.copy()
    user_profile['method'] = method

    # Création ou ajout au fichier CSV
    if os.path.exists(filepath):
        history_df = pd.read_csv(filepath)
        updated_df = pd.concat([history_df, user_profile], ignore_index=True)
    else:
        updated_df = user_profile

    updated_df.to_csv(filepath, index=False)


def append_user_history_as_text(user_name: str, user_profile: pd.DataFrame, method: str, recommendations: pd.DataFrame, filepath: str = "data/history.txt") -> None:
    """
    Ajoute une entrée lisible dans un fichier texte pour chaque session utilisateur.
    """
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"👤 {user_name}\n")
        f.write(f"⚙️ Méthode : {method}\n\n")
        f.write("🎬 Films notés :\n")
        for _, row in user_profile.iterrows():
            f.write(f"- {row['title']} - Note : {row['rating']}\n")

        f.write("\n⭐ Recommandations :\n")
        for _, row in recommendations.iterrows():
            f.write(f"- {row['title']} - Score : {round(row['score'], 2)}\n")

        f.write("\n" + "-" * 50 + "\n\n")

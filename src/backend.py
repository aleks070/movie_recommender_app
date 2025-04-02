# Réécriture de backend.py avec des commentaires détaillés

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from surprise import Dataset, Reader, SVD, KNNBasic, NMF
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import PredictionImpossible



def get_item_user_recommendations(df: pd.DataFrame, user_profile: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Recommande des films similaires à ceux notés par l'utilisateur,
    en utilisant la méthode de filtrage collaboratif de type Item-User.
    """

    # 1. Création de la matrice utilisateur-film (lignes : utilisateurs, colonnes : films, valeurs : notes)
    pivot = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    # 2. Calcul de la similarité cosinus entre les colonnes (films)
    similarity_matrix = cosine_similarity(pivot.T)

    # 3. Création d'un DataFrame pour retrouver les similarités par nom de film
    similarity_df = pd.DataFrame(similarity_matrix, index=pivot.columns, columns=pivot.columns)

    # 4. Initialisation d’un vecteur de scores pondérés
    scores = pd.Series(dtype=float)

    # 5. Pour chaque film noté par l'utilisateur :
    for idx, row in user_profile.iterrows():
        movie = row['title']
        rating = row['rating']

        # Si le film est dans la matrice de similarité
        if movie in similarity_df:
            # On récupère les similarités avec les autres films et on les pondère par la note donnée
            similar_movies = similarity_df[movie] * rating

            # On ajoute les scores au vecteur total
            scores = scores.add(similar_movies, fill_value=0)

    # 6. On enlève les films déjà vus par l'utilisateur
    scores = scores.drop(labels=user_profile['title'], errors='ignore')

    # 7. On trie les scores et on garde les top N
    recommendations = scores.sort_values(ascending=False).head(top_n)

    # 8. On retourne un DataFrame propre avec les résultats
    return recommendations.reset_index().rename(columns={'index': 'title', 0: 'score'})



def get_user_user_recommendations(df: pd.DataFrame, user_profile: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Recommande des films en se basant sur la similarité entre utilisateurs (User-User Collaborative Filtering),
    avec normalisation des scores par la somme des similarités.
    """

    # 1. Création de la matrice utilisateur-film
    pivot = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    # 2. Ajout de l'utilisateur temporaire
    new_user_id = "new_user"
    user_vector = pd.Series(0, index=pivot.columns)
    for _, row in user_profile.iterrows():
        user_vector[row['title']] = row['rating']
    pivot.loc[new_user_id] = user_vector

    # 3. Calcul de la similarité cosinus entre utilisateurs
    similarity_matrix = cosine_similarity(pivot)
    similarity_df = pd.DataFrame(similarity_matrix, index=pivot.index, columns=pivot.index)

    # 4. Récupération des utilisateurs similaires (hors new_user)
    similar_users = similarity_df[new_user_id].drop(index=new_user_id).sort_values(ascending=False)

    weighted_ratings = pd.Series(dtype=float)
    similarity_sums = pd.Series(dtype=float)

    # 5. Pondération et accumulation
    for user, sim in similar_users.items():
        ratings = pivot.loc[user]
        weighted_ratings = weighted_ratings.add(ratings * sim, fill_value=0)
        similarity_sums = similarity_sums.add((ratings != 0) * sim, fill_value=0)

    # 6. Normalisation
    normalized_scores = weighted_ratings / similarity_sums
    normalized_scores = normalized_scores.drop(labels=user_profile['title'], errors='ignore')

    # 7. Tri et retour des recommandations
    recommendations = normalized_scores.sort_values(ascending=False).head(top_n)
    return recommendations.reset_index().rename(columns={'index': 'title', 0: 'score'})



def get_model_based_recommendations(df: pd.DataFrame, user_profile: pd.DataFrame, algo_name: str = "SVD", top_n: int = 5) -> pd.DataFrame:
    """
    Recommande des films à l'aide d'un modèle collaboratif basé sur Surprise (SVD, KNNBasic, NMF).

    :param df: DataFrame original contenant les colonnes 'userId', 'title', 'rating'
    :param user_profile: DataFrame utilisateur avec colonnes 'title', 'rating'
    :param algo_name: Nom de l'algorithme à utiliser ("SVD", "KNN", "NMF")
    :param top_n: Nombre de recommandations à retourner
    :return: DataFrame des recommandations
    """

    # Création d'un identifiant temporaire pour l'utilisateur
    new_user_id = "new_user"

    # Préparation des données avec Surprise
    data = df[['userId', 'title', 'rating']].copy()
    
    new_rows = pd.DataFrame({
    'userId': [new_user_id] * len(user_profile),
    'title': user_profile['title'],
    'rating': user_profile['rating']
    })
    data = pd.concat([data, new_rows], ignore_index=True)

    # Utilisation du Reader pour créer un dataset Surprise
    reader = Reader(rating_scale=(0.0, 5.0))
    surprise_data = Dataset.load_from_df(data[['userId', 'title', 'rating']], reader)
    trainset = surprise_data.build_full_trainset()

    # Choix de l'algorithme
    if algo_name == "SVD":
        algo = SVD()
    elif algo_name == "KNN":
        algo = KNNBasic()
    elif algo_name == "NMF":
        algo = NMF()
    else:
        raise ValueError("Algorithme non reconnu. Utilisez 'SVD', 'KNN' ou 'NMF'.")

    # Entraînement
    algo.fit(trainset)

    # Liste des films que l'utilisateur n'a pas encore notés
    all_titles = df['title'].unique()
    unseen = [title for title in all_titles if title not in user_profile['title'].values]

    # Prédiction des notes pour tous les films non vus
    predictions = []
    for title in unseen:
        try:
            pred = algo.predict(new_user_id, title)
            predictions.append((title, pred.est))
        except PredictionImpossible:
            continue

    # Tri par note estimée
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    # Résultat sous forme de DataFrame
    return pd.DataFrame(recommendations, columns=["title", "score"])



def get_content_based_recommendations(df: pd.DataFrame, user_profile: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Recommande des films basés sur la similarité des genres entre
    les films notés par l'utilisateur et l'ensemble du catalogue.
    """

    # 1. On transforme les genres en liste (ex: "Action|Comedy" -> ["Action", "Comedy"])
    df = df.copy()
    df['genres_list'] = df['genres'].apply(lambda x: x.split('|'))

    # 2. On encode les genres sous forme de variables binaires (one-hot encoding)
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genres_list'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=df.index)
    genre_df['title'] = df['title']

    # 3. On fait la même transformation pour les films que l'utilisateur a notés
    user_profile = user_profile.copy()
    user_profile['genres_list'] = user_profile['genres'].apply(lambda x: x.split('|'))
    user_genres_matrix = mlb.transform(user_profile['genres_list'])

    # 4. On pondère les genres des films notés par les notes de l'utilisateur
    user_weights = np.array(user_profile['rating']).reshape(-1, 1)
    user_profile_vector = np.average(user_genres_matrix, axis=0, weights=user_weights.flatten())

    # 5. On calcule la similarité cosinus entre le profil utilisateur et tous les films
    similarity_scores = cosine_similarity([user_profile_vector], genre_matrix)[0]

    # 6. On crée un DataFrame contenant le titre et la similarité
    results = pd.DataFrame({
        'title': df['title'],
        'score': similarity_scores
    })

    # 7. On retire les films que l'utilisateur a déjà notés
    results = results[~results['title'].isin(user_profile['title'])]

    # 8. On retire les doublons éventuels (plusieurs lignes pour un même film)
    results = results.drop_duplicates(subset='title')

    # 9. On trie par score décroissant et retourne les top N recommandations
    return results.sort_values(by='score', ascending=False).head(top_n).reset_index(drop=True)



def get_content_based_on_best_rated(df: pd.DataFrame, user_profile: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Recommande des films basés sur le film le mieux noté par l'utilisateur,
    en comparant les genres uniquement avec ce film.
    """

    # 1. Sélection du film avec la meilleure note
    best_row = user_profile.loc[user_profile['rating'].idxmax()]
    best_title = best_row['title']
    best_genres = best_row['genres'].split('|')

    # 2. Transformation des genres en one-hot encoding (MultiLabelBinarizer)
    df = df.copy()
    df['genres_list'] = df['genres'].apply(lambda x: x.split('|'))

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genres_list'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=df.index)
    genre_df['title'] = df['title']

    # 3. Transformation du film le mieux noté en vecteur
    best_genre_vector = mlb.transform([best_genres])

    # 4. Calcul de la similarité cosinus entre ce film et tous les autres
    similarity_scores = cosine_similarity(best_genre_vector, genre_matrix)[0]

    # 5. Création du DataFrame résultat
    results = pd.DataFrame({
        'title': df['title'],
        'score': similarity_scores
    })

    # 6. Suppression des films déjà vus par l'utilisateur
    results = results[~results['title'].isin(user_profile['title'])]

    # 7. Suppression des doublons et tri
    results = results.drop_duplicates(subset='title')
    return results.sort_values(by='score', ascending=False).head(top_n).reset_index(drop=True)





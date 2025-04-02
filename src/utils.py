
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset des films avec notes et genres.
    """
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    return df

def get_user_profile(titles: list, ratings: list, genres: list) -> pd.DataFrame:
    """
    Retourne un DataFrame contenant les 3 films notés par l'utilisateur.
    """
    data = {
        'title': titles,
        'rating': ratings,
        'genres': genres
    }
    return pd.DataFrame(data)

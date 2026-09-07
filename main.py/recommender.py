import pandas as pd
import joblib
from sklearn.metrics.pairwise import euclidean_distances


# -----------------------------
# Load saved files
# -----------------------------

df = pd.read_csv("data/spotify_processed.csv", index_col=0)

X_scaled_df = pd.read_csv("data/X_scaled.csv", index_col=0)

kmeans = joblib.load("models/kmeans_model.pkl")

scaler = joblib.load("models/scaler.pkl")


# -----------------------------
# Audio features
# -----------------------------

features = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]


# -----------------------------
# Recommendation Function
# -----------------------------

def recommend_similar_songs(song_name, n=5):

    selected_song_df = df[
        df["track_name"].str.contains(
            song_name,
            case=False,
            na=False
        )
    ]

    # If song doesn't exist
    if selected_song_df.empty:
        return "Song not found in the dataset!"

    # Get first matching song
    selected_index = selected_song_df.index[0]

    selected_cluster = df.loc[selected_index, "cluster"]

    selected_track_name = df.loc[
        selected_index, "track_name"
    ]

    selected_artist = df.loc[
        selected_index, "track_artist"
    ]

    # Get songs from same cluster
    cluster_indices = df[
        (df["cluster"] == selected_cluster) &
        ~(
            (df["track_name"] == selected_track_name) &
            (df["track_artist"] == selected_artist)
        )
    ].index

    # Selected song features
    selected_features = X_scaled_df.loc[[selected_index]]

    # Features of songs in same cluster
    cluster_features = X_scaled_df.loc[cluster_indices]

    # Calculate Euclidean distance
    distances = euclidean_distances(
        selected_features,
        cluster_features
    )[0]

    # Create recommendation dataframe
    recommendations = df.loc[cluster_indices].copy()

    recommendations["distance"] = distances

    # Sort by distance
    recommendations = recommendations.sort_values(
        by="distance"
    )

    # Remove duplicate songs
    recommendations = recommendations.drop_duplicates(
        subset=["track_name", "track_artist"]
    )

    # Return top recommendations
    return recommendations[
        [
            "track_name",
            "track_artist",
            "playlist_genre",
            "distance"
        ]
    ].head(n)


# Test recommendation system

result = recommend_similar_songs("Believer", n=5)

print(result)
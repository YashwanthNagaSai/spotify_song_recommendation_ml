import streamlit as st
import pandas as pd
from recommender import recommend_similar_songs

df = pd.read_csv("data\\spotify_processed.csv", index_col=0)

st.set_page_config(
    page_title="Spotify Music Recommendation System",
    page_icon="🎵"
)

st.sidebar.title("🎵 Spotify Recommender")

st.sidebar.write(
    "This system recommends songs based on "
    "audio characteristics using K-Means clustering "
    "and Euclidean distance."
)

st.sidebar.markdown("### 🔧 Technologies")
st.sidebar.write("• Python")
st.sidebar.write("• Pandas")
st.sidebar.write("• Scikit-learn")
st.sidebar.write("• K-Means")
st.sidebar.write("• Streamlit")

st.title("🎵 Spotify Music Recommendation System")

st.write(
    "Select a song and discover the 5 most similar songs "
    "based on audio characteristics."
)

st.subheader("Select a Song")

song_names = sorted(
    df["track_name"].dropna().unique()
)

song_name = st.selectbox(
    "🎵 Select a song to get recommendations:",
    song_names
)

if st.button("🔍 Recommend"):

    if song_name.strip() == "":
        st.warning("Please enter a song name.")

    else:
        recommendations = recommend_similar_songs(song_name, n=5)

        if isinstance(recommendations, str):
            st.error(recommendations)

        else:
            st.divider()
            st.subheader(f"🎵 Songs Similar to '{song_name}'")
            st.success("Here are your top 5 recommended songs!")

            for i, (_, row) in enumerate(recommendations.iterrows(), start=1):

                with st.container(border=True):

                    st.markdown(
                        f"### #{i} 🎵 {row['track_name']}"
                    )

                    st.write(
                        f"🎤 **Artist:** {row['track_artist']}"
                    )

                    st.write(
                        f"🎼 **Genre:** {row['playlist_genre']}"
                    )

                    st.caption(
                        f"Similarity Distance: {row['distance']:.3f}"
                    )

with st.expander("🔍 How does the recommendation system work?"):

    st.write("1️⃣ The selected song's audio features are identified.")

    st.write("2️⃣ K-Means clustering identifies the song's music group.")

    st.write("3️⃣ Songs from the same cluster are selected.")

    st.write("4️⃣ Euclidean distance compares their audio characteristics.")

    st.write("5️⃣ The 5 songs with the smallest distances are recommended.")

with st.expander("📚 Project Information"):

    st.write("**Dataset:** Spotify Songs Dataset")

    st.write("**Clustering Algorithm:** K-Means")

    st.write("**Number of Clusters:** 5")

    st.write("**Similarity Method:** Euclidean Distance")

    st.write("**Audio Features:** 11")

    st.write(
        "**Audio Features Used:** Danceability, Energy, Key, "
        "Loudness, Mode, Speechiness, Acousticness, "
        "Instrumentalness, Liveness, Valence, Tempo"
    )

st.divider()

st.caption(
    "🎵 Spotify Music Recommendation System | "
    "BTech Machine Learning Project"
)
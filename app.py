# ==========================================================
# Amazon Music Cluster Predictor
# ==========================================================

# ----------------------------------------------------------
# Import Required Libraries
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Amazon Music Cluster Predictor",
    page_icon="🎵",
    layout="wide"
)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\gopin\OneDrive\Documents\Data_Science\Streamlit Training files\Amazon Music Clustering\amc\Scripts\amazon_music_clustered.csv")


df = load_data()

# ----------------------------------------------------------
# Load Trained Model
# ----------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(r"C:\Users\gopin\OneDrive\Documents\Data_Science\Streamlit Training files\Amazon Music Clustering\amc\Scripts\kmeans_model.pkl")
    scaler = joblib.load(r"C:\Users\gopin\OneDrive\Documents\Data_Science\Streamlit Training files\Amazon Music Clustering\amc\Scripts\std_scaler.pkl")


    return model, scaler

kmeans, scaler = load_model()

# ----------------------------------------------------------
# Cluster Names
# ----------------------------------------------------------

cluster_names = {

    0: "🎸 Acoustic & Soft Songs",

    1: "🎉 Dance & High-Energy Songs",

    2: "🎙 Spoken Word & Audio Stories",

    3: "🎼 Instrumental & Ambient Music"

}

# ----------------------------------------------------------
# Cluster Description
# ----------------------------------------------------------

cluster_description = {

    0:
    """
    Calm and relaxing acoustic songs with low energy,
    suitable for peaceful listening.
    """,

    1:
    """
    Energetic and upbeat songs with high danceability,
    perfect for workouts and parties.
    """,

    2:
    """
    Audio stories, narration and spoken-word content
    characterized by very high speechiness.
    """,

    3:
    """
    Instrumental, ambient and classical music
    with little or no vocals.
    """

}

# ----------------------------------------------------------
# Application Title
# ----------------------------------------------------------

st.title("🎵 Amazon Music Cluster Predictor")

st.markdown("""
Predict the music cluster of a **new song**
using its audio characteristics.

The application also recommends similar songs
from the dataset.
""")

# ----------------------------------------------------------
# Dashboard Statistics
# ----------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Songs",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Artists",
        f"{df['name_artists'].nunique():,}"
    )

with col3:

    st.metric(
        "Genres",
        f"{df['genres'].nunique():,}"
    )

with col4:

    st.metric(
        "Clusters",
        df["Cluster"].nunique()
    )

st.divider()
# ----------------------------------------------------------
# Sidebar - Song Audio Features
# ----------------------------------------------------------

st.sidebar.header("🎼 Enter Song Audio Features")

st.sidebar.markdown(
    "Adjust the audio feature values below and click **Predict Cluster**."
)

# Danceability
danceability = st.sidebar.slider(
    "Danceability",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.01
)

# Energy
energy = st.sidebar.slider(
    "Energy",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

# Loudness
loudness = st.sidebar.number_input(
    "Loudness (dB)",
    min_value=-60.0,
    max_value=5.5,
    value=-10.0,
    step=0.1
)

# Speechiness
speechiness = st.sidebar.slider(
    "Speechiness",
    0.0,
    1.0,
    0.05,
    0.01
)

# Acousticness
acousticness = st.sidebar.slider(
    "Acousticness",
    0.0,
    1.0,
    0.50,
    0.01
)

# Instrumentalness
instrumentalness = st.sidebar.slider(
    "Instrumentalness",
    0.0,
    1.0,
    0.00,
    0.01
)

# Liveness
liveness = st.sidebar.slider(
    "Liveness",
    0.0,
    1.0,
    0.20,
    0.01
)

# Valence
valence = st.sidebar.slider(
    "Valence",
    0.0,
    1.0,
    0.50,
    0.01
)

# Tempo
tempo = st.sidebar.number_input(
    "Tempo (BPM)",
    min_value=0.0,
    max_value=250.0,
    value=120.0,
    step=1.0
)

st.sidebar.divider()

# Predict Button
predict = st.sidebar.button(
    "🎯 Predict Cluster",
    use_container_width=True
)

# ----------------------------------------------------------
# Predict Music Cluster
# ----------------------------------------------------------

if predict:

    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        "danceability": [danceability],
        "energy": [energy],
        "loudness": [loudness],
        "speechiness": [speechiness],
        "acousticness": [acousticness],
        "instrumentalness": [instrumentalness],
        "liveness": [liveness],
        "valence": [valence],
        "tempo": [tempo]
    })

    # Scale the input using the saved StandardScaler
    input_scaled = scaler.transform(input_data)

    # Predict cluster using trained K-Means model
    predicted_cluster = int(kmeans.predict(input_scaled)[0])

    # Get cluster name
    cluster_name = cluster_names[predicted_cluster]

    # ----------------------------------------------------------
    # Display Prediction
    # ----------------------------------------------------------

    st.success("✅ Prediction Completed Successfully")

    st.subheader("🎯 Predicted Music Cluster")

    st.info(cluster_name)

    st.subheader("📝 Cluster Description")

    st.write(
        cluster_description[predicted_cluster]
    )

    # Display Input Values
    st.subheader("🎼 Input Audio Features")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )

 
    # ----------------------------------------------------------
    # Top 10 Recommended Songs
    # ----------------------------------------------------------
    from sklearn.metrics.pairwise import euclidean_distances

    # Filter songs from predicted cluster
    cluster_df = df[df["Cluster"] == predicted_cluster].copy()

    # Features used for clustering
    feature_cols = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]

    # Scale cluster songs
    cluster_scaled = scaler.transform(cluster_df[feature_cols])

    # Calculate Euclidean distance
    distances = euclidean_distances(input_scaled, cluster_scaled)[0]

    # Add distance column
    cluster_df["Distance"] = distances

    # Top 10 nearest songs
    recommended_songs = (
        cluster_df
        .sort_values("Distance")
        .head(10)
    )



    # Display recommendations
    st.subheader("🎼 Similar Song List")
    st.dataframe(
        recommended_songs[
            [
                "name_song",
                "name_artists",
                "genres",
                "Distance"
            ]
        ].rename(
            columns={
                "name_song": "Song",
                "name_artists": "Artist",
                "genres": "Genre",
            }
        ),
        use_container_width=True,
        hide_index=True
    )
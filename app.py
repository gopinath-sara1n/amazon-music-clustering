# ==========================================================
# 🎵 Amazon Music Cluster Predictor & Recommendation System
# ==========================================================
# Unsupervised K-Means Clustering & Content-Based Recommendation
# Author: Gopinath S
# ==========================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.decomposition import PCA

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Amazon Music Cluster Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# File Paths (Safe Cross-Platform Resolution)
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "amazon_music_clustered.csv")
MODEL_PATH = os.path.join(BASE_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ALT_SCALER_PATH = os.path.join(BASE_DIR, "std_scaler.pkl")

FEATURE_COLS = [
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

CLUSTER_NAMES = {
    0: "🎸 Acoustic & Soft Songs",
    1: "🎉 Dance & High-Energy Songs",
    2: "🎙 Spoken Word & Audio Stories",
    3: "🎼 Instrumental & Ambient Music"
}

CLUSTER_DESCRIPTIONS = {
    0: """Calm and relaxing acoustic songs with low energy, suitable for peaceful listening.""",
    1: """Energetic and upbeat songs with high danceability, perfect for workouts and parties.""",
    2: """Audio stories, narration and spoken-word content characterized by very high speechiness.""",
    3: """Instrumental, ambient and classical music with little or no vocals."""
}

CLUSTER_COLORS = {
    0: "#10B981", # Emerald
    1: "#F59E0B", # Amber
    2: "#EC4899", # Pink
    3: "#8B5CF6"  # Purple
}

# ----------------------------------------------------------
# Load Dataset & Models (Cached)
# ----------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    
    # Clean genre column formatting for display
    def clean_genre(val):
        if pd.isna(val) or val == "[]" or not str(val).strip():
            return "Uncategorized"
        cleaned = str(val).replace("[", "").replace("]", "").replace("'", "").replace('"', '').strip()
        return cleaned if cleaned else "Uncategorized"

    df["display_genres"] = df["genres"].apply(clean_genre)
    
    # Format duration in mm:ss
    def format_duration(ms):
        if pd.isna(ms):
            return "N/A"
        secs = int(ms // 1000)
        mins = secs // 60
        secs = secs % 60
        return f"{mins}:{secs:02d}"

    df["duration_fmt"] = df["duration_ms"].apply(format_duration)
    return df


@st.cache_resource
def load_model():
    scaler_file = SCALER_PATH if os.path.exists(SCALER_PATH) else ALT_SCALER_PATH
    if not os.path.exists(scaler_file):
        st.error("Scaler file not found!")
        st.stop()
    scaler = joblib.load(scaler_file)

    if not os.path.exists(MODEL_PATH):
        st.error("K-Means model file not found!")
        st.stop()
    model = joblib.load(MODEL_PATH)

    return model, scaler


@st.cache_data
def get_scaled_catalog(_df, _scaler):
    return _scaler.transform(_df[FEATURE_COLS])


@st.cache_data
def get_pca_projection(_df, _X_scaled, sample_size=3000):
    sample_df = _df.groupby("Cluster", group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 4), random_state=42)
    )
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(_X_scaled[sample_df.index])
    sample_df = sample_df.copy()
    sample_df["pca_x"] = coords[:, 0]
    sample_df["pca_y"] = coords[:, 1]
    return sample_df, pca


df = load_data()
kmeans, scaler = load_model()
X_scaled = get_scaled_catalog(df, scaler)

# ----------------------------------------------------------
# Session State for User Inputs (Allows Manual Input & Presets)
# ----------------------------------------------------------
DEFAULT_VALUES = {
    "danceability": 0.60,
    "energy": 0.50,
    "loudness": -10.0,
    "speechiness": 0.05,
    "acousticness": 0.50,
    "instrumentalness": 0.00,
    "liveness": 0.20,
    "valence": 0.50,
    "tempo": 120.0
}

for key, val in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Presets Library
PRESET_VIBES = {
    "🎯 Custom / Default": DEFAULT_VALUES,
    "🎉 High-Energy Dance & Party": {
        "danceability": 0.82, "energy": 0.85, "loudness": -5.5, "speechiness": 0.08,
        "acousticness": 0.05, "instrumentalness": 0.01, "liveness": 0.22, "valence": 0.80, "tempo": 128.0
    },
    "🎸 Acoustic & Soft Melodies": {
        "danceability": 0.50, "energy": 0.32, "loudness": -12.5, "speechiness": 0.04,
        "acousticness": 0.78, "instrumentalness": 0.01, "liveness": 0.12, "valence": 0.42, "tempo": 105.0
    },
    "🎙 Spoken Word & Audiobooks": {
        "danceability": 0.66, "energy": 0.45, "loudness": -13.5, "speechiness": 0.85,
        "acousticness": 0.60, "instrumentalness": 0.00, "liveness": 0.40, "valence": 0.58, "tempo": 100.0
    },
    "🎼 Instrumental & Ambient": {
        "danceability": 0.45, "energy": 0.25, "loudness": -16.0, "speechiness": 0.04,
        "acousticness": 0.70, "instrumentalness": 0.85, "liveness": 0.15, "valence": 0.30, "tempo": 90.0
    }
}

# ----------------------------------------------------------
# Sidebar - User Input Controls
# ----------------------------------------------------------
st.sidebar.header("🎼 Enter Song Audio Features")
st.sidebar.markdown(
    "Adjust the audio feature values below and click **Predict Cluster**."
)

# Optional Quick Preset Loader
preset_selection = st.sidebar.selectbox(
    "⚡ Quick Load Preset (Optional):",
    list(PRESET_VIBES.keys()),
    index=0
)

# Update state if a preset is selected
if "last_preset" not in st.session_state:
    st.session_state.last_preset = preset_selection

if preset_selection != st.session_state.last_preset:
    st.session_state.last_preset = preset_selection
    chosen_preset = PRESET_VIBES[preset_selection]
    for k, v in chosen_preset.items():
        st.session_state[k] = v

# Optional Library Song Selector to populate user input
with st.sidebar.expander("🔎 Or Load Features from a Real Song", expanded=False):
    search_term = st.text_input("Search Artist or Title:", placeholder="e.g. Pop Smoke, Queen, Arcade")
    if search_term.strip():
        search_hits = df[
            df["name_song"].str.contains(search_term, case=False, na=False) |
            df["name_artists"].str.contains(search_term, case=False, na=False)
        ].head(15)
        if len(search_hits) > 0:
            song_options = {
                f"{r['name_song']} - {r['name_artists']}": r["id_songs"]
                for _, r in search_hits.iterrows()
            }
            selected_song_key = st.selectbox("Pick matching track:", list(song_options.keys()))
            if st.button("Apply Song Audio Features", use_container_width=True):
                chosen_row = df[df["id_songs"] == song_options[selected_song_key]].iloc[0]
                for f in FEATURE_COLS:
                    st.session_state[f] = float(chosen_row[f])
                st.success(f"Loaded features from: {chosen_row['name_song']}")
                st.rerun()

st.sidebar.markdown("---")

# DIRECT USER INPUTS: SLIDERS & NUMERIC INPUTS
# 1. Danceability
danceability = st.sidebar.slider(
    "Danceability",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state["danceability"]),
    step=0.01,
    key="slider_danceability"
)

# 2. Energy
energy = st.sidebar.slider(
    "Energy",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state["energy"]),
    step=0.01,
    key="slider_energy"
)

# 3. Loudness (dB)
loudness = st.sidebar.number_input(
    "Loudness (dB)",
    min_value=-60.0,
    max_value=5.5,
    value=float(st.session_state["loudness"]),
    step=0.1,
    key="input_loudness"
)

# 4. Speechiness
speechiness = st.sidebar.slider(
    "Speechiness",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state["speechiness"]),
    step=0.01,
    key="slider_speechiness"
)

# 5. Acousticness
acousticness = st.sidebar.slider(
    "Acousticness",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state["acousticness"]),
    step=0.01,
    key="slider_acousticness"
)

# 6. Instrumentalness
instrumentalness = st.sidebar.slider(
    "Instrumentalness",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state["instrumentalness"]),
    step=0.01,
    key="slider_instrumentalness"
)

# 7. Liveness
liveness = st.sidebar.slider(
    "Liveness",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state["liveness"]),
    step=0.01,
    key="slider_liveness"
)

# 8. Valence
valence = st.sidebar.slider(
    "Valence",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state["valence"]),
    step=0.01,
    key="slider_valence"
)

# 9. Tempo (BPM)
tempo = st.sidebar.number_input(
    "Tempo (BPM)",
    min_value=0.0,
    max_value=250.0,
    value=float(st.session_state["tempo"]),
    step=1.0,
    key="input_tempo"
)

st.sidebar.markdown("---")

# Recommendation Options
st.sidebar.subheader("⚙️ Recommendation Settings")
metric_type = st.sidebar.selectbox(
    "Distance / Similarity Metric:",
    ["Euclidean Distance (Original)", "Cosine Similarity (Directional Fit)"],
    index=0
)
top_k = st.sidebar.slider("Number of Similar Songs:", 5, 25, 10)
exclude_duplicate_artists = st.sidebar.checkbox("Exclude duplicate artists (Diversity)", value=False)

st.sidebar.divider()

# Predict Button
predict_btn = st.sidebar.button(
    "🎯 Predict Cluster",
    type="primary",
    use_container_width=True
)

# Update state with current widget values
st.session_state["danceability"] = danceability
st.session_state["energy"] = energy
st.session_state["loudness"] = loudness
st.session_state["speechiness"] = speechiness
st.session_state["acousticness"] = acousticness
st.session_state["instrumentalness"] = instrumentalness
st.session_state["liveness"] = liveness
st.session_state["valence"] = valence
st.session_state["tempo"] = tempo

# ----------------------------------------------------------
# Main Page Header & Dashboard Statistics
# ----------------------------------------------------------
st.title("🎵 Amazon Music Cluster Predictor")

st.markdown("""
Predict the music cluster of a **new song**
using its audio characteristics.

The application also recommends similar songs
from the dataset.
""")

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
# Core Prediction & Distance Logic
# ----------------------------------------------------------
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

# Scale the input using the saved StandardScaler (with column names to avoid warnings)
input_scaled = pd.DataFrame(scaler.transform(input_data), columns=FEATURE_COLS)

# Predict cluster using trained K-Means model
predicted_cluster = int(kmeans.predict(input_scaled)[0])

# Get cluster name and description
cluster_name = CLUSTER_NAMES[predicted_cluster]
cluster_desc = CLUSTER_DESCRIPTIONS[predicted_cluster]

# Filter candidate songs from predicted cluster
cluster_df = df[df["Cluster"] == predicted_cluster].copy()
cluster_scaled = scaler.transform(cluster_df[FEATURE_COLS])

if "Cosine" in metric_type:
    sim_scores = cosine_similarity(input_scaled, cluster_scaled)[0]
    cluster_df["Distance"] = np.round(1.0 - sim_scores, 4)
    cluster_df["Match_Score"] = np.clip(sim_scores * 100, 0, 100)
    if exclude_duplicate_artists:
        cluster_df = cluster_df.drop_duplicates(subset=["name_artists"])
    recommended_songs = cluster_df.sort_values("Distance", ascending=True).head(top_k)
else:
    distances = euclidean_distances(input_scaled, cluster_scaled)[0]
    cluster_df["Distance"] = np.round(distances, 4)
    cluster_df["Match_Score"] = np.clip(100 / (1.0 + distances * 0.35), 0, 100)
    if exclude_duplicate_artists:
        cluster_df = cluster_df.drop_duplicates(subset=["name_artists"])
    recommended_songs = cluster_df.sort_values("Distance", ascending=True).head(top_k)

# ----------------------------------------------------------
# Multi-Tab Layout for Seamless Exploration
# ----------------------------------------------------------
tab_predict, tab_analytics, tab_pca, tab_method, tab_deploy = st.tabs([
    "🎯 Predictor & Similar Songs",
    "📊 Cluster Analytics & EDA",
    "🪐 2D PCA Cluster Map",
    "🔬 Model Methodology & Metrics",
    "🚀 Cloud Deployment Guide"
])

# ==========================================================
# TAB 1: Main Prediction & Similar Songs (Original + Enhanced)
# ==========================================================
with tab_predict:
    st.success("✅ Prediction Completed Successfully")

    p_col1, p_col2 = st.columns([1.2, 0.8])

    with p_col1:
        st.subheader("🎯 Predicted Music Cluster")
        st.info(f"### {cluster_name}")

        st.subheader("📝 Cluster Description")
        st.write(cluster_desc)

        # Display Input Values Table
        st.subheader("🎼 Input Audio Features")
        st.dataframe(
            input_data.style.format({
                "danceability": "{:.2f}",
                "energy": "{:.2f}",
                "loudness": "{:.2f} dB",
                "speechiness": "{:.2f}",
                "acousticness": "{:.2f}",
                "instrumentalness": "{:.2f}",
                "liveness": "{:.2f}",
                "valence": "{:.2f}",
                "tempo": "{:.1f} BPM"
            }),
            use_container_width=True,
            hide_index=True
        )

    with p_col2:
        # Audio Radar Chart comparing User Input to Cluster Centroid Average
        st.subheader("🕸️ Audio Feature Radar Profile")
        
        categories = ["Dance", "Energy", "Loudness", "Speech", "Acoustic", "Instrumental", "Liveness", "Valence", "Tempo"]
        
        def normalize_radar(vals):
            norm = [
                vals["danceability"],
                vals["energy"],
                np.clip((vals["loudness"] + 50) / 55, 0, 1),
                vals["speechiness"],
                vals["acousticness"],
                vals["instrumentalness"],
                vals["liveness"],
                vals["valence"],
                np.clip(vals["tempo"] / 220, 0, 1)
            ]
            return norm

        user_norm = normalize_radar(input_data.iloc[0].to_dict())
        cluster_mean_vals = df[df["Cluster"] == predicted_cluster][FEATURE_COLS].mean().to_dict()
        cluster_norm = normalize_radar(cluster_mean_vals)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_norm + [user_norm[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Your Input",
            line=dict(color="#1DB954", width=2.5),
            fillcolor="rgba(29, 185, 84, 0.3)"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_norm + [cluster_norm[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=f"Cluster {predicted_cluster} Avg",
            line=dict(color=CLUSTER_COLORS[predicted_cluster], width=2, dash="dash"),
            fillcolor=f"rgba(139, 92, 246, 0.15)"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                bgcolor="rgba(15, 23, 42, 0.5)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=25, r=25, t=25, b=25),
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # ----------------------------------------------------------
    # Similar Song List & Spotify Playback
    # ----------------------------------------------------------
    st.subheader("🎼 Similar Song List")
    st.caption(f"Top {top_k} nearest songs filtered from **{cluster_name}** ({len(cluster_df):,} tracks in cluster).")

    # Embedded Player for #1 Top Recommended Song
    if len(recommended_songs) > 0:
        top_song = recommended_songs.iloc[0]
        top_id = top_song["id_songs"]
        st.markdown(f"**🎧 Top Match Preview:** *{top_song['name_song']}* by *{top_song['name_artists']}*")
        st.components.v1.html(
            f'<iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/{top_id}?utm_source=generator&theme=0" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
            height=85
        )

    # Format table for display
    display_table = recommended_songs[[
        "name_song",
        "name_artists",
        "genres",
        "Distance",
        "id_songs"
    ]].copy().rename(
        columns={
            "name_song": "Song",
            "name_artists": "Artist",
            "genres": "Genre",
        }
    )
    
    display_table["Spotify Link"] = display_table["id_songs"].apply(lambda sid: f"https://open.spotify.com/track/{sid}")
    
    st.dataframe(
        display_table[["Song", "Artist", "Genre", "Distance", "Spotify Link"]],
        column_config={
            "Spotify Link": st.column_config.LinkColumn("Listen", display_text="▶️ Open Spotify")
        },
        use_container_width=True,
        hide_index=True
    )

    # Download CSV button
    csv_bytes = display_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Similar Songs as CSV",
        data=csv_bytes,
        file_name=f"amazon_music_cluster_{predicted_cluster}_recommendations.csv",
        mime="text/csv"
    )

# ==========================================================
# TAB 2: Cluster Analytics & Exploratory Data Analysis
# ==========================================================
with tab_analytics:
    st.subheader("📊 Catalog Analysis & Cluster Insights")

    eda_c1, eda_c2 = st.columns([1, 1.4])

    with eda_c1:
        # Donut Chart of Songs per Cluster
        cluster_counts = df["Cluster_Name"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Songs"]
        
        fig_donut = px.pie(
            cluster_counts,
            names="Cluster",
            values="Songs",
            hole=0.48,
            title="Song Distribution per Cluster",
            color="Cluster",
            color_discrete_map={
                "Acoustic & Melodic Music": "#10B981",
                "Energetic & Danceable Music": "#F59E0B",
                "Speech & Narration Audio": "#EC4899",
                "Instrumental & Ambient Music": "#8B5CF6"
            }
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=40, b=40)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with eda_c2:
        # Feature Barplots Across Clusters
        means_data = df.groupby("Cluster_Name")[FEATURE_COLS].mean().reset_index()
        melted = pd.melt(means_data, id_vars=["Cluster_Name"], value_vars=["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"])
        
        fig_bars = px.bar(
            melted,
            x="variable",
            y="value",
            color="Cluster_Name",
            barmode="group",
            title="Average Feature Values (0.0 to 1.0) Across Clusters",
            color_discrete_map={
                "Acoustic & Melodic Music": "#10B981",
                "Energetic & Danceable Music": "#F59E0B",
                "Speech & Narration Audio": "#EC4899",
                "Instrumental & Ambient Music": "#8B5CF6"
            }
        )
        fig_bars.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="Mean Value",
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=40, b=40)
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    st.divider()

    st.subheader("🎵 Listen to Representative Songs from Each Cluster")
    sample_cluster_id = st.radio(
        "Select Cluster to Sample:",
        [0, 1, 2, 3],
        format_func=lambda c: f"Cluster {c}: {CLUSTER_NAMES[c]}",
        horizontal=True
    )
    
    sample_tracks = df[df["Cluster"] == sample_cluster_id].sort_values("popularity_songs", ascending=False).head(4)
    cols = st.columns(4)
    for i, (_, track) in enumerate(sample_tracks.iterrows()):
        with cols[i]:
            st.markdown(f"**{track['name_song']}**")
            st.caption(f"{track['name_artists']}")
            t_id = track["id_songs"]
            st.components.v1.html(
                f'<iframe style="border-radius:8px" src="https://open.spotify.com/embed/track/{t_id}?utm_source=generator&theme=0" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
                height=85
            )

# ==========================================================
# TAB 3: 2D PCA Cluster Map
# ==========================================================
with tab_pca:
    st.subheader("🪐 2D Principal Component Analysis (PCA) Projection")
    st.markdown("""
    Dimensionality reduction of the 9 audio features into 2 principal axes.
    Your active input song is projected as the bright **Gold Star ⭐**.
    """)
    
    pca_df, pca_model = get_pca_projection(df, X_scaled, sample_size=3000)
    user_coords = pca_model.transform(input_scaled)[0]

    fig_pca = px.scatter(
        pca_df,
        x="pca_x",
        y="pca_y",
        color="Cluster_Name",
        hover_data=["name_song", "name_artists"],
        color_discrete_map={
            "Acoustic & Melodic Music": "#10B981",
            "Energetic & Danceable Music": "#F59E0B",
            "Speech & Narration Audio": "#EC4899",
            "Instrumental & Ambient Music": "#8B5CF6"
        },
        opacity=0.6,
        title=f"PCA Latent Space (Var Explained: {pca_model.explained_variance_ratio_[0]*100:.1f}% + {pca_model.explained_variance_ratio_[1]*100:.1f}%)"
    )
    
    fig_pca.add_trace(go.Scatter(
        x=[user_coords[0]],
        y=[user_coords[1]],
        mode="markers+text",
        marker=dict(symbol="star", size=20, color="#FBBF24", line=dict(color="#FFFFFF", width=2)),
        text=["⭐ Your Input"],
        textposition="top center",
        name="Your Song"
    ))
    
    fig_pca.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        xaxis=dict(title="Principal Component 1", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Principal Component 2", gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=520
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# ==========================================================
# TAB 4: Model Methodology & Evaluation Metrics
# ==========================================================
with tab_method:
    st.subheader("🔬 Machine Learning Methodology")
    
    st.markdown("""
    ### ⚙️ Pipeline Overview
    1. **Audio Feature Selection**: 9 numerical audio features extracted from Amazon Music.
    2. **StandardScaler Normalization**: $z = \\frac{x - \\mu}{\\sigma}$ applied so every audio feature contributes equally.
    3. **K-Means Clustering**: Unsupervised learning with $K=4$ clusters.
    4. **Recommendation Engine**: Euclidean Distance / Cosine Similarity within the predicted cluster.
    """)

    st.subheader("📊 Optimal Cluster Evaluation Table")
    eval_table = pd.DataFrame({
        "K": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "WCSS": [685950.97, 573286.79, 507958.11, 464185.95, 437304.11, 411371.56, 387592.12, 366813.54, 353306.28],
        "Silhouette Score": [0.2102, 0.2494, 0.2394, 0.1960, 0.1691, 0.1751, 0.1828, 0.1828, 0.1749],
        "Davies-Bouldin Index (DBI)": [1.8692, 1.5169, 1.4638, 1.6164, 1.6251, 1.5432, 1.5102, 1.4526, 1.4732]
    })
    st.dataframe(eval_table.style.highlight_max(subset=["Silhouette Score"], color="#065f46")
                                 .highlight_min(subset=["Davies-Bouldin Index (DBI)"], color="#1e3a8a"),
                 use_container_width=True, hide_index=True)

    st.markdown("""
    **Why K = 4 was selected:**
    - Although $K=3$ has a slightly higher silhouette score ($0.2494$ vs $0.2394$), $K=4$ achieves the **best Davies-Bouldin Index ($1.4638$)**.
    - $K=4$ successfully separates **Instrumental & Ambient tracks** from **Acoustic vocal songs**, creating distinct, musically coherent clusters.
    """)

# ==========================================================
# TAB 5: Cloud Deployment Guide
# ==========================================================
with tab_deploy:
    st.subheader("🚀 Cloud Deployment Guide")
    st.markdown("""
    ### Option 1: Streamlit Community Cloud (Free 1-Click Hosting)
    1. Fork or push to GitHub: `https://github.com/gopinath-sara1n/amazon-music-clustering`
    2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
    3. Click **"New App"** and select:
       - **Repository:** `gopinath-sara1n/amazon-music-clustering`
       - **Branch:** `main`
       - **Main file path:** `app.py`
    4. Click **"Deploy"**! The app will be live on a public URL.

    ---

    ### Option 2: Docker Container (Render / Hugging Face Spaces / GCP)
    ```bash
    # Build container
    docker build -t amazon-music-clustering .

    # Run container
    docker run -p 8501:8501 amazon-music-clustering
    ```
    Access at `http://localhost:8501`.
    """)
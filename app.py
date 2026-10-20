# ==============================================================================
# 🎵 Amazon Music AI Recommender & Cluster Explorer
# ==============================================================================
# Unsupervised Machine Learning Clustering (K-Means) & Content-Based Recommendation
# Author: Gopinath S
# ==============================================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Music AI Recommender & Cluster Explorer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Custom CSS for Modern Music App Theme
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(22, 30, 46, 0.95), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(29, 185, 84, 0.25);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(29, 185, 84, 0.6);
    }
    
    /* Cluster Badge */
    .cluster-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }
    .cluster-0 {
        background: linear-gradient(135deg, #10B981, #059669);
        color: #ffffff;
    }
    .cluster-1 {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: #ffffff;
    }
    .cluster-2 {
        background: linear-gradient(135deg, #EC4899, #DB2777);
        color: #ffffff;
    }
    .cluster-3 {
        background: linear-gradient(135deg, #8B5CF6, #6D28D9);
        color: #ffffff;
    }

    /* Track Recommendation Card */
    .track-card {
        background: rgba(22, 30, 46, 0.7);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .track-card:hover {
        background: rgba(30, 41, 59, 0.9);
        border-color: rgba(29, 185, 84, 0.5);
    }

    /* Highlight Score */
    .match-score {
        font-weight: 700;
        color: #1DB954;
        font-size: 1.1rem;
    }

    /* Tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Constants & File Paths (Relative & Cross-Platform Safe)
# ------------------------------------------------------------------------------
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
    0: "🎸 Acoustic & Melodic Music",
    1: "⚡ Energetic & Danceable Music",
    2: "🎙️ Speech & Narration Audio",
    3: "🎹 Instrumental & Ambient Music"
}

CLUSTER_DESCRIPTIONS = {
    0: "Calm, organic, and acoustic tracks with warm melodic textures, lower energy, and rich vocal presence.",
    1: "High-octane, rhythmic songs featuring elevated danceability, energetic beats, and loud soundscapes perfect for fitness and social events.",
    2: "High-speechiness content including audiobooks, stand-up comedy, podcasts, and spoken-word storytelling with pronounced vocal frequency.",
    3: "Atmospheric, ambient, and classical instrumental compositions with minimal or absent lyrical vocals, ideal for deep focus and relaxation."
}

CLUSTER_COLORS = {
    0: "#10B981", # Emerald
    1: "#F59E0B", # Amber
    2: "#EC4899", # Pink
    3: "#8B5CF6"  # Purple
}

# ------------------------------------------------------------------------------
# Data & Model Loaders (Cached)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading Amazon Music Dataset (95,837 tracks)...")
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset file not found at: {DATA_PATH}")
        st.stop()
    
    df = pd.read_csv(DATA_PATH)
    
    # Clean genre column for display
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
    
    # Extract release year
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    
    return df


@st.cache_resource(show_spinner="Loading K-Means Model & StandardScaler...")
def load_models():
    # Load Scaler
    scaler_file = SCALER_PATH if os.path.exists(SCALER_PATH) else ALT_SCALER_PATH
    if not os.path.exists(scaler_file):
        st.error("Scaler file not found!")
        st.stop()
    scaler = joblib.load(scaler_file)

    # Load Model
    if not os.path.exists(MODEL_PATH):
        st.error("K-Means model file not found!")
        st.stop()
    model = joblib.load(MODEL_PATH)

    return model, scaler


@st.cache_data(show_spinner="Precomputing audio feature embeddings...")
def get_scaled_features(_df, _scaler):
    return _scaler.transform(_df[FEATURE_COLS])


@st.cache_data(show_spinner="Computing PCA cluster coordinates...")
def compute_pca_sample(_df, _X_scaled, sample_size=3000):
    # Stratified sample for responsive interactive charting
    sample_df = _df.groupby("Cluster", group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 4), random_state=42)
    )
    sample_idx = sample_df.index
    sample_scaled = _X_scaled[sample_idx]
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(sample_scaled)
    
    sample_df = sample_df.copy()
    sample_df["pca_x"] = coords[:, 0]
    sample_df["pca_y"] = coords[:, 1]
    
    return sample_df, pca


# Load resources
df = load_dataset()
kmeans, scaler = load_models()
X_scaled = get_scaled_features(df, scaler)

# ------------------------------------------------------------------------------
# Sidebar Controls & Audio Synthesizer
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ea/Amazon_Music_logo.svg", width=180)
    st.title("🎛️ Sound Navigator")
    st.caption("Tune audio features or discover via the 95,837 track library.")
    
    input_mode = st.radio(
        "Select Input Method:",
        ["🔎 Search Library Track", "🎚️ Custom Audio Synthesizer"],
        index=0
    )
    
    st.divider()

    # Preset dictionaries for quick loading in Custom mode
    PRESETS = {
        "⚡ High-Energy Dance / EDM": {
            "danceability": 0.82, "energy": 0.88, "loudness": -5.2, "speechiness": 0.08,
            "acousticness": 0.06, "instrumentalness": 0.02, "liveness": 0.18, "valence": 0.78, "tempo": 128.0
        },
        "☕ Acoustic Coffeehouse Chill": {
            "danceability": 0.52, "energy": 0.32, "loudness": -12.5, "speechiness": 0.04,
            "acousticness": 0.78, "instrumentalness": 0.01, "liveness": 0.12, "valence": 0.45, "tempo": 105.0
        },
        "🧘 Deep Ambient & Focus": {
            "danceability": 0.28, "energy": 0.22, "loudness": -18.0, "speechiness": 0.04,
            "acousticness": 0.72, "instrumentalness": 0.85, "liveness": 0.11, "valence": 0.22, "tempo": 88.0
        },
        "🎙️ Spoken Word & Audiobooks": {
            "danceability": 0.65, "energy": 0.42, "loudness": -14.0, "speechiness": 0.85,
            "acousticness": 0.65, "instrumentalness": 0.00, "liveness": 0.35, "valence": 0.55, "tempo": 96.0
        },
        "🎷 Melodic Pop & Indie": {
            "danceability": 0.68, "energy": 0.65, "loudness": -7.8, "speechiness": 0.06,
            "acousticness": 0.28, "instrumentalness": 0.00, "liveness": 0.15, "valence": 0.62, "tempo": 120.0
        }
    }

    # Selected track metadata container
    selected_track_row = None

    if input_mode == "🔎 Search Library Track":
        st.subheader("🎵 Select a Song")
        
        # Popular curated recommendations dropdown
        curated_titles = [
            "Astronaut In The Ocean - Masked Wolf",
            "WITHOUT YOU - The Kid LAROI",
            "What You Know Bout Love - Pop Smoke",
            "Arcade - Duncan Laurence",
            "Goosebumps - HVME",
            "Jealous - Eyedress",
            "Buttercup - Jack Stauber",
            "Gentleman - SL",
            "Hold On - Chord Overstreet",
            "Show Me How - Men I Trust"
        ]
        
        search_query = st.text_input("Search Title or Artist:", placeholder="e.g. Queen, Pop Smoke, Arcade...")
        
        if search_query.strip():
            matches = df[
                df["name_song"].str.contains(search_query, case=False, na=False) |
                df["name_artists"].str.contains(search_query, case=False, na=False)
            ].head(40)
            
            if len(matches) > 0:
                track_options = {
                    f"{r['name_song']} - {r['name_artists']} ({r['release_year']})": r["id_songs"]
                    for _, r in matches.iterrows()
                }
                chosen_key = st.selectbox("Matching Tracks:", list(track_options.keys()))
                chosen_id = track_options[chosen_key]
                selected_track_row = df[df["id_songs"] == chosen_id].iloc[0]
            else:
                st.warning("No tracks found matching your query. Showing popular curated default.")
                selected_track_row = df[df["name_song"] == "Astronaut In The Ocean"].iloc[0]
        else:
            chosen_preset = st.selectbox("Iconic Showcase Songs:", curated_titles)
            chosen_song_name = chosen_preset.split(" - ")[0]
            subset = df[df["name_song"] == chosen_song_name]
            selected_track_row = subset.iloc[0] if len(subset) > 0 else df.iloc[0]

        # Use selected track features
        feature_values = {f: float(selected_track_row[f]) for f in FEATURE_COLS}
        
    else:
        # Custom Audio Synthesizer Mode
        st.subheader("⚡ Quick Audio Presets")
        preset_choice = st.selectbox("Load Musical Vibe:", ["Custom Sliders"] + list(PRESETS.keys()))
        
        if preset_choice != "Custom Sliders":
            initial_vals = PRESETS[preset_choice]
        else:
            initial_vals = {
                "danceability": 0.60, "energy": 0.50, "loudness": -10.0, "speechiness": 0.05,
                "acousticness": 0.50, "instrumentalness": 0.00, "liveness": 0.20, "valence": 0.50, "tempo": 120.0
            }

        st.subheader("🎼 Audio Sliders")
        with st.expander("⚡ Rhythm & Intensity", expanded=True):
            danceability = st.slider("Danceability", 0.0, 1.0, float(initial_vals["danceability"]), 0.01)
            energy = st.slider("Energy", 0.0, 1.0, float(initial_vals["energy"]), 0.01)
            loudness = st.slider("Loudness (dB)", -50.0, 5.0, float(initial_vals["loudness"]), 0.5)
            tempo = st.slider("Tempo (BPM)", 50.0, 220.0, float(initial_vals["tempo"]), 1.0)

        with st.expander("🎻 Texture & Acoustics", expanded=True):
            acousticness = st.slider("Acousticness", 0.0, 1.0, float(initial_vals["acousticness"]), 0.01)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, float(initial_vals["instrumentalness"]), 0.01)
            speechiness = st.slider("Speechiness", 0.0, 1.0, float(initial_vals["speechiness"]), 0.01)
            liveness = st.slider("Liveness", 0.0, 1.0, float(initial_vals["liveness"]), 0.01)

        with st.expander("😊 Emotional Tone", expanded=True):
            valence = st.slider("Valence (Positiveness)", 0.0, 1.0, float(initial_vals["valence"]), 0.01)

        feature_values = {
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo
        }

    st.divider()
    st.subheader("⚙️ Recommendation Filters")
    metric_choice = st.selectbox(
        "Similarity Metric:",
        ["Cosine Similarity (Directional Fit)", "Euclidean Distance (Spatial Proximity)"]
    )
    scope_choice = st.radio(
        "Candidate Scope:",
        ["Same Predicted Cluster (Coherent)", "Global Catalog (All 95K Tracks)"],
        index=0
    )
    top_n = st.slider("Number of Recommendations:", 5, 25, 10)
    exclude_same_artist = st.checkbox("Exclude Same Artist (More Variety)", value=True)

# ------------------------------------------------------------------------------
# Core Model Prediction & Distance Engine
# ------------------------------------------------------------------------------
input_df = pd.DataFrame([feature_values])
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURE_COLS)
predicted_cluster = int(kmeans.predict(input_scaled)[0])
cluster_name = CLUSTER_NAMES[predicted_cluster]
cluster_desc = CLUSTER_DESCRIPTIONS[predicted_cluster]

# Prepare candidate pool
if scope_choice == "Same Predicted Cluster (Coherent)":
    candidate_mask = (df["Cluster"] == predicted_cluster)
else:
    candidate_mask = pd.Series(True, index=df.index)

# Exclude current song if selected
if selected_track_row is not None:
    candidate_mask = candidate_mask & (df["id_songs"] != selected_track_row["id_songs"])
    if exclude_same_artist:
        candidate_mask = candidate_mask & (df["name_artists"] != selected_track_row["name_artists"])

candidate_df = df[candidate_mask].copy()
candidate_scaled = X_scaled[candidate_mask]

# Compute similarity / distance
if "Cosine" in metric_choice:
    cos_sim = cosine_similarity(input_scaled, candidate_scaled)[0]
    candidate_df["score"] = cos_sim
    # Scale to human 0-100%
    candidate_df["match_pct"] = np.clip(cos_sim * 100, 0, 100)
    recommended_df = candidate_df.sort_values("score", ascending=False).head(top_n)
else:
    euc_dist = euclidean_distances(input_scaled, candidate_scaled)[0]
    candidate_df["score"] = euc_dist
    # Convert distance to intuitive match percentage
    candidate_df["match_pct"] = np.clip(100 / (1.0 + euc_dist * 0.35), 0, 100)
    recommended_df = candidate_df.sort_values("score", ascending=True).head(top_n)

# ------------------------------------------------------------------------------
# Header & KPI Metrics
# ------------------------------------------------------------------------------
st.title("🎵 Amazon Music AI Recommender & Cluster Explorer")
st.markdown("""
Discover musical relationships using **K-Means Unsupervised Clustering** and **Content-Based Filtering**.
Trained on **95,837 tracks** across **9 continuous audio characteristics**.
""")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Tracks Analyzed", f"{len(df):,}", "Amazon Catalog")
kpi2.metric("Unique Artists", f"{df['name_artists'].nunique():,}", "Global Voices")
kpi3.metric("Acoustic Clusters", f"{df['Cluster'].nunique()}", "K = 4 Optimal")
kpi4.metric("Active Metric", "Cosine Similarity" if "Cosine" in metric_choice else "Euclidean", "Normalized")

st.divider()

# ------------------------------------------------------------------------------
# Main Application Tabs
# ------------------------------------------------------------------------------
tab_recs, tab_eda, tab_pca, tab_method, tab_about = st.tabs([
    "🎧 Song Discovery & Recommender",
    "📊 Cluster Analytics & EDA",
    "🪐 Interactive PCA Space",
    "🔬 Machine Learning Methodology",
    "ℹ️ Project Info & Deployment"
])

# ==============================================================================
# TAB 1: Song Discovery & Recommendations
# ==============================================================================
with tab_recs:
    col_input, col_pred = st.columns([1.1, 0.9])
    
    with col_input:
        if selected_track_row is not None:
            st.subheader("📌 Selected Input Track")
            
            # Spotify Embed Player
            spotify_id = selected_track_row["id_songs"]
            embed_url = f"https://open.spotify.com/embed/track/{spotify_id}?utm_source=generator&theme=0"
            st.components.v1.html(
                f'<iframe style="border-radius:12px" src="{embed_url}" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
                height=90
            )
            
            # Track details table
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.markdown(f"**Artist:** {selected_track_row['name_artists']}")
            m_col2.markdown(f"**Duration:** {selected_track_row['duration_fmt']}")
            m_col3.markdown(f"**Year:** {selected_track_row['release_year']}")
            st.caption(f"**Genres:** {selected_track_row['display_genres']}")
        else:
            st.subheader("🎛️ Active Synthesizer Features")
            st.caption("Custom audio parameters configured in the sidebar.")
            
        # Display Audio Features Bar/Values
        feat_df = pd.DataFrame([feature_values])
        st.dataframe(feat_df.style.format({
            "danceability": "{:.2f}",
            "energy": "{:.2f}",
            "loudness": "{:.1f} dB",
            "speechiness": "{:.2f}",
            "acousticness": "{:.2f}",
            "instrumentalness": "{:.2f}",
            "liveness": "{:.2f}",
            "valence": "{:.2f}",
            "tempo": "{:.0f} BPM"
        }), use_container_width=True, hide_index=True)

    with col_pred:
        st.subheader("🎯 Predicted Acoustic Cluster")
        
        # Styled cluster banner
        cluster_class = f"cluster-{predicted_cluster}"
        st.markdown(
            f'<div class="cluster-badge {cluster_class}">{cluster_name} (Cluster #{predicted_cluster})</div>',
            unsafe_allow_html=True
        )
        st.info(cluster_desc)
        
        # Audio Radar Chart
        categories = ["Dance", "Energy", "Loudness", "Speech", "Acoustic", "Instrumental", "Liveness", "Valence", "Tempo"]
        
        # Min-max normalization for balanced radar view
        def normalize_radar(vals):
            norm = []
            norm.append(vals["danceability"])
            norm.append(vals["energy"])
            norm.append(np.clip((vals["loudness"] + 50) / 55, 0, 1))
            norm.append(vals["speechiness"])
            norm.append(vals["acousticness"])
            norm.append(vals["instrumentalness"])
            norm.append(vals["liveness"])
            norm.append(vals["valence"])
            norm.append(np.clip(vals["tempo"] / 220, 0, 1))
            return norm

        input_norm = normalize_radar(feature_values)
        
        # Cluster centroid unscaled mean
        cluster_mean_raw = df[df["Cluster"] == predicted_cluster][FEATURE_COLS].mean().to_dict()
        cluster_norm = normalize_radar(cluster_mean_raw)
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=input_norm + [input_norm[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Input Track",
            line=dict(color="#1DB954", width=2.5),
            fillcolor="rgba(29, 185, 84, 0.25)"
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
                bgcolor="rgba(15, 23, 42, 0.6)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=30, t=25, b=25),
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # Recommendations Section
    st.subheader(f"✨ Top {top_n} Recommended Tracks")
    st.caption(f"Ranked via **{metric_choice.split(' ')[0]}** across **{len(candidate_df):,}** eligible candidate songs.")

    if len(recommended_df) == 0:
        st.warning("No candidate tracks matched your filters. Try selecting 'Global Catalog' or relaxing artist exclusion.")
    else:
        # Highlight #1 Top Match with an interactive embedded player
        top_match = recommended_df.iloc[0]
        top_spotify_id = top_match["id_songs"]
        
        st.markdown(f"#### 🏆 #1 Nearest Match: **{top_match['name_song']}** by **{top_match['name_artists']}** ({top_match['match_pct']:.1f}% Match)")
        top_embed_url = f"https://open.spotify.com/embed/track/{top_spotify_id}?utm_source=generator&theme=0"
        st.components.v1.html(
            f'<iframe style="border-radius:12px" src="{top_embed_url}" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
            height=90
        )
        
        # Display recommendations table
        rec_display = recommended_df[[
            "name_song",
            "name_artists",
            "display_genres",
            "release_year",
            "duration_fmt",
            "popularity_songs",
            "match_pct",
            "id_songs"
        ]].copy()
        
        rec_display["Rank"] = [f"#{i+1}" for i in range(len(rec_display))]
        rec_display["Spotify Link"] = rec_display["id_songs"].apply(lambda sid: f"https://open.spotify.com/track/{sid}")
        
        rec_table = rec_display[[
            "Rank",
            "name_song",
            "name_artists",
            "display_genres",
            "release_year",
            "duration_fmt",
            "match_pct",
            "Spotify Link"
        ]].rename(columns={
            "name_song": "Title",
            "name_artists": "Artist",
            "display_genres": "Genre",
            "release_year": "Year",
            "duration_fmt": "Duration",
            "match_pct": "Match Score (%)"
        })
        
        st.dataframe(
            rec_table,
            column_config={
                "Spotify Link": st.column_config.LinkColumn("Listen", display_text="▶️ Open Spotify"),
                "Match Score (%)": st.column_config.ProgressColumn(
                    "Match %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            },
            use_container_width=True,
            hide_index=True
        )

        # Download button for playlist
        csv_data = rec_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Recommended Playlist as CSV",
            data=csv_data,
            file_name=f"amazon_music_recommendations_cluster_{predicted_cluster}.csv",
            mime="text/csv"
        )

# ==============================================================================
# TAB 2: Cluster Analytics & EDA
# ==============================================================================
with tab_eda:
    st.subheader("📊 Catalog Composition & Cluster Distributions")
    
    eda_col1, eda_col2 = st.columns([1, 1.4])
    
    with eda_col1:
        # Donut Chart of Cluster Breakdown
        cluster_counts = df["Cluster_Name"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Track Count"]
        
        fig_donut = px.pie(
            cluster_counts,
            names="Cluster",
            values="Track Count",
            hole=0.48,
            color="Cluster",
            color_discrete_map={
                "Acoustic & Melodic Music": "#10B981",
                "Energetic & Danceable Music": "#F59E0B",
                "Speech & Narration Audio": "#EC4899",
                "Instrumental & Ambient Music": "#8B5CF6"
            },
            title="Song Distribution Across Clusters"
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=40, b=40)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with eda_col2:
        # Average Audio Features by Cluster
        means_df = df.groupby("Cluster_Name")[FEATURE_COLS].mean().reset_index()
        
        # Melt for grouped bar chart
        melted_means = pd.melt(means_df, id_vars=["Cluster_Name"], value_vars=FEATURE_COLS, var_name="Feature", value_name="Average Value")
        
        # Filter for 0-1 features for clear bar chart visual
        unit_features = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
        unit_melted = melted_means[melted_means["Feature"].isin(unit_features)]
        
        fig_bars = px.bar(
            unit_melted,
            x="Feature",
            y="Average Value",
            color="Cluster_Name",
            barmode="group",
            title="Acoustic Profile Signatures by Cluster (0.0 - 1.0)",
            color_discrete_map={
                "Acoustic & Melodic Music": "#10B981",
                "Energetic & Danceable Music": "#F59E0B",
                "Speech & Narration Audio": "#EC4899",
                "Instrumental & Ambient Music": "#8B5CF6"
            }
        )
        fig_bars.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="Average Magnitude",
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=40, b=40)
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    st.divider()

    st.subheader("🎧 Cluster Audio Showcases (Sample Songs)")
    explore_c = st.radio("Choose a Cluster to Inspect:", [0, 1, 2, 3], format_func=lambda c: f"Cluster {c}: {CLUSTER_NAMES[c]}", horizontal=True)
    
    top_cluster_tracks = df[df["Cluster"] == explore_c].sort_values("popularity_songs", ascending=False).head(5)
    
    c_cols = st.columns(5)
    for idx, (_, song_row) in enumerate(top_cluster_tracks.iterrows()):
        with c_cols[idx]:
            st.markdown(f"**{song_row['name_song']}**")
            st.caption(f"{song_row['name_artists']}")
            s_url = f"https://open.spotify.com/embed/track/{song_row['id_songs']}?utm_source=generator&theme=0"
            st.components.v1.html(
                f'<iframe style="border-radius:8px" src="{s_url}" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
                height=85
            )

# ==============================================================================
# TAB 3: Interactive PCA Space
# ==============================================================================
with tab_pca:
    st.subheader("🪐 2D Principal Component Analysis (PCA) Cluster Projection")
    st.markdown("""
    High-dimensional audio space (9 features) projected onto the top 2 orthogonal eigenvectors.
    The glowing **Gold Star ⭐** marks the location of your active input track.
    """)
    
    sample_pca_df, pca_obj = compute_pca_sample(df, X_scaled, sample_size=3200)
    
    # Project user input into PCA space
    user_pca = pca_obj.transform(input_scaled)[0]
    
    fig_pca = px.scatter(
        sample_pca_df,
        x="pca_x",
        y="pca_y",
        color="Cluster_Name",
        hover_data=["name_song", "name_artists", "display_genres"],
        color_discrete_map={
            "Acoustic & Melodic Music": "#10B981",
            "Energetic & Danceable Music": "#F59E0B",
            "Speech & Narration Audio": "#EC4899",
            "Instrumental & Ambient Music": "#8B5CF6"
        },
        opacity=0.65,
        title=f"PCA Latent Space (Explained Variance: {pca_obj.explained_variance_ratio_[0]*100:.1f}% + {pca_obj.explained_variance_ratio_[1]*100:.1f}%)"
    )
    
    # Add active user track marker
    fig_pca.add_trace(go.Scatter(
        x=[user_pca[0]],
        y=[user_pca[1]],
        mode="markers+text",
        marker=dict(symbol="star", size=20, color="#FBBF24", line=dict(color="#FFFFFF", width=2)),
        text=["⭐ Active Track"],
        textposition="top center",
        name="Your Track"
    ))
    
    fig_pca.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        xaxis=dict(title="Principal Component 1", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Principal Component 2", gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=550
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# ==============================================================================
# TAB 4: ML Methodology & Evaluation Metrics
# ==============================================================================
with tab_method:
    st.subheader("🔬 Machine Learning Methodology & Architecture")
    
    st.markdown("""
    ### ⚙️ End-to-End Pipeline
    ```
    Audio Dataset (95,837 tracks)
                 │
                 ▼
    Audio Feature Extraction (9 Features: Danceability, Energy, Loudness, etc.)
                 │
                 ▼
    Feature Standardization via StandardScaler: z = (x - μ) / σ
                 │
                 ▼
    Hyperparameter Tuning & Cluster Optimization (Elbow WCSS, Silhouette, DBI)
                 │
                 ▼
    K-Means Algorithm (Optimal K = 4)
                 │
                 ▼
    Cluster Profiling & Semantic Labeling (Acoustic, Dance, Speech, Instrumental)
                 │
                 ▼
    Content-Based Filtering via Cosine Similarity / Euclidean Distance
    ```
    """)
    
    st.subheader("📈 Clustering Hyperparameter Evaluation")
    
    eval_data = {
        "K (Clusters)": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "WCSS (Inertia)": [685950.97, 573286.79, 507958.11, 464185.95, 437304.11, 411371.56, 387592.12, 366813.54, 353306.28],
        "Silhouette Score": [0.2102, 0.2494, 0.2394, 0.1960, 0.1691, 0.1751, 0.1828, 0.1828, 0.1749],
        "Davies-Bouldin Index (DBI)": [1.8692, 1.5169, 1.4638, 1.6164, 1.6251, 1.5432, 1.5102, 1.4526, 1.4732]
    }
    eval_df = pd.DataFrame(eval_data)
    
    st.dataframe(eval_df.style.highlight_max(subset=["Silhouette Score"], color="#065f46")
                             .highlight_min(subset=["Davies-Bouldin Index (DBI)"], color="#1e3a8a"),
                 use_container_width=True, hide_index=True)
    
    st.markdown("""
    #### 💡 Rationale for Selecting K = 4:
    * **Silhouette Score:** While $K=3$ achieves a marginally higher silhouette score ($0.2494$ vs $0.2394$), $K=4$ cleanly segregates **Instrumental tracks** from **Acoustic vocal tracks**, which are functionally and musically distinct listening genres.
    * **Davies-Bouldin Index:** $K=4$ achieves the lowest DBI among primary candidates ($1.4638$), demonstrating tighter cluster cohesion and superior separation.
    * **Content Recommendation Quality:** 4 clusters provide clear semantic boundaries (Acoustic, Dance, Speech, Ambient) preventing podcast spoken word from contaminating dance playlists.
    """)

# ==============================================================================
# TAB 5: Project Info & Cloud Deployment
# ==============================================================================
with tab_about:
    st.subheader("ℹ️ About the Project")
    st.markdown("""
    **Amazon Music Clustering & Content-Based Recommendation System**
    
    * **Developer:** Gopinath S (B.Tech Leather Technology | Data Science & ML Enthusiast)
    * **GitHub Repository:** [gopinath-sara1n/amazon-music-clustering](https://github.com/gopinath-sara1n/amazon-music-clustering)
    * **LinkedIn:** [linkedin.com/in/gopinaths](https://www.linkedin.com/in/gopinaths/)
    * **License:** MIT License
    
    ---
    
    ### 🚀 Deployment Options
    
    #### 1. Streamlit Community Cloud (Recommended - Free 1-Click Hosting)
    1. Fork or push this repository to GitHub: `https://github.com/gopinath-sara1n/amazon-music-clustering`
    2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
    3. Click **"New App"** and select:
       * **Repository:** `gopinath-sara1n/amazon-music-clustering`
       * **Branch:** `main`
       * **Main file path:** `app.py`
    4. Click **"Deploy"**! The app will be live with a permanent public HTTPS URL.
    
    #### 2. Docker Container Deployment (Hugging Face Spaces / Render / GCP)
    Run the bundled `Dockerfile`:
    ```bash
    # Build Docker image
    docker build -t amazon-music-clustering .
    
    # Run container locally
    docker run -p 8501:8501 amazon-music-clustering
    ```
    Access at `http://localhost:8501`.
    """)
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
# Custom Modern CSS (Polished Music Streaming Aesthetic)
# ----------------------------------------------------------
st.markdown("""
<style>
    /* Global Card & Container Styling */
    .stApp {
        background-color: #0b0f19;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #131b2e 0%, #0f172a 100%);
        border: 1px solid rgba(29, 185, 84, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(29, 185, 84, 0.5);
    }
    
    /* Cluster Badges */
    .cluster-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 10px;
    }
    .badge-0 { background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
    .badge-1 { background-color: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
    .badge-2 { background-color: rgba(236, 72, 153, 0.2); color: #F472B6; border: 1px solid #EC4899; }
    .badge-3 { background-color: rgba(139, 92, 246, 0.2); color: #A78BFA; border: 1px solid #8B5CF6; }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

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
    0: "Calm and relaxing acoustic songs with warm melodic textures and low energy, suitable for peaceful listening.",
    1: "Energetic and upbeat songs with high danceability, fast tempo, and loud rhythm, perfect for workouts and parties.",
    2: "Audio stories, narration, podcasts, and spoken-word content characterized by dominant speechiness.",
    3: "Instrumental, ambient, and classical soundscapes with little to no vocals, ideal for deep focus and sleep."
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
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
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
    sample_df = _df.sample(frac=1.0, random_state=42).groupby("Cluster").head(sample_size // 4)
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
# Preset Profiles & State Initialization
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
    "🎙 Spoken Word & Audio Stories": {
        "danceability": 0.66, "energy": 0.45, "loudness": -13.5, "speechiness": 0.85,
        "acousticness": 0.60, "instrumentalness": 0.00, "liveness": 0.40, "valence": 0.58, "tempo": 100.0
    },
    "🎼 Instrumental & Ambient": {
        "danceability": 0.45, "energy": 0.25, "loudness": -16.0, "speechiness": 0.04,
        "acousticness": 0.70, "instrumentalness": 0.85, "liveness": 0.15, "valence": 0.30, "tempo": 90.0
    }
}

# Callback to reliably apply preset values directly to slider widget state
def on_preset_change():
    selected_preset = st.session_state.get("preset_selector")
    if selected_preset in PRESET_VIBES:
        for feat_name, feat_val in PRESET_VIBES[selected_preset].items():
            st.session_state[feat_name] = float(feat_val)
        st.session_state["loaded_song_info"] = None

# ----------------------------------------------------------
# Sidebar - User Input Audio Features
# ----------------------------------------------------------
st.sidebar.header("🎼 Enter Song Audio Features")
st.sidebar.markdown(
    "Adjust the audio features below to see instant cluster predictions and similar song recommendations in real time."
)

# Preset Selector with on_change callback
preset_selection = st.sidebar.selectbox(
    "⚡ Quick Load Preset (Optional):",
    list(PRESET_VIBES.keys()),
    key="preset_selector",
    on_change=on_preset_change
)

# Song Name Search & Audio Feature Loader
search_song_term = st.sidebar.text_input(
    "🔎 Or Search Song Name to Load Values:",
    placeholder="e.g. Dior, Arcade, Queen...",
    key="search_song_input"
)

if search_song_term.strip():
    search_hits = df[
        df["name_song"].str.contains(search_song_term, case=False, na=False) |
        df["name_artists"].str.contains(search_song_term, case=False, na=False)
    ].head(25)
    
    if len(search_hits) > 0:
        track_map = {
            f"🎵 {r['name_song']} - {r['name_artists']} ({r['release_year']})": r["id_songs"]
            for _, r in search_hits.iterrows()
        }
        chosen_track_key = st.sidebar.selectbox(
            "Select Matching Song:",
            list(track_map.keys()),
            key="matching_song_select"
        )
        if st.sidebar.button("📥 Load This Song's Audio Values", use_container_width=True, key="btn_load_song_values"):
            chosen_id = track_map[chosen_track_key]
            chosen_row = df[df["id_songs"] == chosen_id].iloc[0]
            for feat in FEATURE_COLS:
                st.session_state[feat] = float(chosen_row[feat])
            st.session_state["loaded_song_info"] = f"{chosen_row['name_song']} by {chosen_row['name_artists']}"
            st.rerun()
    else:
        st.sidebar.caption("⚠️ No songs found matching that query.")

if st.session_state.get("loaded_song_info"):
    st.sidebar.success(f"🎵 Active Song: **{st.session_state['loaded_song_info']}**")

st.sidebar.markdown("---")

# DIRECT AUDIO CONTROLS (Bound directly to session_state keys)
danceability = st.sidebar.slider(
    "Danceability",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="danceability"
)

energy = st.sidebar.slider(
    "Energy",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="energy"
)

loudness = st.sidebar.number_input(
    "Loudness (dB)",
    min_value=-60.0,
    max_value=5.5,
    step=0.1,
    key="loudness"
)

speechiness = st.sidebar.slider(
    "Speechiness",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="speechiness"
)

acousticness = st.sidebar.slider(
    "Acousticness",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="acousticness"
)

instrumentalness = st.sidebar.slider(
    "Instrumentalness",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="instrumentalness"
)

liveness = st.sidebar.slider(
    "Liveness",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="liveness"
)

valence = st.sidebar.slider(
    "Valence",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="valence"
)

tempo = st.sidebar.number_input(
    "Tempo (BPM)",
    min_value=0.0,
    max_value=250.0,
    step=1.0,
    key="tempo"
)

st.sidebar.markdown("---")

# Recommendation Options
st.sidebar.subheader("⚙️ Settings")
metric_type = st.sidebar.selectbox(
    "Distance Metric:",
    ["Euclidean Distance", "Cosine Similarity"],
    index=0
)
top_k = st.sidebar.slider("Number of Recommendations:", 5, 25, 10)
exclude_duplicate_artists = st.sidebar.checkbox("Exclude duplicate artists", value=False)


# ----------------------------------------------------------
# Main Page Header & Dashboard Statistics
# ----------------------------------------------------------
st.title("🎵 Amazon Music Cluster Predictor")

st.markdown("""
Predict the music cluster of a **new song** using its audio characteristics, 
recommend similar songs, or **browse tracks by category**.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Songs", f"{len(df):,}")
with col2:
    st.metric("Artists", f"{df['name_artists'].nunique():,}")
with col3:
    st.metric("Genres", f"{df['genres'].nunique():,}")
with col4:
    st.metric("Clusters", df["Cluster"].nunique())

st.divider()

# ----------------------------------------------------------
# Prediction & Similarity Logic
# ----------------------------------------------------------
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

input_scaled = pd.DataFrame(scaler.transform(input_data), columns=FEATURE_COLS)
predicted_cluster = int(kmeans.predict(input_scaled)[0])
cluster_name = CLUSTER_NAMES[predicted_cluster]
cluster_desc = CLUSTER_DESCRIPTIONS[predicted_cluster]

# Candidate pool from predicted cluster
cluster_df = df[df["Cluster"] == predicted_cluster].copy()
cluster_scaled = scaler.transform(cluster_df[FEATURE_COLS])

if "Cosine" in metric_type:
    sim_scores = cosine_similarity(input_scaled, cluster_scaled)[0]
    cluster_df["Distance"] = np.round(1.0 - sim_scores, 4)
    if exclude_duplicate_artists:
        cluster_df = cluster_df.drop_duplicates(subset=["name_artists"])
    recommended_songs = cluster_df.sort_values("Distance", ascending=True).head(top_k)
else:
    distances = euclidean_distances(input_scaled, cluster_scaled)[0]
    cluster_df["Distance"] = np.round(distances, 4)
    if exclude_duplicate_artists:
        cluster_df = cluster_df.drop_duplicates(subset=["name_artists"])
    recommended_songs = cluster_df.sort_values("Distance", ascending=True).head(top_k)

# ----------------------------------------------------------
# Main Navigation Tabs (Cleaned & Focused)
# ----------------------------------------------------------
tab_predict, tab_category_browse, tab_analytics, tab_pca, tab_method = st.tabs([
    "🎯 Predict & Similar Songs",
    "📻 Browse Songs by Category",
    "📊 Cluster Analytics",
    "🪐 2D PCA Cluster Map",
    "🔬 Model Methodology"
])

# ==========================================================
# TAB 1: Prediction & Similar Songs
# ==========================================================
with tab_predict:
    st.success("✅ Prediction Completed Successfully")

    p_col1, p_col2 = st.columns([1.1, 0.9])

    with p_col1:
        st.subheader("🎯 Predicted Music Cluster")
        badge_class = f"badge-{predicted_cluster}"
        st.markdown(f'<div class="cluster-badge {badge_class}">{cluster_name}</div>', unsafe_allow_html=True)

        st.subheader("📝 Cluster Description")
        st.write(cluster_desc)

        # Input Audio Features Table
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
        st.subheader("🕸️ Audio Feature Radar Profile")
        
        categories = ["Dance", "Energy", "Loudness", "Speech", "Acoustic", "Instrumental", "Liveness", "Valence", "Tempo"]
        
        def normalize_radar(vals):
            return [
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
            name=f"Cluster Avg",
            line=dict(color=CLUSTER_COLORS[predicted_cluster], width=2, dash="dash"),
            fillcolor=f"rgba(139, 92, 246, 0.15)"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                bgcolor="rgba(19, 27, 46, 0.6)"
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

    # Similar Song List
    st.subheader("🎼 Similar Song List")
    st.caption(f"Top {top_k} nearest songs filtered from **{cluster_name}** ({len(cluster_df):,} tracks in cluster).")

    # Spotlight Player for Top 1 Recommendation
    if len(recommended_songs) > 0:
        top_song = recommended_songs.iloc[0]
        top_id = top_song["id_songs"]
        st.markdown(f"**🎧 Top Match Preview:** *{top_song['name_song']}* by *{top_song['name_artists']}*")
        st.components.v1.html(
            f'<iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/{top_id}?utm_source=generator&theme=0" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
            height=85
        )

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

    csv_bytes = display_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Similar Songs as CSV",
        data=csv_bytes,
        file_name=f"amazon_music_cluster_{predicted_cluster}_recommendations.csv",
        mime="text/csv"
    )

# ==========================================================
# TAB 2: Browse Songs by Category
# ==========================================================
with tab_category_browse:
    st.subheader("📻 Browse Songs by Category")
    st.markdown("Select a music category to explore all tracks, listen to audio previews, and filter by genre or popularity.")

    cat_c1, cat_c2 = st.columns([1.4, 2.6])
    
    with cat_c1:
        chosen_cat_id = st.selectbox(
            "Select Music Category:",
            options=[0, 1, 2, 3],
            format_func=lambda c: f"{CLUSTER_NAMES[c]} ({len(df[df['Cluster'] == c]):,} tracks)"
        )
    
    with cat_c2:
        cat_subset = df[df["Cluster"] == chosen_cat_id]
        m1, m2, m3 = st.columns(3)
        m1.metric("Category Songs", f"{len(cat_subset):,}")
        m2.metric("Unique Artists", f"{cat_subset['name_artists'].nunique():,}")
        top_genre_label = cat_subset['display_genres'].value_counts().index[0] if len(cat_subset) > 0 else "N/A"
        m3.metric("Leading Genre", top_genre_label[:20] + "..." if len(top_genre_label) > 20 else top_genre_label)

    st.info(f"**Musical Profile:** {CLUSTER_DESCRIPTIONS[chosen_cat_id]}")

    # Search & Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.2, 1.2, 0.8])
    with f_col1:
        cat_search_term = st.text_input(
            "Search in Category (Title/Artist):",
            placeholder="e.g. Pop Smoke, Queen, Mistinguett...",
            key="cat_search_input"
        )
    with f_col2:
        top_cluster_genres = ["All Genres"] + [g for g in cat_subset["display_genres"].value_counts().head(35).index if g != "Uncategorized"]
        selected_genre_filter = st.selectbox("Filter Genre:", top_cluster_genres, key="cat_genre_select")
    with f_col3:
        cat_sort_order = st.selectbox(
            "Sort Order:",
            ["Most Popular", "Least Popular", "Newest", "Oldest", "Highest Danceability", "Highest Energy", "Highest Acousticness"],
            key="cat_sort_select"
        )
    with f_col4:
        cat_page_size = st.selectbox("Show Songs:", [10, 25, 50, 100], index=1, key="cat_page_limit")

    # Apply Filters
    cat_filtered = cat_subset.copy()
    if cat_search_term.strip():
        cat_filtered = cat_filtered[
            cat_filtered["name_song"].str.contains(cat_search_term, case=False, na=False) |
            cat_filtered["name_artists"].str.contains(cat_search_term, case=False, na=False)
        ]
    if selected_genre_filter != "All Genres":
        cat_filtered = cat_filtered[cat_filtered["display_genres"].str.contains(selected_genre_filter, case=False, na=False)]

    if cat_sort_order == "Most Popular":
        cat_filtered = cat_filtered.sort_values("popularity_songs", ascending=False)
    elif cat_sort_order == "Least Popular":
        cat_filtered = cat_filtered.sort_values("popularity_songs", ascending=True)
    elif cat_sort_order == "Newest":
        cat_filtered = cat_filtered.sort_values("release_year", ascending=False)
    elif cat_sort_order == "Oldest":
        cat_filtered = cat_filtered.sort_values("release_year", ascending=True)
    elif cat_sort_order == "Highest Danceability":
        cat_filtered = cat_filtered.sort_values("danceability", ascending=False)
    elif cat_sort_order == "Highest Energy":
        cat_filtered = cat_filtered.sort_values("energy", ascending=False)
    elif cat_sort_order == "Highest Acousticness":
        cat_filtered = cat_filtered.sort_values("acousticness", ascending=False)

    total_cat_matches = len(cat_filtered)
    results_display = cat_filtered.head(cat_page_size)

    st.markdown(f"**Found {total_cat_matches:,} songs** in {CLUSTER_NAMES[chosen_cat_id]} (Displaying top {len(results_display)}):")

    if total_cat_matches > 0:
        preview_map = {
            f"🎵 {r['name_song']} - {r['name_artists']} (Popularity: {r['popularity_songs']})": r["id_songs"]
            for _, r in results_display.iterrows()
        }
        chosen_preview_label = st.selectbox("🎧 Select a Song to Listen with Spotify:", list(preview_map.keys()), key="cat_preview_player")
        chosen_preview_track_id = preview_map[chosen_preview_label]
        st.components.v1.html(
            f'<iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/{chosen_preview_track_id}?utm_source=generator&theme=0" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
            height=85
        )

        browse_table = results_display[[
            "name_song",
            "name_artists",
            "display_genres",
            "release_year",
            "duration_fmt",
            "popularity_songs",
            "id_songs"
        ]].copy().rename(columns={
            "name_song": "Song",
            "name_artists": "Artist",
            "display_genres": "Genre",
            "release_year": "Year",
            "duration_fmt": "Duration",
            "popularity_songs": "Popularity"
        })
        browse_table["Spotify Link"] = browse_table["id_songs"].apply(lambda sid: f"https://open.spotify.com/track/{sid}")
        browse_table.insert(0, "#", range(1, len(browse_table) + 1))

        st.dataframe(
            browse_table[["#", "Song", "Artist", "Genre", "Year", "Duration", "Popularity", "Spotify Link"]],
            column_config={
                "Spotify Link": st.column_config.LinkColumn("Listen", display_text="▶️ Open Spotify"),
                "Popularity": st.column_config.ProgressColumn("Popularity", format="%d", min_value=0, max_value=100)
            },
            use_container_width=True,
            hide_index=True
        )

        cat_csv = browse_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {CLUSTER_NAMES[chosen_cat_id]} Songs as CSV",
            data=cat_csv,
            file_name=f"amazon_music_category_{chosen_cat_id}_songs.csv",
            mime="text/csv"
        )
    else:
        st.warning("No songs matched your query in this category. Try adjusting your search query or genre filter.")

# ==========================================================
# TAB 3: Cluster Analytics
# ==========================================================
with tab_analytics:
    st.subheader("📊 Catalog Composition & Audio Signatures")

    eda_c1, eda_c2 = st.columns([1, 1.4])

    with eda_c1:
        cluster_counts = df["Cluster_Name"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Songs"]
        
        fig_donut = px.pie(
            cluster_counts,
            names="Cluster",
            values="Songs",
            hole=0.48,
            title="Song Distribution Across Categories",
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
        means_data = df.groupby("Cluster_Name")[FEATURE_COLS].mean().reset_index()
        melted = pd.melt(means_data, id_vars=["Cluster_Name"], value_vars=["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"])
        
        fig_bars = px.bar(
            melted,
            x="variable",
            y="value",
            color="Cluster_Name",
            barmode="group",
            title="Average Audio Characteristics (0.0 to 1.0)",
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

    st.markdown("---")
    st.subheader("📋 Cluster Comparison Summary")
    summary_table = df.groupby("Cluster_Name")[FEATURE_COLS].mean().round(3)
    st.dataframe(summary_table, use_container_width=True)

# ==========================================================
# TAB 4: 2D PCA Cluster Map
# ==========================================================
with tab_pca:
    st.subheader("🪐 2D Principal Component Analysis (PCA) Projection")
    st.markdown("""
    Dimensionality reduction of the 9 audio features into 2 principal axes.
    Your active input song is projected as the bright **Gold Star ⭐**.
    """)
    
    pca_df, pca_model = get_pca_projection(df, X_scaled, sample_size=3000)
    user_coords = pca_model.transform(input_scaled.values)[0]

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
        plot_bgcolor="rgba(19, 27, 46, 0.5)",
        xaxis=dict(title="Principal Component 1", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Principal Component 2", gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=520
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# ==========================================================
# TAB 5: Model Methodology
# ==========================================================
with tab_method:
    st.subheader("🔬 Machine Learning Methodology")
    
    st.markdown("""
    ### ⚙️ Pipeline Overview
    1. **Audio Feature Extraction**: 9 continuous musical features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo).
    2. **StandardScaler Normalization**: Centers and scales features ($z = \\frac{x - \\mu}{\\sigma}$) to prevent high-magnitude features (like tempo or loudness) from dominating.
    3. **K-Means Clustering**: Partitioned 95,837 tracks into $K=4$ clusters.
    4. **Recommendation Scoring**: Computes Euclidean Distance or Cosine Similarity within the candidate cluster.
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
    - $K=4$ cleanly separates **Instrumental & Ambient tracks** from **Acoustic vocal tracks**, creating distinct, musically coherent listening categories.
    """)
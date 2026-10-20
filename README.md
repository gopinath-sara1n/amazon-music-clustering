# 🎵 Amazon Music Clustering & Recommendation System

<p align="center">
  <a href="https://amazon-music-clustering-l4vs6wlkhjbtld8nq4trdv.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" height="36">
  </a>
</p>

<p align="center">
  <a href="https://amazon-music-clustering-l4vs6wlkhjbtld8nq4trdv.streamlit.app/">
    👉 <b>https://amazon-music-clustering-l4vs6wlkhjbtld8nq4trdv.streamlit.app/</b> 👈
  </a>
</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-KMeans-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</p>

---

## 📖 Overview

The **Amazon Music Clustering & Recommendation System** is an unsupervised machine learning web platform that groups and indexes **95,837 musical tracks** into **4 distinct acoustic categories** using **K-Means Clustering** ($K=4$) with **StandardScaler normalization**.

Featuring a **zero-delay reactive audio engine**, the application allows users to dynamically adjust musical attributes, load existing tracks, or select sound presets to immediately predict cluster membership, compare radar audio signatures, and explore nearest neighbor track recommendations with **playable Spotify embeds**.

---

## 🌟 Live Application

Experience the deployed, production application directly in your browser:

🔗 **[Live App](https://amazon-music-clustering-l4vs6wlkhjbtld8nq4trdv.streamlit.app/)**

---

## ✨ Key Features & Capabilities

### ⚡ 1. Real-Time Reactive Synthesizer
- **Instant Reactive Inference**: Adjust any of the 9 audio sliders or switch presets — the application recomputes cluster membership, distances, radar signatures, and recommendations instantly in real time without requiring manual submission.
- **9 Core Audio Dimensions**: Full slider control over *Danceability, Energy, Loudness (dB), Speechiness, Acousticness, Instrumentalness, Liveness, Valence, and Tempo (BPM)*.

### 🔎 2. Song Search & Auto-Filler
- **Search Across 95,837 Tracks**: Search any title or artist in the sidebar.
- **One-Click Audio Extraction**: Selecting a track automatically populates all 9 sliders with the song's exact recorded audio profile.

### 🎵 3. Quick-Load Sound Presets
- One-click presets for rapid testing and demonstration:
  - 💃 **High-Energy Dance / Pop**: High danceability, high energy, punchy loudness.
  - 🎸 **Acoustic Melodies & Ballads**: High acousticness, soft dynamics, low tempo.
  - 🎙️ **Spoken Word & Audio Stories**: High speechiness, low melodic features.
  - 🎼 **Instrumental Focus & Ambient**: High instrumentalness, atmospheric soundscapes.

### 🎧 4. Interactive Spotify Player Embeds
- **Similar Songs Player**: In the *Predict & Similar Songs* tab, use the `🎧 Select a Song to Listen with Spotify:` dropdown to preview and play audio clips for **any** recommended candidate song directly in the app.
- **Category Explorer Player**: In the *Browse Songs by Category* tab, pick any filtered track to stream its preview player.

### 📻 5. Browse Songs by Category
- Explore the entire dataset across any of the 4 clusters.
- **Search & Filter**: Keyword search on song/artist name and dropdown genre filters.
- **Multi-Column Sorting**: Sort by popularity, release year, duration, song name, or artist.
- **CSV Data Export**: Download filtered subsets directly for external analysis.

### 📊 6. Analytics, Visualizations & Latent Projection
- **Interactive Audio Radar**: Spider chart comparing user audio features against the cluster benchmark.
- **2D PCA Latent Space**: Dimensionality reduction via Principal Component Analysis showing where the current track sits relative to the cluster distributions.
- **Cluster Demographics**: Interactive Plotly donut chart and feature comparison bars across all 4 categories.

---

## 🧭 Application Walkthrough (5 Core Tabs)

| Tab | Icon | Key Functionality |
| :--- | :---: | :--- |
| **Predict & Similar Songs** | 🎯 | Real-time cluster classification, radar feature comparison, nearest-neighbor song recommendations table, and selectable Spotify audio preview player. |
| **Browse Songs by Category** | 📻 | Category-based library explorer, multi-parameter search, genre filtering, popularity sorting, Spotify audio player, and CSV export. |
| **Cluster Analytics** | 📊 | Cluster size distribution donut chart, cross-cluster comparative feature bar charts, and summary statistics. |
| **2D PCA Cluster Map** | 🪐 | PCA dimensionality reduction projection showing the user's active song plotted alongside dataset clusters. |
| **Model Methodology** | 🔬 | Complete documentation of feature normalization (StandardScaler), clustering evaluation metrics, and distance formulations. |

---

## 🎼 The 4 Music Categories ($K = 4$)

| Category | Signature Name | Track Count | Share | Core Musical Profile |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **🎸 Acoustic & Soft Songs** | 29,173 | 30.4% | High acousticness ($0.70$), low energy ($0.34$), organic melodic vocals |
| **1** | **🎉 Dance & High-Energy Songs** | 46,343 | 48.4% | High energy ($0.71$), high loudness ($-7.4\text{ dB}$), high valence ($0.69$), fast tempo |
| **2** | **🎙 Spoken Word & Audio Stories** | 12,474 | 13.0% | Dominant speechiness ($0.84$), podcasts, radio plays, spoken-word narrations |
| **3** | **🎼 Instrumental & Ambient Music** | 7,847 | 8.2% | Dominant instrumentalness ($0.81$), ambient soundscapes, focus & classical pieces |

---

## 📂 Dataset & Selected Audio Features

The model utilizes **9 core audio features** extracted from **95,837 tracks**:

| Feature | Description | Range / Unit |
| :--- | :--- | :--- |
| **Danceability** | How suitable a track is for dancing based on tempo, rhythm, and beat strength | $0.0 - 1.0$ |
| **Energy** | Perceptual measure of intensity, dynamic range, and timbre | $0.0 - 1.0$ |
| **Loudness** | Overall loudness of a track in decibels (dB) | $-60.0 \text{ to } +5.5 \text{ dB}$ |
| **Speechiness** | Detects presence of spoken words (podcasts/narration vs singing) | $0.0 - 1.0$ |
| **Acousticness** | Confidence measure of whether the track is acoustic | $0.0 - 1.0$ |
| **Instrumentalness** | Predicts whether a track contains no vocals | $0.0 - 1.0$ |
| **Liveness** | Detects presence of an audience in the recording | $0.0 - 1.0$ |
| **Valence** | Musical positiveness (happy, euphoric vs sad, depressed) | $0.0 - 1.0$ |
| **Tempo** | Overall estimated pace of a track in beats per minute (BPM) | $0 - 250 \text{ BPM}$ |

---

## 📊 Optimal Cluster Selection ($K = 4$)

The optimal number of clusters was validated using the **Elbow Method (WCSS)**, **Silhouette Score**, and **Davies-Bouldin Index (DBI)**:

| $K$ | WCSS (Inertia) | Silhouette Score | Davies-Bouldin Index (DBI) | Evaluation & Decision |
| :---: | :---: | :---: | :---: | :--- |
| 2 | 685,950.97 | 0.2102 | 1.8692 | Under-segmentation; collapses genres together |
| 3 | 573,286.79 | **0.2494** | 1.5169 | Merges instrumental & ambient tracks with acoustic music |
| **4** | **507,958.11** | **0.2394** | **1.4638** | **Selected Optimal: Lowest DBI, cleanly isolates Instrumental vs Acoustic** |
| 5 | 464,185.95 | 0.1960 | 1.6164 | Creates fragmented, redundant sub-clusters |
| 6 | 437,304.11 | 0.1691 | 1.6251 | Degraded separation and increased overlap |

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Machine Learning:** Scikit-Learn (`KMeans`, `StandardScaler`, `PCA`, `pairwise_distances`)
- **Web Framework:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Interactive Visualizations:** Plotly (`Scatterpolar`, `Scatter`, `Pie`, `Bar`)
- **Model Serialization:** Joblib
- **Audio Integration:** Spotify Embed iFrames

---

## 🚀 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/gopinath-sara1n/amazon-music-clustering.git
cd amazon-music-clustering
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
amazon-music-clustering/
│
├── .streamlit/
│   └── config.toml             # Custom music dark theme
├── app.py                      # Main production Streamlit web application
├── amazon_music_clustered.csv  # 95,837 Clustered tracks with Spotify IDs
├── kmeans_model.pkl            # Pre-trained K-Means (K=4) model
├── scaler.pkl                  # Fitted StandardScaler object
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Project documentation & live app link
```

---

## 👨‍💻 Author

**Gopinath S**
- 🐙 **GitHub:** [@gopinath-sara1n](https://github.com/gopinath-sara1n)
- 💼 **LinkedIn:** [Gopinath S](https://www.linkedin.com/in/gopinaths/)
- 📧 **Email:** *gopinath13107@gmail.com*

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

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

The **Amazon Music Clustering & Recommendation System** is an unsupervised machine learning application that segments **95,837 musical tracks** into **4 distinct acoustic categories** using **K-Means Clustering** with **StandardScaler normalization**. 

Using the clustered audio space, the application provides a **real-time reactive music recommendation engine**: adjust any audio characteristic or pick a musical preset, and the app instantly predicts the song's cluster, computes vector similarities, visualizes an audio radar profile, and recommends similar tracks with **playable Spotify embeds**.

---

## 🌟 Live Application

Launch the live application directly in your browser:

🔗 **[https://amazon-music-clustering-l4vs6wlkhjbtld8nq4trdv.streamlit.app/](https://amazon-music-clustering-l4vs6wlkhjbtld8nq4trdv.streamlit.app/)**

---

## ✨ Features

- ⚡ **Instant Real-Time Prediction**: No click-to-predict delay — changing any slider or picking a preset instantly recomputes cluster predictions and recommendations in real time.
- 🎛️ **Audio Synthesizer & Quick Presets**: Adjust 9 audio sliders (Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo) or load one-click sound profiles (*High-Energy Dance, Acoustic Soft Melodies, Spoken Word, Instrumental Ambient*).
- 📻 **Browse Songs by Category**: Select any of the 4 music categories to browse all tracks, search by title/artist, filter by genre, sort by popularity or release date, listen with Spotify, and export results to CSV.
- 🎧 **Interactive Spotify Embeds**: Play preview audio directly inside the web app for top recommended tracks and browsed category songs.
- 🧭 **Dual Distance Engine**: Choose between **Euclidean Distance** (spatial distance) or **Cosine Similarity** (directional alignment) for recommendations.
- 🕸️ **Plotly Audio Radar Chart**: Multi-layer spider chart visually comparing your audio input against the cluster's average sound signature.
- 🪐 **Interactive 2D PCA Cluster Map**: Dimensionality reduction via Principal Component Analysis, plotting where your active song sits in latent cluster space.
- 📊 **Cluster Analytics & EDA**: Donut distribution breakdown, cross-cluster feature comparison bar charts, and category summary tables.

---

## 🎼 The 4 Music Categories ($K = 4$)

| Category | Signature Name | Track Count | Core Musical Profile |
| :---: | :--- | :---: | :--- |
| **0** | **🎸 Acoustic & Soft Songs** | 29,173 (30.4%) | High acousticness ($0.70$), low energy ($0.34$), organic melodic vocals |
| **1** | **🎉 Dance & High-Energy Songs** | 46,343 (48.4%) | High energy ($0.71$), high loudness ($-7.4\text{ dB}$), high valence ($0.69$), fast tempo |
| **2** | **🎙 Spoken Word & Audio Stories** | 12,474 (13.0%) | Dominant speechiness ($0.84$), podcasts, radio plays, spoken-word narrations |
| **3** | **🎼 Instrumental & Ambient Music** | 7,847 (8.2%) | Dominant instrumentalness ($0.81$), ambient soundscapes, focus & classical pieces |

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

The optimal number of clusters was evaluated using the **Elbow Method (WCSS)**, **Silhouette Score**, and **Davies-Bouldin Index (DBI)**:

| $K$ | WCSS (Inertia) | Silhouette Score | Davies-Bouldin Index (DBI) | Rationale |
| :---: | :---: | :---: | :---: | :--- |
| 2 | 685,950.97 | 0.2102 | 1.8692 | Under-segmentation |
| 3 | 573,286.79 | **0.2494** | 1.5169 | Merges instrumental & ambient with acoustic |
| **4** | **507,958.11** | **0.2394** | **1.4638** | **Selected Optimal: Lowest DBI, cleanly separates Instrumental vs Acoustic** |
| 5 | 464,185.95 | 0.1960 | 1.6164 | Redundant splits |
| 6 | 437,304.11 | 0.1691 | 1.6251 | Degraded separation |

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Machine Learning:** Scikit-Learn (`KMeans`, `StandardScaler`, `PCA`, `pairwise_distances`)
- **Web Framework:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Interactive Visualization:** Plotly (`Scatterpolar`, `Scatter`, `Pie`, `Bar`)
- **Model Serialization:** Joblib

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

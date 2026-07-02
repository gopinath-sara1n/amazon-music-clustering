# 🎵 Amazon Music Clustering and Recommendation System

An end-to-end Machine Learning project that clusters Amazon Music tracks using **K-Means Clustering** and recommends similar songs based on **Euclidean Distance**. The project also includes an interactive **Streamlit Web Application** for predicting the cluster of a new song and recommending similar songs.

---

## 📌 Project Overview

Music streaming platforms contain millions of songs with different audio characteristics. Organizing these songs into meaningful groups helps users discover music that matches their preferences and improves recommendation systems.

This project uses **unsupervised machine learning (K-Means Clustering)** to group songs according to their audio features. The trained model is integrated into a **Streamlit application** that predicts the cluster of a song and recommends the most similar songs from the dataset.

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Select meaningful audio features
- Scale features using StandardScaler
- Determine the optimal number of clusters
- Build a K-Means clustering model
- Analyze and name each cluster
- Visualize clustering results
- Save trained model using Pickle
- Develop a Streamlit recommendation system

---

# 📂 Dataset Information

| Property | Value |
|----------|-------|
| Dataset | Amazon Music |
| Total Songs | **95,837** |
| Total Columns | **23** |
| Float Columns | 10 |
| Integer Columns | 7 |
| String Columns | 6 |
| Missing Values | None |
| Duplicate Rows | None |

---

# 🎵 Selected Audio Features

The clustering model uses the following audio features:

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo

These features represent the musical characteristics of each song.

---

# ⚙️ Feature Scaling

The selected features are standardized using **StandardScaler** before training the K-Means model.

Benefits:

- Equal contribution from every feature
- Better distance calculations
- Faster convergence
- Improved clustering performance

---

# 📊 Cluster Selection

Several K values (2–10) were evaluated using:

- Elbow Method (WCSS)
- Silhouette Score
- Davies-Bouldin Index

## Evaluation Summary

| K | Silhouette | DBI |
|---|------------|------|
| 2 | 0.2102 | 1.8692 |
| 3 | **0.2494** | 1.5169 |
| 4 | 0.2394 | **1.4638** |
| 5 | 0.1960 | 1.6164 |
| 6 | 0.1691 | 1.6251 |
| 7 | 0.1751 | 1.5432 |
| 8 | 0.1828 | 1.5102 |
| 9 | 0.1828 | 1.4526 |
| 10 | 0.1749 | 1.4732 |

Although **K = 3** achieved the highest Silhouette Score, **K = 4** provided:

- Better cluster interpretation
- Better separation
- Better overall ranking
- Better recommendation quality

Therefore, **K = 4** was selected.

---

# 🎼 Final Music Clusters

| Cluster | Music Type |
|----------|-------------------------------|
| 0 | Acoustic & Melodic Music |
| 1 | Energetic & Danceable Music |
| 2 | Speech & Narration Audio |
| 3 | Instrumental & Ambient Music |

Cluster names were assigned by:

- Inspecting random songs from each cluster
- Comparing average feature values

---

# 📈 Visualizations

The project includes several visualizations:

- PCA Cluster Visualization
- Elbow Curve
- Silhouette Score Plot
- Davies-Bouldin Score Plot
- Average Feature Bar Charts
- Heatmap
- Boxplots
- Cluster Distribution Chart

---

# 💾 Model Saving

The trained objects are saved using **Pickle**.

Saved files:

- `kmeans_model.pkl`
- `scaler.pkl`

---

# 💻 Streamlit Application

The project includes an interactive Streamlit web application.

## Features

- Predict song cluster
- Display cluster information
- Recommend Top 10 similar songs
- Content-based recommendation using Euclidean Distance

### Workflow

1. User enters the nine audio features.
2. Input is converted into a DataFrame.
3. StandardScaler scales the features.
4. K-Means predicts the music cluster.
5. Songs from the predicted cluster are filtered.
6. Euclidean distance is calculated.
7. Top 10 nearest songs are recommended.

---

# 📁 Project Structure

```
amazon-music-clustering/
│
├── Amazon_Music_EDA_KMeans_Recommendation.ipynb
├── app.py
├── requirements.txt
├── README.md
├── dataset.csv
├── kmeans_model.pkl
├── scaler.pkl
├── images/
│   ├── pca.png
│   ├── heatmap.png
│   ├── boxplot.png
│   ├── cluster_distribution.png
│   └── feature_barplots.png
└── report/
    └── Amazon Music Clustering - Project Report.pdf
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Pickle

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/amazon-music-clustering.git
```

Move into the project folder

```bash
cd amazon-music-clustering
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📊 Results

- Successfully clustered **95,837 songs**
- Identified **4 meaningful music groups**
- Built a content-based recommendation engine
- Developed an interactive Streamlit application
- Generated accurate recommendations using Euclidean Distance

---

# 🚀 Future Improvements

- Genre-aware recommendations
- Artist similarity
- Deep Learning embeddings
- Cosine Similarity comparison
- Spotify API integration
- Deployment on Streamlit Community Cloud

---

# 👨‍💻 Author

**Gopinath S**

Machine Learning | Data Science | Python | Streamlit

---

## ⭐ If you found this project useful, please give it a Star!

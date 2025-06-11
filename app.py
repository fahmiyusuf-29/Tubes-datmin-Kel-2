# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA

# ========== LOAD DATA ========== #
df = pd.read_csv('enhanced_anxiety_dataset (1).csv')
st.title("📊 Dashboard Kecemasan Sosial - Data Mining")

# ========== SIDEBAR FILTER ========== #
st.sidebar.header("🎛 Filter Data")
gender_filter = st.sidebar.multiselect("Pilih Gender:", df['Gender'].unique(), default=df['Gender'].unique())
df = df[df['Gender'].isin(gender_filter)]

# ========== DATA EXPLORATION ========== #
st.header("📌 Statistik Deskriptif")
st.dataframe(df.describe())

st.subheader("📈 Distribusi Tingkat Kecemasan")
fig, ax = plt.subplots()
sns.histplot(df["Anxiety Level (1-10)"], bins=10, kde=True, ax=ax)
st.pyplot(fig)

# ========== ENCODING ========== #
# Label encoding untuk data kategorikal
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# ========== OUTLIER DETECTION & REMOVAL ========== #
# Gunakan Z-score untuk menghapus outlier (|z| > 3)
numerical_cols = df_encoded.select_dtypes(include=['float64', 'int64']).columns
z_scores = np.abs((df_encoded[numerical_cols] - df_encoded[numerical_cols].mean()) / df_encoded[numerical_cols].std())
df_encoded = df_encoded[(z_scores < 3).all(axis=1)]

# ========== FEATURE SELECTION ========== #
# Korelasi Pearson untuk pilih fitur paling relevan terhadap Anxiety Level
correlation = df_encoded.corr()['Anxiety Level (1-10)'].abs().sort_values(ascending=False)
selected_features = correlation[1:8].index.tolist()  # Ambil 7 fitur teratas (selain target)
st.subheader("📌 Korelasi Fitur dengan Tingkat Kecemasan")
st.write(correlation)

# ========== SCALING ========== #
scaler = MinMaxScaler()
X = df_encoded[selected_features]
X_scaled = scaler.fit_transform(X)

# Binning target: Low Anxiety = 0, High Anxiety = 1
y = (df_encoded['Anxiety Level (1-10)'] >= 5).astype(int)

# ========== LOGISTIC REGRESSION ========== #
st.header("🤖 Klasifikasi: Logistic Regression")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# ========== K-MEANS CLUSTERING ========== #
st.header("📌 Clustering: K-Means")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Tambahkan hasil cluster ke dataframe
df_encoded['Cluster'] = clusters

st.write("📊 Rata-rata tiap fitur berdasarkan cluster:")
st.dataframe(df_encoded.groupby("Cluster").mean()[selected_features + ['Anxiety Level (1-10)']])

# Silhouette Score
score = silhouette_score(X_scaled, clusters)
st.success(f"Silhouette Score: {score:.2f}")

# ========== VISUALISASI CLUSTERING ========== #
st.subheader("📊 Visualisasi Clustering (PCA 2D)")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_encoded['pca1'] = pca_result[:,0]
df_encoded['pca2'] = pca_result[:,1]

fig, ax = plt.subplots()
sns.scatterplot(data=df_encoded, x='pca1', y='pca2', hue='Cluster', palette='viridis', ax=ax)
st.pyplot(fig)

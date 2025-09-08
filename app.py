import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kagglehub import dataset_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- Page Configuration ---
st.set_page_config(
    page_title="BBC News Segmentation",
    page_icon="📰",
    layout="wide"
)


# --- Caching Functions for Performance ---

@st.cache_data
def load_data():
    """Downloads and loads the BBC news dataset."""
    path = dataset_download("yufengdev/bbc-fulltext-and-category")
    df = pd.read_csv(path + "/bbc-text.csv")
    return df


@st.cache_resource
def vectorize_text(df):
    """Creates and fits a TF-IDF vectorizer, then transforms the text data."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_df=0.8,
        min_df=3,
        ngram_range=(1, 2)
    )
    X_text = vectorizer.fit_transform(df['text'])
    return vectorizer, X_text


# CORRECTED FUNCTION: The 'X_text' parameter is now '_X_text'
@st.cache_data
def run_kmeans_and_pca(_vectorizer, _X_text, k=5):
    """Runs KMeans clustering and PCA dimensionality reduction."""
    # K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    # Use _X_text here
    labels = kmeans.fit_predict(_X_text)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    # Use _X_text here
    X_reduced = pca.fit_transform(_X_text.toarray())
    centers_reduced = pca.transform(kmeans.cluster_centers_)

    # Get top words for theme interpretation
    terms = _vectorizer.get_feature_names_out()
    top_words = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        top_indices = center.argsort()[-10:][::-1]
        top_words[i] = [terms[j] for j in top_indices]

    # Calculate Silhouette Score
    # Use _X_text here
    score = silhouette_score(_X_text, labels)

    return labels, X_reduced, centers_reduced, top_words, score


# --- Main App ---

st.title("📰 BBC News Segmentation using K-Means Clustering")

st.markdown("""
This application demonstrates unsupervised machine learning to segment news articles from the BBC dataset. 
It uses **TF-IDF** to convert text into numerical features and **K-Means** to cluster the articles into distinct topics. 
**Principal Component Analysis (PCA)** is used to visualize the high-dimensional data in 2D.
""")

# --- Load and Prepare Data ---
with st.spinner('Downloading and loading dataset...'):
    df = load_data()
    vectorizer, X_text = vectorize_text(df)

# --- Section 1: Exploratory Data Analysis (EDA) ---
st.header("1. Exploratory Data Analysis")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Dataset Sample")
    st.dataframe(df.head())

with col2:
    st.subheader("News Category Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x=df['category'], ax=ax, palette='viridis')
    ax.set_title("Distribution of News Categories")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Articles")
    st.pyplot(fig)
    st.write("The dataset is fairly balanced across 5 distinct categories, making it a good candidate for clustering.")

# --- Section 2: Interactive Clustering Analysis ---
st.header("2. Interactive Clustering Analysis")
st.write("Use the slider below to change the number of clusters (k) and see how it affects the segmentation.")

# User input for number of clusters
k = st.slider(
    'Select the number of clusters (k)',
    min_value=2,
    max_value=8,
    value=5,  # Default value based on original analysis
    help="The optimal value for this dataset is 5, corresponding to the original news categories."
)

# Run analysis based on selected k
labels, X_reduced, centers_reduced, top_words, score = run_kmeans_and_pca(vectorizer, X_text, k)

# Add cluster labels and themes to the DataFrame
df['cluster'] = labels

# --- Display Results ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Cluster Visualization for k={k}")
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.scatter(centers_reduced[:, 0], centers_reduced[:, 1], s=200, c='red', marker='X', label='Centroids')
    ax.set_title('News Articles Clustered and Visualized with PCA')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # Create a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                                  markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10) for i in range(k)]
    ax.legend(handles=legend_elements, title="Clusters")

    st.pyplot(fig)

with col2:
    st.subheader("Interpreting the Clusters")
    st.write("Top 10 keywords that define each cluster:")

    theme_map = {}
    for cluster_id, words in top_words.items():
        theme_name = f"Cluster {cluster_id}"
        with st.expander(theme_name):
            st.write(", ".join(words))

    st.subheader("Clustering Performance")
    st.metric(label="Silhouette Score", value=f"{score:.3f}")
    st.info("""
    **Silhouette Score:**
    - Ranges from -1 to 1.
    - **+1:** Clusters are dense and well-separated.
    - **0:** Clusters are overlapping.
    - **-1:** Data points may be assigned to the wrong clusters.

    A score close to 0, like the one here, is common for high-dimensional text data and indicates some overlap between topics.
    """)

# --- Section 3: Explore Clustered Data ---
st.header("3. Explore the Results")
st.write("The original data with the predicted cluster for each article.")

# Assign themes based on the top words for k=5 (the optimal value)
if k == 5:
    # This mapping is based on your notebook's output for k=5
    theme_map_final = {
        0: "Sports (Football)",
        1: "Business/Tech",
        2: "Sports (General/Rugby)",
        3: "Entertainment/Arts",
        4: "Politics"
    }
    df['theme'] = df['cluster'].map(theme_map_final)

    # Allow user to filter by theme
    all_themes = ["All"] + list(theme_map_final.values())
    selected_theme = st.selectbox("Filter articles by predicted theme:", all_themes)

    if selected_theme == "All":
        st.dataframe(df[['text', 'cluster', 'theme']])
    else:
        st.dataframe(df[df['theme'] == selected_theme][['text', 'cluster', 'theme']])
else:
    st.warning("Theme mapping is only available when k=5.")
    st.dataframe(df[['text', 'cluster']])
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Custom CSS Styling
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #141e30, #243b55);
}
h1 {
    text-align: center;
    color: white;
}
.movie-card {
    background-color: #1f2c3d;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    color: white;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
}
.genre-badge {
    background-color: #ff4b4b;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 12px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title Section
st.title("🎬 Smart Movie Recommendation System")
st.write("Discover movies similar to your favorites using AI-powered similarity analysis.")

# Load Dataset
data = pd.read_csv("movies.csv")
data["content"] = data["genre"] + " " + data["description"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["content"])
similarity_matrix = cosine_similarity(tfidf_matrix)

# Sidebar Controls
st.sidebar.header("⚙️ Customize Recommendations")
selected_movie = st.sidebar.selectbox("Select a Movie", data["title"].values)
num_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# Recommendation Logic
if st.sidebar.button("🎯 Generate Recommendations"):
    index = data[data["title"] == selected_movie].index[0]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    st.subheader(f"✨ Top {num_recommendations} Recommendations for '{selected_movie}'")

    cols = st.columns(2)

    count = 0
    for i in sorted_movies[1:]:
        movie_title = data.iloc[i[0]]["title"]
        genre = data.iloc[i[0]]["genre"]
        description = data.iloc[i[0]]["description"]
        score = round(i[1] * 100, 2)

        with cols[count % 2]:
            st.markdown(f"""
            <div class="movie-card">
                <h3>🎥 {movie_title}</h3>
                <span class="genre-badge">{genre}</span>
                <p>{description}</p>
                <p>⭐ Similarity Score: {score}%</p>
            </div>
            """, unsafe_allow_html=True)

        count += 1
        if count >= num_recommendations:
            break

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align:center; font-size:2.2em; font-weight:bold; background: linear-gradient(90deg, #ff4b4b, #ffb347, #43e97b, #38f9d7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; filter: drop-shadow(0 0 8px #ffb347); letter-spacing: 2px;'>
    Unleash The Fun!!
</p>
""", unsafe_allow_html=True)
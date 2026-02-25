import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("movies.csv")

# Combine text features
data["content"] = data["genre"] + " " + data["description"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["content"])

# Cosine Similarity Matrix
similarity_matrix = cosine_similarity(tfidf_matrix)


def recommend_movie(movie_title, num_recommendations=5):
    movie_title = movie_title.lower()

    # Partial matching
    matching_movies = data[data["title"].str.lower().str.contains(movie_title)]

    if matching_movies.empty:
        print("❌ Movie not found in dataset.")
        return

    index = matching_movies.index[0]

    similarity_scores = list(enumerate(similarity_matrix[index]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print("\n🎬 Recommended Movies:\n")

    count = 0
    for i in sorted_movies[1:]:
        print(f"⭐ {data.iloc[i[0]]['title']}  (Score: {round(i[1], 2)})")
        count += 1
        if count >= num_recommendations:
            break


# User input
movie_name = input("Enter a movie name: ")
num = int(input("How many recommendations do you want? "))

recommend_movie(movie_name, num)
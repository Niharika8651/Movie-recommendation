# 🎬 Smart Movie Recommendation System

##  Project Overview

This project implements an AI-powered Content-Based Movie Recommendation System using Machine Learning and Natural Language Processing (NLP) techniques. 

The system recommends similar movies based on genre and description similarity using TF-IDF vectorization and Cosine Similarity.

The application is deployed as an interactive web interface using Streamlit with a modern Netflix-style UI.

---

## Features

- Content-Based Filtering
- TF-IDF Vectorization
- Cosine Similarity Scoring
- 60+ Movie Custom Dataset
- Creative Netflix-Style UI
- Sidebar Custom Controls
- Similarity Percentage Display
- Responsive Layout

---

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Natural Language Processing (NLP)
- VS Code

---

## How The System Works

1. Combine movie genre and description into a single text feature.
2. Apply TF-IDF Vectorization to convert text into numerical vectors.
3. Compute Cosine Similarity between all movie vectors.
4. Sort similarity scores.
5. Recommend Top-N most similar movies selected by the user.

---


---

## ▶️ Installation & Execution

### Step 1: Install Dependencies
 pip install pandas scikit-learn streamlit

### Step 2: Run Application
 streamlit run app.py


The application will open automatically in your browser.

---

## Machine Learning Concepts Applied

- Natural Language Processing (NLP)
- TF-IDF (Term Frequency – Inverse Document Frequency)
- Cosine Similarity
- Content-Based Filtering
- Text Feature Engineering

---

## Future Enhancements

- Hybrid Recommendation Model (Collaborative + Content-Based)
- Integration with Movie APIs (TMDB for Posters)
- User Login & History Tracking
- Cloud Deployment (Streamlit Cloud / Render)
- Model Evaluation Metrics

---

## Academic Relevance

This project demonstrates practical implementation of:

- Text preprocessing
- Feature extraction
- Similarity-based recommendation algorithms
- Web deployment of ML models


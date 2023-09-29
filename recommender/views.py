from django.shortcuts import render, redirect

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# Create your views here.
def home(request):
    return render(request, "index.html")

def train_model(request):
    credits = pd.read_csv("static/Movie Credits.csv")

    credits = credits.rename(index=str, columns={"movie_id": "id"})

    movies = pd.read_csv("static/Movies.csv")

    movies = movies.merge(credits, on="id")

    movies = movies.drop(columns=["homepage", "title_x", "title_y", "status", "production_countries"])

    movies["overview"] = movies["overview"].fillna("")

    movies.to_csv("static/movies_cleaned.csv")

    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents="unicode",
                        analyzer="word", token_pattern="\w{1,}",
                        ngram_range=(1, 3), stop_words="english")

    tfv_matrix = tfv.fit_transform(movies["overview"])

    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

    with open("static/sigmoid", "wb") as file:
        joblib.dump(sig, file)
    
    return redirect("home")

def recommend(request):
    # Getting the movie input
    title = request.POST['movie_input']

    # Loading required files
    movies = pd.read_csv("static/movies_cleaned.csv")
    sig = joblib.load("static/sigmoid")

    # Building a Series of indices
    indices = pd.Series(movies.index, index=movies["original_title"]).drop_duplicates()

    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwise similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    data = {'data': movies["original_title"].iloc[movie_indices]}
    return render(request, "output.html", data)
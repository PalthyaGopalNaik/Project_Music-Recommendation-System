from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import faiss

app = Flask(__name__)

df = pd.read_csv("dataset/lyrics.csv")
df.columns = df.columns.str.strip()

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = vectorizer.fit_transform(df['lyrics'].fillna(""))

song_features_array = tfidf_matrix.toarray().astype('float32')

index = faiss.IndexFlatIP(song_features_array.shape[1])
index.add(song_features_array)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['query']
    query_type = request.form['query_type'].strip().lower()

    if query_type == "song":
        return recommend_songs(user_input)
    elif query_type == "artist":
        return recommend_artist(user_input)
    elif query_type == "lyrics":
        return recommend_lyrics(user_input)
    else:
        return jsonify({"error": "Invalid input type. Choose song, artist, or lyrics."})

def recommend_songs(name, top_n=5):
    if name not in df['name'].values:
        return jsonify({"message": "Song not found. Showing nearest matches:", "recommendations": recommend_nearest_songs(name, top_n)})

    song_lyrics = df[df['name'] == name]['lyrics'].values[0]
    song_genre = df[df['name'] == name]['genre'].values[0]
    query_vector = vectorizer.transform([song_lyrics]).toarray().astype('float32')

    _, song_indices = index.search(query_vector, top_n)
    recommendations = df.iloc[song_indices[0]][['name', 'artists', 'genre', 'lyrics']]
    return jsonify({"message": "Here are the best recommendations:", "recommendations": recommendations.to_dict(orient='records')})

def recommend_artist(artist_name, top_n=5):
    artist_songs = df[df['artists'].str.contains(artist_name, case=False, na=False)].head(top_n)
    if artist_songs.empty:
        return jsonify({"message": "Artist not found. Showing similar artists:", "recommendations": recommend_nearest_songs(artist_name, top_n)})
    return jsonify({"message": "Songs from this artist:", "recommendations": artist_songs[['name', 'artists', 'genre']].to_dict(orient='records')})

def recommend_lyrics(lyrics, top_n=5):
    query_vector = vectorizer.transform([lyrics]).toarray().astype('float32')
    _, song_indices = index.search(query_vector, top_n)
    recommendations = df.iloc[song_indices[0]][['name', 'artists', 'genre', 'lyrics']]
    return jsonify({"message": "Here are the best recommendations:", "recommendations": recommendations.to_dict(orient='records')})

def recommend_nearest_songs(name, top_n=5):
    query_vector = vectorizer.transform([name]).toarray().astype('float32')
    _, song_indices = index.search(query_vector, top_n)
    recommendations = df.iloc[song_indices[0]][['name', 'artists', 'genre', 'lyrics']]
    return recommendations.to_dict(orient='records')

if __name__ == '__main__':
    app.run(debug=True)

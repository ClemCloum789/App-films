
from flask import Flask, request, jsonify, render_template
import pandas as pd
import difflib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch
import os

# 🔧 Supprime le warning des tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

# 🔹 Chargement du modèle sémantique
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔹 Chargement des données
movies = pd.read_csv('data/movies.csv')

# 🧹 Nettoyage des champs texte
for col in ['keywords', 'tagline']:
    if col in movies.columns:
        movies[col] = movies[col].fillna('').astype(str)

# 🔸 Filtrage des films valides (overview obligatoire)
movies = movies.dropna(subset=['overview']).copy()

# 🔸 Traitement des genres (⚠️ après filtrage)
movies['genres'] = movies['genres'].fillna('').str.split('|')
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genres'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=movies.index)  # ✅ index synchro

# 🔸 Fusion des champs texte
def combine_metadata(row):
    genre_text = ' '.join(row['genres']) if isinstance(row['genres'], list) else ''
    return f"{row.get('overview', '')} {genre_text} {row.get('tagline', '')} {row.get('keywords', '')}"

movies['metadata'] = movies.apply(combine_metadata, axis=1)

# 🔸 Encodage sémantique
print("Encodage des métadonnées enrichies...")
movies['embedding'] = movies['metadata'].apply(lambda x: model.encode(x, convert_to_tensor=True))

# 🎯 Recommandation hybride
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_title = data.get('title')

    # 🔍 Recherche du film
    valid_titles = movies['title'].tolist()
    closest_titles = difflib.get_close_matches(movie_title, valid_titles, n=1, cutoff=0.6)
    if not closest_titles:
        return jsonify({'error': 'Film non trouvé'}), 404

    matched_title = closest_titles[0]

    # ✅ Utiliser l'index numérique pour cohérence
    matched_idx = movies[movies['title'] == matched_title].index[0]
    query_index = movies.index.get_loc(matched_idx)

    # 🔁 Matrice de similarité genres
    genre_sim_matrix = cosine_similarity(genre_df.values, genre_df.values)
    genre_sim = genre_sim_matrix[query_index]

    # 🔁 Similarité sémantique
    query_embedding = movies.loc[matched_idx]['embedding'].cpu()
    all_embeddings = torch.stack([e.cpu() for e in movies['embedding']])
    semantic_sim = util.pytorch_cos_sim(query_embedding, all_embeddings).squeeze(0).numpy()

    # ✅ Fusion
    combined_sim = 0.5 * genre_sim + 0.5 * semantic_sim

    # 📌 Exclure le film de base
    sim_scores = list(enumerate(combined_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sim_scores if i != query_index][:5]

    recommendations = movies.iloc[top_indices]['title'].tolist()
    return jsonify({'recommendations': recommendations})


# 🌐 Accueil HTML
@app.route('/')
def home():
    return render_template('index.html')


# 🚀 Lancement
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

    #python3 -m venv venv
    #source venv/bin/activate
    #cd ~/Documents/app.films
    #python backend.py 
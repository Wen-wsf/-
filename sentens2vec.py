import pandas as pd
import os
import pickle
import torch
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from sentence_transformers import SentenceTransformer

def load_youtube_to_movie_mapping(filepath):
    df = pd.read_csv(filepath)
    movie_to_youtube = pd.Series(df.youtubeId.values, index=df.movieId.astype(str)).to_dict()
    return movie_to_youtube

def extract_audio_to_text(video_path, lang='en-US', start_sec=10, end_sec=60):
    try:
        clip = VideoFileClip(video_path).subclip(start_sec, end_sec)
        audio_path = "temp_audio.wav"
        clip.audio.write_audiofile(audio_path)
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language=lang)
        os.remove(audio_path)
        clip.close()
        return text
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None

def text_to_vector_or_zero(text, vector_length=384):
    if text:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text)
    else:
        return torch.zeros(vector_length)

def add_text_features_to_graph(g, movie_id, text_features):
    if 'features' not in g.nodes['movietext'].data:
        num_movies = g.number_of_nodes('movietext')
        num_features = text_features.shape[0] if text_features is not None else 384
        g.nodes['movietext'].data['features'] = torch.zeros((num_movies, num_features), dtype=torch.float32)
    idx = g.nodes['movie'].data['movie_id'].tolist().index(int(movie_id))
    g.nodes['movietext'].data['features'][idx] = torch.tensor(text_features, dtype=torch.float32)

def process_videos(graph, mapping_file, video_folder, output_file):
    movie_to_youtube = load_youtube_to_movie_mapping(mapping_file)
    all_movie_ids = set([mid.item() for mid in graph.nodes['movie'].data['movie_id']])
    processed_count = 0
    problematic_movies = []

    for movie_id in all_movie_ids:
        youtube_id = movie_to_youtube.get(str(movie_id))
        if youtube_id:
            video_path = os.path.join(video_folder, f"{youtube_id}.mp4")
            if os.path.exists(video_path):
                text = extract_audio_to_text(video_path)
                text_vector = text_to_vector_or_zero(text)
                add_text_features_to_graph(graph, movie_id, text_vector)
                processed_count += 1
                if text is None:
                    problematic_movies.append(movie_id)
                    print(f"No text extracted for movie ID {movie_id}.")
            else:
                problematic_movies.append(movie_id)
                print(f"Video file not found: {video_path}")
        else:
            problematic_movies.append(movie_id)
            print(f"No YouTube ID found for movie ID {movie_id}.")

    print(f"Total processed videos: {processed_count}")
    with open(output_file, 'w') as f:
        for movie_id in problematic_movies:
            f.write(f"{movie_id}\n")

# File paths
video_folder = r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\data\videos'
mapping_file = r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\data\ml-youtube_cleaned.csv'

# Process each graph
graphs_to_process = [
    r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph03.pkl',
    r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph05.pkl',
    r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph08.pkl',
    r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph03_with_features.pkl',
    r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph05_with_features.pkl',
    r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph08_with_features.pkl'
]

for graph_path in graphs_to_process:
    graph = pickle.load(open(graph_path, 'rb'))
    output_file = graph_path.replace('.pkl', '_problematic_movies.txt')
    process_videos(graph, mapping_file, video_folder, output_file)
    updated_graph_path = graph_path.replace('.pkl', '_with_text_features.pkl')
    with open(updated_graph_path, 'wb') as f:
        pickle.dump(graph, f)

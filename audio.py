import pandas as pd
import numpy as np
import librosa
import os
from moviepy.editor import VideoFileClip
import pickle
from tqdm import tqdm
import torch
import pickle

def load_graph_with_images(graph_path):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    return graph

def load_audio_features(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def add_audio_features_to_graph(g, audio_features_dict):
    for node_data in tqdm(g.nodes['movie'].data['movie_id'], desc="Updating graph with audio features"):
        movie_id = str(node_data.item())
        if movie_id in audio_features_dict:
            idx = g.nodes['movie'].data['movie_id'].tolist().index(int(movie_id))
            features = torch.tensor(audio_features_dict[movie_id], dtype=torch.float32)
            g.nodes['movieaudio'].data['features'][idx] = features


def load_youtube_to_movie_mapping(filepath):
    df = pd.read_csv(filepath)
    return pd.Series(df.youtubeId.values, index=df.movieId.astype(str)).to_dict()

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    return np.mean(log_S, axis=1)

def extract_audio_from_video(video_path):
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        start_time = min(10, duration)
        end_time = min(60, duration)
        audio_path = "temp_audio.wav"
        if start_time < end_time:
            clip.subclip(start_time, end_time).audio.write_audiofile(audio_path)
            return audio_path
    return None

def extract_and_save_audio_features(video_folder, mapping_file, output_file):
    movie_to_youtube = load_youtube_to_movie_mapping(mapping_file)
    audio_features_dict = {}
    for movie_id, youtube_id in tqdm(movie_to_youtube.items(), desc="Extracting audio features"):
        video_path = os.path.join(video_folder, f"{youtube_id}.mp4")
        if os.path.exists(video_path):
            audio_path = extract_audio_from_video(video_path)
            if audio_path:
                audio_features = extract_audio_features(audio_path)
                audio_features_dict[movie_id] = audio_features
                os.remove(audio_path)  # Clean up the temporary audio file
    with open(output_file, 'wb') as f:
        pickle.dump(audio_features_dict, f)

# Set paths and run extraction
graph_paths = [
    'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph03_with_features.pkl',
    'D:/CODE/multi-model knowledge graph multi-graph recommendation system/code/mainmodel/hetero_graph03_with_text_features.pkl',
    'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph03.pkl',
    'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph05_with_features.pkl',
    'D:/CODE/multi-model knowledge graph multi-graph recommendation system/code/mainmodel/hetero_graph05_with_text_features.pkl',
    'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph05.pkl',
    'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph08_with_features.pkl',
    'D:/CODE/multi-model knowledge graph multi-graph recommendation system/code/mainmodel/hetero_graph08_with_text_features.pkl',
    'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\hetero_graph08.pkl',
    # Include other graph paths as needed
]

video_folder = 'D:/CODE/multi-model knowledge graph multi-graph recommendation system/data/videos'
mapping_file = 'D:/CODE/multi-model knowledge graph multi-graph recommendation system/data/ml-youtube_cleaned.csv'
output_file = 'audio_features.pkl'
extract_and_save_audio_features(video_folder, mapping_file, output_file)
audio_features_dict = load_audio_features('audio_features.pkl')

def save_graph(graph, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)


for graph_path in graph_paths:
    graph = load_graph_with_images(graph_path)
    add_audio_features_to_graph(graph, audio_features_dict)
    save_graph(graph, graph_path.replace('.pkl', '_with_audio_features.pkl'))

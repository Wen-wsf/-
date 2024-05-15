import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
import torch
from PIL import Image
from moviepy.editor import VideoFileClip
import numpy as np
from tqdm import tqdm
import os
import pickle
import dgl

def load_youtube_to_movie_mapping(filepath):
    df = pd.read_csv(filepath)
    movie_to_youtube = pd.Series(df.youtubeId.values, index=df.movieId.astype(str)).to_dict()
    return movie_to_youtube

def load_pretrained_resnet50():
    resnet50 = models.resnet50(pretrained=True)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
    resnet50.eval()
    resnet50 = resnet50.to('cuda')
    return resnet50

def preprocess_frames(frames, size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch_tensor = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames])
    return batch_tensor.to('cuda')

def extract_features_from_video(video_path, model, preprocess, start_sec=10, end_sec=60, fps=1, batch_size=32):
    try:
        clip = VideoFileClip(video_path).subclip(start_sec, end_sec)
        total_frames = int((end_sec - start_sec) * fps)
        frame_features = []
        batch = []
        for frame in tqdm(clip.iter_frames(fps=fps, dtype='uint8'), total=total_frames, desc=f"Processing {os.path.basename(video_path)}"):
            batch.append(frame)
            if len(batch) == batch_size:
                img_tensor = preprocess(batch)
                with torch.no_grad():
                    features = model(img_tensor)
                    features = features.view(features.size(0), -1)
                frame_features.extend(features.cpu().numpy())
                batch = []
        if batch:
            img_tensor = preprocess(batch)
            with torch.no_grad():
                features = model(img_tensor)
                features = features.view(features.size(0), -1)
            frame_features.extend(features.cpu().numpy())
        clip.close()
        return np.mean(frame_features, axis=0)
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def add_features_to_graph(g, features_dict):
    num_movies = g.number_of_nodes('movieimage')
    num_features = 2048
    features_tensor = torch.zeros((num_movies, num_features))
    for movie_id, features in features_dict.items():
        try:
            idx = g.nodes['movie'].data['movie_id'].tolist().index(int(movie_id))
            if features is not None:
                feature_tensor = torch.tensor(features, dtype=torch.float32)
                if feature_tensor.numel() == num_features:
                    features_tensor[idx] = feature_tensor
                else:
                    print(f"特征维度不一致: {feature_tensor.numel()} 期望: {num_features}")
            else:
                print(f"无特征数据: movie_id {movie_id}")
        except ValueError:
            print(f"movie_id {movie_id} 在图中找不到对应的索引")
    g.nodes['movieimage'].data['features'] = features_tensor
    print("特征已成功添加到图中。")

movie_to_youtube = load_youtube_to_movie_mapping('D:\\CODE\\multi-model knowledge graph multi-graph recommendation system\\data\\ml-youtube_cleaned.csv')
resnet50 = load_pretrained_resnet50()
video_folder = 'D:\\CODE\\multi-model knowledge graph multi-graph recommendation system\\data\\videos'
graphs_to_process = ['D:\\CODE\\multi-model knowledge graph multi-graph recommendation system\\code\\mainmodel\\hetero_graph03.pkl', 'D:\\CODE\\multi-model knowledge graph multi-graph recommendation system\\code\\mainmodel\\hetero_graph05.pkl', 'D:\\CODE\\multi-model knowledge graph multi-graph recommendation system\\code\\mainmodel\\hetero_graph08.pkl']

all_movie_ids = set()
for graph_path in graphs_to_process:
    with open(graph_path, 'rb') as f:
        hetero_graph = pickle.load(f)
    user_movie_edges = hetero_graph.edges(etype=('user', 'rates', 'movie'))
    for user_id in torch.unique(user_movie_edges[0]):
        movie_ids = user_movie_edges[1][user_movie_edges[0] == user_id]
        for movie_id in movie_ids:
            actual_movie_id = hetero_graph.nodes['movie'].data['movie_id'][movie_id].item()
            all_movie_ids.add(str(actual_movie_id))

features_dict = {}
for actual_movie_id in all_movie_ids:
    youtube_id = movie_to_youtube.get(actual_movie_id)
    if youtube_id:
        video_path = os.path.join(video_folder, f"{youtube_id}.mp4")
        if os.path.exists(video_path):
            features = extract_features_from_video(video_path, resnet50, preprocess_frames)
            if features is not None:
                features_dict[actual_movie_id] = features

for graph_path in graphs_to_process:
    with open(graph_path, 'rb') as f:
        hetero_graph = pickle.load(f)
    add_features_to_graph(hetero_graph, features_dict)
    updated_graph_path = graph_path.replace('.pkl', '_with_features.pkl')
    with open(updated_graph_path, 'wb') as f:
        pickle.dump(hetero_graph, f)

import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
import torch.optim as optim
import dgl.function as fn
from dgl.nn import GraphConv, GATConv
import pickle
# 加载数据
data_path = 'D:\\CODE\\multi-model knowledge graph multi-graph recommendation system\\data\\cleanuser_rating.csv'
df = pd.read_csv(data_path)

# 将用户ID和电影ID编码为连续整数
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df['userId'] = user_encoder.fit_transform(df['userId'])
df['movieId'] = item_encoder.fit_transform(df['movieId'])

# 分割数据为训练集和验证集
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

class BPRDataset(Dataset):
    def __init__(self, user_ids, item_ids, num_items, num_negatives=1):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.num_items = num_items
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user = self.user_ids[idx]
        pos_item = self.item_ids[idx]
        neg_items = []
        for _ in range(self.num_negatives):
            neg_item = np.random.randint(0, self.num_items)
            while neg_item == pos_item:
                neg_item = np.random.randint(0, self.num_items)
            neg_items.append(neg_item)
        return user, pos_item, np.array(neg_items)

# 示例：将 DataFrame 列转换为适用于 BPRDataset 的格式
num_users = df['userId'].nunique()
num_items = df['movieId'].nunique()

# 创建 BPRDataset 实例
train_dataset = BPRDataset(df['userId'].values, df['movieId'].values, num_items, num_negatives=1)
valid_dataset = DataLoader(train_dataset, batch_size=128, shuffle=True)
train_loader = BPRDataset(df['userId'].values, df['movieId'].values, num_items, num_negatives=1)
valid_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# 对于验证集，您可以继续使用现有逻辑，或者也适配为 BPR 格式。


# Path to your saved graph
graph_path = r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\data\dataprocess\hetero_graph_with_images_text_audio.pkl'

# Load the graph
with open(graph_path, 'rb') as f:
    graph = pickle.load(f)

##############################################################################
class HeteroGraphConv(nn.Module):
    def __init__(self, num_users, num_items, feature_sizes, embedding_dim=128, id_embedding_size=16, num_heads=4):
        super(HeteroGraphConv, self).__init__()
        # Embedding layers for users and items
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Embedding layers for ID-based nodes
        self.embeddings = nn.ModuleDict({
            node_type: nn.Embedding(10000, id_embedding_size) for node_type in feature_sizes if node_type not in ['movie', 'movieimage', 'movietext', 'movieaudio']
        })

        # Transformation layers for modal-specific features
        self.image_transform = nn.Linear(feature_sizes['movieimage'], 32)
        self.text_transform = nn.Linear(feature_sizes['movietext'], 32)
        self.audio_transform = nn.Linear(feature_sizes['movieaudio'], 32)

        # GAT layers for each modality
        self.gat_layers = nn.ModuleDict({
            'movieimage': GATConv(32, 32, num_heads=num_heads),
            'movietext': GATConv(32, 32, num_heads=num_heads),
            'movieaudio': GATConv(32, 32, num_heads=num_heads)
        })

        # Additional layer to process concatenated features
        self.fc = nn.Linear(32 * 3 * num_heads, 128)  # Adjust the size according to the output of GAT

    def forward(self, user_ids, item_ids, graph=None, features=None):
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        if graph is not None and features is not None:
            # Process modal-specific features if provided
            image_features = F.relu(self.image_transform(features['movieimage']))
            text_features = F.relu(self.text_transform(features['movietext']))
            audio_features = F.relu(self.audio_transform(features['movieaudio']))
            image_features = self.gat_layers['movieimage'](graph, image_features).flatten(1)
            text_features = self.gat_layers['movietext'](graph, text_features).flatten(1)
            audio_features = self.gat_layers['movieaudio'](graph, audio_features).flatten(1)
            combined_features = torch.cat((image_features, text_features, audio_features), dim=1)
            combined_features = F.relu(self.fc(combined_features))
            return user_emb, item_emb, combined_features
        return user_emb, item_emb


def dcg_at_k(scores, k=5):
    ranks = torch.log2(torch.arange(2, k+2).float()).to(scores.device)  # Log term in DCG formula
    return (scores / ranks).sum()

def ndcg_at_k(predicted_scores, true_relevance, k=5):
    _, indices = torch.sort(predicted_scores, descending=True)
    true_sorted_by_pred = true_relevance[indices]
    ideal_sorted, _ = torch.sort(true_relevance, descending=True)

    dcg = dcg_at_k(true_sorted_by_pred[:k])
    idcg = dcg_at_k(ideal_sorted[:k])
    return (dcg / idcg).item() if idcg > 0 else 0.0
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
from dgl.nn import GraphConv, GATConv

class BPRDataset(Dataset):
    def __init__(self, user_item_pairs, total_items):
        self.user_item_pairs = user_item_pairs
        self.total_items = total_items

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, pos_item = self.user_item_pairs[idx]
        # 选择多个负样本，可能增强模型学习复杂场景的能力
        neg_items = []
        for _ in range(5):  # 比如选择5个负样本
            neg_item = np.random.randint(0, self.total_items)
            while neg_item == pos_item or neg_item in neg_items:
                neg_item = np.random.randint(0, self.total_items)
            neg_items.append(neg_item)
        return user, pos_item, neg_items

    
graph_path = r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\graph\hetero_graph03_with_V_T_A_features.pkl'
with open(graph_path, 'rb') as f:
    graph = pickle.load(f)
data_path = 'D:\\CODE\\multi-model knowledge graph multi-graph recommendation system\\data\\cleanuser_rating.csv'
df = pd.read_csv(data_path)
total_items = df['movieId'].max() + 1  # 确保这是正确的电影总数
print("Total items (movies):", total_items)
# 将用户ID和电影ID编码为连续整数
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df['userId'] = user_encoder.fit_transform(df['userId'])
df['movieId'] = item_encoder.fit_transform(df['movieId'])
num_users = df['userId'].nunique()  # 计算唯一的用户数量
num_items = df['movieId'].nunique()  # 计算唯一的电影数量
# 分割数据为训练集和验证集
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# 转换为用户-项目配对
train_pairs = list(zip(train_df['userId'].values, train_df['movieId'].values))
valid_pairs = list(zip(valid_df['userId'].values, valid_df['movieId'].values))

# 总电影数
total_items = df['movieId'].max() + 1

# 创建数据集
train_dataset = BPRDataset(train_pairs, total_items)
valid_dataset = BPRDataset(valid_pairs, total_items)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
class MMGAT(nn.Module):
    def __init__(self, num_users, num_items, feature_sizes, embedding_dim=128, id_embedding_size=16, num_heads=4):
        super(MMGAT, self).__init__()
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
            image_features = F.leaky_relu(self.image_transform(features['movieimage']))
            text_features = F.leaky_relu(self.text_transform(features['movietext']))
            audio_features = F.leaky_relu(self.audio_transform(features['movieaudio']))
            image_features = self.gat_layers['movieimage'](graph, image_features).flatten(1)
            text_features = self.gat_layers['movietext'](graph, text_features).flatten(1)
            audio_features = self.gat_layers['movieaudio'](graph, audio_features).flatten(1)
            combined_features = torch.cat((image_features, text_features, audio_features), dim=1)
            combined_features = F.leaky_relu(self.fc(combined_features))
            return user_emb, item_emb, combined_features
        return user_emb, item_emb
def bpr_loss(users, pos_items, neg_items, lambda_reg, model):
    # 獲取用戶和項目的嵌入
    user_embeddings = model.embedding_user(users)
    pos_item_embeddings = model.embedding_item(pos_items)
    neg_item_embeddings = model.embedding_item(neg_items)
    
    # 計算用戶對正樣本和負樣本的偏好預測
    pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
    neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)
    
    # 計算 BPR 損失
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
    
    # 添加 L2 正則化
    reg_loss = lambda_reg * (user_embeddings.norm(2).pow(2) + 
                             pos_item_embeddings.norm(2).pow(2) +
                             neg_item_embeddings.norm(2).pow(2))
    
    return loss + reg_loss
import torch
def dcg_at_k(scores, k=10):
    ranks = torch.log2(torch.arange(2, k+2).float()).to(scores.device)  # Log term in DCG formula
    return (scores[:k] / ranks).sum()  # Only consider the top k scores


def ndcg_at_k(predicted_scores, true_relevance, k=5):
    _, indices = torch.sort(predicted_scores, descending=True)
    true_sorted_by_pred = true_relevance[indices]
    ideal_sorted, _ = torch.sort(true_relevance, descending=True)

    dcg = dcg_at_k(true_sorted_by_pred[:k])
    idcg = dcg_at_k(ideal_sorted[:k])
    return (dcg / idcg).item() if idcg > 0 else 0.0
def train_model(model, train_loader, epochs, lambda_reg, optimizer, device, validation_loader=None):
    model.to(device)  # 确保模型在正确的设备上
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for user_ids, pos_item_ids, neg_item_ids in train_loader:
            user_ids, pos_item_ids, neg_item_ids = user_ids.to(device), pos_item_ids.to(device), neg_item_ids.to(device)
            
            optimizer.zero_grad()
            user_embeddings, pos_item_embeddings = model(user_ids, pos_item_ids)
            _, neg_item_embeddings = model(user_ids, neg_item_ids)
            
            # BPR Loss
            pos_scores = (user_embeddings * pos_item_embeddings).sum(dim=1)
            neg_scores = (user_embeddings * neg_item_embeddings).sum(dim=1)
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
            
            reg_loss = lambda_reg * (user_embeddings.norm(2).pow(2) +
                                     pos_item_embeddings.norm(2).pow(2) +
                                     neg_item_embeddings.norm(2).pow(2))
            total_loss_val = loss + reg_loss
            total_loss_val.backward()
            optimizer.step()
            total_loss += total_loss_val.item()

        print(f'Epoch {epoch+1}: Average Training Loss: {total_loss / len(train_loader)}')

        if validation_loader:
            evaluate_metrics(model, validation_loader)
def evaluate_metrics(model, validation_loader, k=10):
    device = next(model.parameters()).device
    model.eval()
    total_recall = 0
    num_batches = 0

    with torch.no_grad():
        for user_ids, pos_item_ids, neg_item_ids in validation_loader:
            user_ids, pos_item_ids, neg_item_ids = user_ids.to(device), pos_item_ids.to(device), neg_item_ids.to(device)
            
            user_embeddings = model.user_embeddings(user_ids)
            pos_item_embeddings = model.item_embeddings(pos_item_ids)
            neg_item_embeddings = model.item_embeddings(neg_item_ids)

            pos_scores = (user_embeddings * pos_item_embeddings).sum(dim=1)
            neg_scores = (user_embeddings * neg_item_embeddings).sum(dim=1)

            # Concatenate positive and negative scores and calculate top-k items
            scores = torch.cat((pos_scores, neg_scores))
            top_k_scores, top_k_indices = torch.topk(scores, k)
            
            # Check if any of the top-k indices are within the range of positive items
            num_positives = pos_item_ids.size(0)
            top_k_is_pos = top_k_indices < num_positives
            recall_batch = top_k_is_pos.float().sum() / num_positives
            total_recall += recall_batch
            num_batches += 1

        average_recall = total_recall / num_batches
        print(f'Average Recall@{k}: {average_recall:.4f}')



# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_sizes = {
    'movie': 131263,    # 如果电影节点存储所有多媒体特征的总和
    'actor': 50,      # 演员的嵌入维度
    'actress': 50,    # 女演员的嵌入维度
    'director': 50,   # 导演的嵌入维度
    'producer': 50,   # 制片的嵌入维度
    'movieimage': 2048,  # 图像特征维度
    'movietext': 384,    # 文字特征维度
    'movieaudio': 128,   # 音频特征维度
    'user': 12171         # 假设用戶的嵌入维度，可能需要根据用户数量和系统复杂性进行调整
}
# 初始化和配置模型，加载数据，然后开始训练和评估
model = MMGAT(num_users, num_items, feature_sizes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, 10, 0.01, optimizer, device, valid_loader)

# 保存模型
torch.save(model.state_dict(), 'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\data\model.pth')

# 加载模型
model.load_state_dict(torch.load('D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\data\model.pth'))
model.eval()



import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import scatter_
from torch.nn import Parameter
from Basicgcn import Base_gcn
import pdb
import random
from matplotlib import cbook
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import random
class GCN(torch.nn.Module):
    """
        初始化GCN模型。

        Args:
            datasets (str): 數據集的名稱。
            batch_size (int): 批次大小。
            num_user (int): 用戶的數量。
            num_item (int): 商品的數量。
            dim_id (int): ID 的維度。
            aggr_mode (str): 聚合模式。
            num_layer (int): GCN 層的數量。
            has_id (bool): 是否有 ID。
            dropout (float): Dropout 機率。
            dim_latent (int, optional): 潛在空間的維度。預設為 None。
            device (str, optional): 設備類型。預設為 None。
            features (torch.Tensor, optional): 圖形特徵。預設為 None。
        """
    def __init__(self,datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None,device = None,features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        if self.datasets =='tiktok' or self.datasets =='tiktok_new' or self.datasets == 'cold_tiktok':
             self.dim_feat = 128
        elif self.datasets == 'Movielens' or self.datasets == 'cold_movie':
             self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_latent),dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4*self.dim_latent)
            self.MLP_1 = nn.Linear(4*self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_feat),dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop,edge_index,features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)

        x_hat =h + x +h_1
        return x_hat, self.preference


class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode,dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features,user_graph,user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix,u_features)
        u_pre = u_pre.squeeze()
        return u_pre

class DualGNN(torch.nn.Module):
    # DualGNN是一個基於圖神經網絡的模型，用於推薦系統。以下是該類的初始化函數。
    def __init__(self, features, edge_index, batch_size, num_user, num_item, aggr_mode, construction,
                    num_layer, has_id, dim_x, reg_weight, drop_rate, sampling, user_item_dict, dataset, cold_start, device=None):
        super(DualGNN, self).__init__()

        # 初始化模型所需的參數
        self.batch_size = batch_size  # 設置批次大小
        self.num_user = num_user  # 用戶數量
        self.num_item = num_item  # 商品數量
        self.k = sampling  # 抽樣數量
        self.aggr_mode = aggr_mode  # 聚合模式
        self.num_layer = num_layer  # 層數
        self.cold_start = cold_start  # 冷啟動標誌
        self.dataset = dataset  # 數據集
        self.construction = construction  # 圖構建方式
        self.reg_weight = reg_weight  # 正則化權重
        self.user_item_dict = user_item_dict  # 用戶-商品字典
        self.drop_rate = drop_rate  # 丟棄率
        self.v_rep = None  # 用戶表示
        self.a_rep = None  # 商品表示
        self.t_rep = None  # 時間表示
        self.device = device  # 設備
        self.v_preference = None  # 用戶偏好
        self.a_preference = None  # 商品偏好
        self.t_preference = None  # 時間偏好
        self.dim_latent = 64  # 潛在維度
        self.dim_feat = 128  # 特徵維度
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)  # 用戶MLP
        self.MLP_a = nn.Linear(self.dim_latent, self.dim_latent, bias=False)  # 商品MLP
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)  # 時間MLP

        v_feat, a_feat, t_feat = features  # 提取特徵
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)  # 邊索引
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)  # 將邊索引與其轉置合併
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_user, 3, 1), dtype=torch.float32, requires_grad=True)))  # 用戶權重
        self.weight_u.data = F.softmax(self.weight_u.data, dim=1)  # 對用戶權重進行softmax正規化

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_item, 3, 1), dtype=torch.float32, requires_grad=True)))  # 商品權重
        self.weight_i.data = F.softmax(self.weight_i.data, dim=1)  # 對商品權重進行softmax正規化

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)  # 商品索引
        index = []  # 索引列表
        # 遍歷所有商品，設置商品索引和建立索引列表
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)

        # 設置丟棄百分比和不同類型的丟棄百分比
        self.drop_percent = drop_rate
        self.single_percent = 1
        self.double_percent = 0

        # 從商品索引中隨機選擇一部分商品進行丟棄
        drop_item = torch.tensor(np.random.choice(self.item_index, int(self.num_item*self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent*len(drop_item))]
        drop_item_double = drop_item[int(self.single_percent*len(drop_item)):]

        # 將丟棄的商品分配到不同類型的節點索引中
        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1/ 3)]
        self.dropa_node_idx_single = drop_item_single[int(len(drop_item_single) * 1/ 3):int(len(drop_item_single) * 2/ 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        # 將單一類型的丟棄節點索引賦值給整體丟棄節點索引
        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropa_node_idx = self.dropa_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        # 初始化節點遮罩計數列表
        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()

        # 計算每個節點的邊的數量
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1

        # 生成用於節點遮罩的列表
        mask_dropv = []
        mask_dropa = []
        mask_dropt = []

        # 根據節點類型填充遮罩列表
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropa.extend(temp_false) if idx in self.dropa_node_idx else mask_dropa.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        # 根據節點遮罩，選擇相應的邊索引
        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropa = edge_index[mask_dropa]
        edge_index_dropt = edge_index[mask_dropt]

        # 將節點遮罩應用於邊索引，並將其轉置
        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropa = torch.tensor(edge_index_dropa).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        # 將邊索引與其轉置合併
        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropa = torch.cat((self.edge_index_dropa, self.edge_index_dropa[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        # 如果數據集是'Movielens'或'cold_movie'，則初始化用戶MLP和特徵張量，並創建對應的GCN模型
        if self.dataset == 'Movielens' or self.dataset == 'cold_movie':
            self.MLP_user = nn.Linear(self.dim_latent*3, self.dim_latent)
            self.v_feat = torch.tensor(v_feat, dtype=torch.float).to(self.device)
            self.a_feat = torch.tensor(a_feat, dtype=torch.float).to(self.device)
            self.t_feat = torch.tensor(t_feat, dtype=torch.float).to(self.device)
            self.v_drop_ze = torch.zeros(len(self.dropv_node_idx),self.v_feat.size(1)).to(self.device)
            self.a_drop_ze = torch.zeros(len(self.dropa_node_idx),self.a_feat.size(1)).to(self.device)
            self.t_drop_ze = torch.zeros(len(self.dropt_node_idx),self.t_feat.size(1)).to(self.device)

            self.v_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_x, self.aggr_mode,
                            num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                            device=self.device, features=self.v_feat)  # 256)
            self.a_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_x, self.aggr_mode,
                            num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                            device=self.device, features=self.a_feat)
            self.t_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_x, self.aggr_mode,
                            num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                            device=self.device, features=self.t_feat)
                # 初始化用戶圖
        self.user_graph = User_Graph_sample(num_user, 'add', self.dim_latent)

        # 初始化結果嵌入權重
        self.result_embed = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)


    def draw_zhexian(self, index, values, path):
        """
        用於繪製折線圖的方法。

        參數：
            index：x 軸索引值
            values：y 軸數值
            path：保存圖像的路徑

        返回：
            無（保存圖像）
        """
        fig, ax = plt.subplots()
        x = index
        y = values
        ax.plot(x, y, 'ko-')
        plt.savefig(path)

    def forward(self, user_nodes, pos_item_nodes, neg_item_nodes, user_graph, user_weight_matrix, user_cons=None):
        # 如果數據集是'tiktok'或'tiktok_new'，計算文本特徵
        if self.dataset == 'tiktok' or self.dataset == 'tiktok_new':
            self.t_feat = scatter_('mean', self.word_embedding(self.word_tensor[1]), self.word_tensor[0]).to(self.device)
        
        # 計算視覺、音頻和文本特徵的表示和偏好
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
        self.a_rep, self.a_preference = self.a_gcn(self.edge_index_dropa, self.edge_index, self.a_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)
        
        # 將視覺、音頻和文本特徵的表示相加
        representation = self.v_rep + self.a_rep + self.t_rep
        
        # 根據不同的構建方法計算用戶表示
        if self.construction == 'weighted_sum':
            # 加權求和方式
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.a_rep = torch.unsqueeze(self.a_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            user_rep = torch.matmul(
                torch.cat((self.v_rep[:self.num_user], self.a_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2),
                self.weight_u)
            user_rep = torch.squeeze(user_rep)
        
        if self.construction == 'mean':
            # 平均方式
            user_rep = (self.v_rep[:self.num_user] + self.a_rep[:self.num_user] + self.t_rep[:self.num_user]) / 3
        
        if self.construction == 'max':
            # 最大值方式
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.a_rep = torch.unsqueeze(self.a_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            user_rep = torch.cat((self.v_rep[:self.num_user], self.a_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
            user_rep = self.weight_u.transpose(1, 2) * user_rep
            user_rep = torch.max(user_rep, dim=2).values
        
        if self.construction == 'cat_mlp':
            # 連接後經過MLP
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.a_rep = torch.unsqueeze(self.a_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            user_rep = torch.cat((self.v_rep[:self.num_user], self.a_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
            user_rep = self.weight_u.transpose(1, 2) * user_rep
            user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1], user_rep[:, :, 2]), dim=1)
            user_rep = self.MLP_user(user_rep)
        
        # 計算項目表示
        item_rep = representation[self.num_user:]
        
        # 更新用戶表示
        h_u1 = self.user_graph(user_rep, user_graph, user_weight_matrix)
        user_rep = user_rep + h_u1
        
        # 合併用戶和項目表示
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        
        # 提取當前批次的用戶和項目表示
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        
        # 計算正樣本和負樣本的分數
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        
        return pos_scores, neg_scores

    def loss_constra(self, user_rep, user_graph, user_weight_matrix, user_cons):
        """
        定義約束損失的方法。

        參數：
            user_rep：用戶表示
            user_graph：用戶圖
            user_weight_matrix：用戶權重矩陣
            user_cons：用戶約束（指示用戶標誌）

        返回：
            loss_constra：約束損失值
        """
        loss_constra = 0
        neg_scores = torch.exp((user_rep[user_cons] * user_rep[self.user_index]).sum(dim=2)).sum(dim=1)
        pos_scores = torch.exp((user_rep[user_graph] * user_rep[self.user_index]).sum(dim=2)).sum(dim=1)
        loss_constra = -torch.log2(pos_scores / (pos_scores + neg_scores)).mean()
        return loss_constra

    def loss(self, data, user_graph, user_weight_matrix, user_cons=None):
        """
        定義損失函數的方法。

        參數：
            data：數據
            user_graph：用戶圖
            user_weight_matrix：用戶權重矩陣
            user_cons：用戶約束（預設為None）

        返回：
            loss_value：損失值
            reg_loss：正則化損失
        """
        user, pos_items, neg_items = data
        pos_scores, neg_scores = self.forward(user.to(self.device), pos_items.to(self.device), neg_items.to(self.device), user_graph, user_weight_matrix.to(self.device), user_cons)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss_v = (self.v_preference[user.to(self.device)] ** 2).mean()
        reg_embedding_loss_a = (self.a_preference[user.to(self.device)] ** 2).mean()
        reg_embedding_loss_t = (self.t_preference[user.to(self.device)] ** 2).mean()
        
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_a + reg_embedding_loss_t)
        
        if self.construction == 'weighted_sum':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
            reg_loss += self.reg_weight * (self.weight_i ** 2).mean()
        elif self.construction == 'cat_mlp':
            reg_loss += self.reg_weight * (self.MLP_user.weight ** 2).mean()
        
        return loss_value + reg_loss, reg_loss

    def gene_ranklist(self, val_data, test_data, step=20000, topk=10):
        """
        生成排名列表的方法。

        參數：
            val_data：驗證數據
            test_data：測試數據
            step：步長（預設為20000）
            topk：前K個（預設為10）

        返回：
            all_index_of_rank_list_tra：所有排名列表（訓練集）
            all_index_of_rank_list_vt：所有排名列表（驗證集）
            all_index_of_rank_list_tt：所有排名列表（測試集）
        """
        user_tensor = self.result_embed[:self.num_user].cpu()
        item_tensor = self.result_embed[self.num_user:self.num_user + self.num_item].cpu()
        start_index = 0
        end_index = self.num_user if step == None else step

        all_index_of_rank_list_tra = torch.LongTensor([])
        all_index_of_rank_list_vt = torch.LongTensor([])
        all_index_of_rank_list_tt = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix_tra = torch.matmul(temp_user_tensor, item_tensor.t())
            score_matrix_vt = score_matrix_tra.clone().detach()
            score_matrix_tt = score_matrix_tra.clone().detach()

            _, index_of_rank_list_tra = torch.topk(score_matrix_tra, topk)
            all_index_of_rank_list_tra = torch.cat((all_index_of_rank_list_tra, index_of_rank_list_tra.cpu() + self.num_user),
                                                dim=0)
            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col)) - self.num_user
                    score_matrix_vt[row][col] = 1e-5
                    score_matrix_tt[row][col] = 1e-5
            for i in range(len(val_data)):
                if val_data[i][0] >= start_index and val_data[i][0] < end_index:
                    row = val_data[i][0] - start_index
                    col = torch.LongTensor(list(val_data[i][1:])) - self.num_user
                    score_matrix_tt[row][col] = 1e-5
            for i in range(len(test_data)):
                if test_data[i][0] >= start_index and test_data[i][0] < end_index:
                    row = test_data[i][0] - start_index
                    col = torch.LongTensor(list(test_data[i][1:])) - self.num_user
                    score_matrix_vt[row][col] = 1e-5
            _, index_of_rank_list_vt = torch.topk(score_matrix_vt, topk)
            all_index_of_rank_list_vt = torch.cat((all_index_of_rank_list_vt, index_of_rank_list_vt.cpu() + self.num_user),
                                                dim=0)
            _, index_of_rank_list_tt = torch.topk(score_matrix_tt, topk)
            all_index_of_rank_list_tt = torch.cat((all_index_of_rank_list_tt, index_of_rank_list_tt.cpu() + self.num_user),
                                                dim=0)

            start_index = end_index

            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        return all_index_of_rank_list_tra, all_index_of_rank_list_vt, all_index_of_rank_list_tt

    def gene_ranklist_cold(self, val_data, test_data, step=20000, topk=10):
        """
        生成冷启动情况下的排名列表的方法。

        參數：
            val_data：驗證數據
            test_data：測試數據
            step：步長（預設為20000）
            topk：前K個（預設為10）

        返回：
            all_index_of_rank_list_tra：所有排名列表（訓練集）
            all_index_of_rank_list_vt：所有排名列表（驗證集）
            all_index_of_rank_list_tt：所有排名列表（測試集）
        """
        # 提取用戶和商品的表示向量
        user_tensor = self.result_embed[:self.num_user].cpu()
        item_tensor = self.result_embed[self.num_user:].cpu()
        # 提取冷启动商品的表示向量
        item_tensor_cold = item_tensor[self.num_item:]

        start_index = 0
        end_index = self.num_user if step == None else step

        all_index_of_rank_list_tra = torch.LongTensor([])
        all_index_of_rank_list_vt = torch.LongTensor([])
        all_index_of_rank_list_tt = torch.LongTensor([])
        # 分批次處理用戶
        while end_index <= self.num_user and start_index < end_index:
            # 提取當前批次的用戶表示向量
            temp_user_tensor = user_tensor[start_index:end_index]
            # 計算用戶與冷启动商品之間的分數矩陣
            score_matrix_tra = torch.matmul(temp_user_tensor, item_tensor_cold.t())
            score_matrix_vt = score_matrix_tra.clone().detach()
            score_matrix_tt = score_matrix_tra.clone().detach()

            # 獲取前K個排名
            _, index_of_rank_list_tra = torch.topk(score_matrix_tra, topk)
            all_index_of_rank_list_tra = torch.cat((all_index_of_rank_list_tra, index_of_rank_list_tra.cpu() + self.num_user),
                                            dim=0)
            # 遍歷用戶-商品字典，將已購買的商品的分數設置為極小值
            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col)) - (self.num_user + self.num_item)
                    # 檢查索引是否越界
                    if torch.gt(torch.LongTensor(list(col)), -1).all() == False:
                        continue
                    score_matrix_vt[row][col] = 1e-5
                    score_matrix_tt[row][col] = 1e-5
            # 將驗證集中的商品對應的分數設置為極小值
            for i in range(len(val_data)):
                if val_data[i][0] >= start_index and val_data[i][0] < end_index:
                    row = val_data[i][0] - start_index
                    col = torch.LongTensor(list(val_data[i][1:])) - (self.num_user + self.num_item)
                    # 檢查索引是否越界
                    if torch.gt(torch.LongTensor(list(col)), -1).all() == False:
                        continue
                    score_matrix_tt[row][col] = 1e-5
            # 將測試集中的商品對應的分數設置為極小值
            for i in range(len(test_data)):
                if test_data[i][0] >= start_index and test_data[i][0] < end_index:
                    row = test_data[i][0] - start_index
                    col = torch.LongTensor(list(test_data[i][1:])) - (self.num_user + self.num_item)
                    # 檢查索引是否越界
                    if torch.gt(torch.LongTensor(list(col)), -1).all() == False:
                        continue
                    score_matrix_vt[row][col] = 1e-5 

            # 獲取前K個排名
            _, index_of_rank_list_vt = torch.topk(score_matrix_vt, topk)
            all_index_of_rank_list_vt = torch.cat((all_index_of_rank_list_vt, index_of_rank_list_vt.cpu() + self.num_user),
                                            dim=0)
            _, index_of_rank_list_tt = torch.topk(score_matrix_tt, topk)
            all_index_of_rank_list_tt = torch.cat((all_index_of_rank_list_tt, index_of_rank_list_tt.cpu() + self.num_user),
                                            dim=0)

            start_index = end_index

            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user
            
        return all_index_of_rank_list_tra, all_index_of_rank_list_vt, all_index_of_rank_list_tt

    # @decorator
    def accuracy(self,rank_list, topk=10):
    #     """
    # 計算排名列表的準確性。

    # 參數：
    #     rank_list：排名列表
    #     topk：前K個（預設為10）

    # 返回：
    #     precision_10：Top-10精確度
    #     recall_10：Top-10召回率
    #     ndcg_10：Top-10 NDCG
    #     precision_5：Top-5精確度
    #     recall_5：Top-5召回率
    #     ndcg_5：Top-5 NDCG
    #     precision_1：Top-1精確度
    #     recall_1：Top-1召回率
    #     ndcg_1：Top-1 NDCG
    # """
        length = self.num_user
        precision_10 = recall_10 = ndcg_10 = 0.0
        precision_5 = recall_5 = ndcg_5 = 0.0
        precision_1 = recall_1 = ndcg_1 = 0.0
        for row, col in self.user_item_dict.items():
            # col = np.array(list(col))-self.num_user
            user = row
            pos_items = set(col)
            # print(pos_items)
            num_pos = len(pos_items)
            items_list_10 = rank_list[user].tolist()
            items_list_5 = items_list_10[:5]
            items_list_1 = items_list_10[:1]
            items_10 = set(items_list_10)
            items_5 = set(items_list_5)
            items_1 = set(items_list_1)

            num_hit_10 = len(pos_items.intersection(items_10))
            precision_10 += float(num_hit_10 / topk)
            recall_10 += float(num_hit_10 / num_pos)
            ndcg_score_10 = 0.0
            max_ndcg_score_10 = 0.0
            for i in range(min(num_hit_10, topk)):
                max_ndcg_score_10 += 1 / math.log2(i + 2)
            if max_ndcg_score_10 == 0:
                continue
            for i, temp_item in enumerate(items_list_10):
                if temp_item in pos_items:
                    ndcg_score_10 += 1 / math.log2(i + 2)
            ndcg_10 += ndcg_score_10 / max_ndcg_score_10

            num_hit_5 = len(pos_items.intersection(items_5))
            precision_5 += float(num_hit_5 / 5)
            recall_5 += float(num_hit_5 / num_pos)
            ndcg_score_5 = 0.0
            max_ndcg_score_5 = 0.0
            for i in range(min(num_hit_5, 5)):
                max_ndcg_score_5 += 1 / math.log2(i + 2)
            if max_ndcg_score_5 == 0:
                continue
            for i, temp_item in enumerate(items_list_5):
                if temp_item in pos_items:
                    ndcg_score_5 += 1 / math.log2(i + 2)
            ndcg_5 += ndcg_score_5 / max_ndcg_score_5

            num_hit_1 = len(pos_items.intersection(items_1))
            precision_1 += float(num_hit_1 / 1)
            recall_1 += float(num_hit_1 / num_pos)
            ndcg_score_1 = 0.0
            max_ndcg_score_1 = 0.0
            for i in range(min(num_hit_1, 1)):
                max_ndcg_score_1 += 1 / math.log2(i + 2)
            if max_ndcg_score_1 == 0:
                continue
            for i, temp_item in enumerate(items_list_1):
                if temp_item in pos_items:
                    ndcg_score_1 += 1 / math.log2(i + 2)
            ndcg_1 += ndcg_score_1 / max_ndcg_score_1

        return precision_10 / length, recall_10 / length, ndcg_10 / length, precision_5 / length, recall_5 / length, ndcg_5 / length, precision_1 / length, recall_1 / length, ndcg_1 / length

    def full_accuracy(self, val_data,rank_list,cold_start,topk=10):
    #      """
    # 計算完整的準確性。

    # 參數：
    #     val_data：驗證數據
    #     rank_list：排名列表
    #     cold_start：冷啟動標誌（1表示是冷啟動）
    #     topk：前K個（預設為10）

    # 返回：
    #     precision_10：Top-10精確度
    #     recall_10：Top-10召回率
    #     ndcg_10：Top-10 NDCG
    #     precision_5：Top-5精確度
    #     recall_5：Top-5召回率
    #     ndcg_5：Top-5 NDCG
    #     precision_1：Top-1精確度
    #     recall_1：Top-1召回率
    #     ndcg_1：Top-1 NDCG
    # """
        length = len(val_data)
        precision_10 = recall_10 = ndcg_10 = 0.0
        precision_5 = recall_5 = ndcg_5 = 0.0
        precision_1 = recall_1 = ndcg_1 = 0.0
        count = 0
        # pdb.set_trace()
        for data in val_data:
            user = data[0]
            pos_i = data[1:]
            pos_temp = []
            # pdb.set_trace()
            if len(pos_i)==0:
                length = length-1
                count+=1
                continue
            else:
                if cold_start == 1:
                    for item in pos_i:
                        # pdb.set_trace()
                        pos_temp.append(item-self.num_item)
                        
                else:
                    for item in pos_i:
                        # pdb.set_trace()
                        pos_temp.append(item)
                # pdb.set_trace()
            # print(pos_items)
            pos_items = set(pos_temp)

            num_pos = len(pos_items)
            items_list_10 = rank_list[user].tolist()
            items_list_5 = items_list_10[:5]
            items_list_1 = items_list_10[:1]
            items_10 = set(items_list_10)
            items_5 = set(items_list_5)
            items_1 = set(items_list_1)

            num_hit_10 = len(pos_items.intersection(items_10))
            precision_10 += float(num_hit_10 / topk)
            recall_10 += float(num_hit_10 / num_pos)
            ndcg_score_10 = 0.0
            max_ndcg_score_10 = 0.0
            for i in range(min(num_hit_10, topk)):
                max_ndcg_score_10 += 1 / math.log2(i + 2)
            if max_ndcg_score_10 == 0:
                continue
            for i, temp_item in enumerate(items_list_10):
                if temp_item in pos_items:
                    ndcg_score_10 += 1 / math.log2(i + 2)
            ndcg_10 += ndcg_score_10 / max_ndcg_score_10

            num_hit_5 = len(pos_items.intersection(items_5))
            precision_5 += float(num_hit_5 / 5)
            recall_5 += float(num_hit_5 / num_pos)
            ndcg_score_5 = 0.0
            max_ndcg_score_5 = 0.0
            for i in range(min(num_hit_5, 5)):
                max_ndcg_score_5 += 1 / math.log2(i + 2)
            if max_ndcg_score_5 == 0:
                continue
            for i, temp_item in enumerate(items_list_5):
                if temp_item in pos_items:
                    ndcg_score_5 += 1 / math.log2(i + 2)
            ndcg_5 += ndcg_score_5 / max_ndcg_score_5

            num_hit_1 = len(pos_items.intersection(items_1))
            precision_1 += float(num_hit_1 / 1)
            recall_1 += float(num_hit_1 / num_pos)
            ndcg_score_1 = 0.0
            max_ndcg_score_1 = 0.0
            for i in range(min(num_hit_1, 1)):
                max_ndcg_score_1 += 1 / math.log2(i + 2)
            if max_ndcg_score_1 == 0:
                continue
            for i, temp_item in enumerate(items_list_1):
                if temp_item in pos_items:
                    ndcg_score_1 += 1 / math.log2(i + 2)
            ndcg_1 += ndcg_score_1 / max_ndcg_score_1
        print(count)
        return precision_10 / length, recall_10 / length, ndcg_10 / length, precision_5 / length, recall_5 / length, ndcg_5 / length, precision_1 / length, recall_1 / length, ndcg_1 / length


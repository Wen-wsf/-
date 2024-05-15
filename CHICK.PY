import dgl
import pickle

with open('D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\graph\hetero_graph03_with_V_T_A_features.pkl', 'rb') as f:
    hetero_graph = pickle.load(f)

# 列印元圖，顯示所有的節點類型和邊類型
print("元圖結構:")
print(hetero_graph.metagraph().edges())

print("\n節點和邊的數量:")
for ntype in hetero_graph.ntypes:
    print(f"{ntype}節點數量: {hetero_graph.number_of_nodes(ntype)}")
for etype in hetero_graph.canonical_etypes:
    print(f"{etype}邊數量: {hetero_graph.number_of_edges(etype)}")

# 检檢查特定的邊類型是否存在
expected_edges = [
    ('user', 'rates', 'movie'),
    ('movie', 'similar', 'movie'),
    ('user', 'similar', 'user'),
    ('movie', 'has_image', 'movieimage'),
    ('movie', 'has_text', 'movietext'),
    ('movie', 'has_audio', 'movieaudio')
]

print("\n檢查預期的邊類型:")
for edge in expected_edges:
    if edge in hetero_graph.canonical_etypes:
        print(f"邊 {edge} 存在.")
    else:
        print(f"邊 {edge} 不存在.")

import dgl
import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from sklearn.model_selection import train_test_split

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True, norm='right')
        self.conv2 = GraphConv(hidden_feats, out_feats, allow_zero_in_degree=True, norm='right')
    
    def forward(self, g, inputs):
        # Usamos las ponderaciones de las aristas en la convolución
        h = self.conv1(g, inputs, edge_weight=g.edata['weight'])
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight=g.edata['weight'])
        return h
    


def predict_for_user(user_item_embeddings, item_embeddings):
    """
    Predice los ítems más probables para un usuario basado en su secuencia de interacciones.
    
    Args:
        user_sequence: Lista de ítems con los que el usuario ha interactuado.
        item_embeddings: Tensor de representaciones de los ítems.
        top_k: Número de ítems a recomendar.
    
    Returns:
        Lista de los top_k ítems recomendados.
    """
    # Calcular la similitud del coseno entre los ítems interactuados y todos los ítems
    #similarity_scores = torch.mm(user_item_embeddings, item_embeddings.t())
    similarity_scores=[]
   
    for i in range(len(item_embeddings)):
        similarity_scores.append(torch.cosine_similarity(user_item_embeddings,item_embeddings[i],0))
    return similarity_scores


# Función para cargar las secuencias desde un archivo .txt
def load_sequences_from_file(filename):
    sequences = []
    with open(filename, 'r') as file:
        for line in file:
            # Dividir la línea en elementos, omitir el primer elemento (ID de usuario)
            items = list(map(int, line.strip().split()[1:]))
            items = [x-1 for x in items]
            sequences.append(items)
            
    return sequences

# Cargar las secuencias desde el archivo .txt
filename = 'sequences.txt'  # Cambia esto por la ruta correcta del archivo
sequences = load_sequences_from_file(filename)


# Construcción del grafo
def build_item_graph(sequences):
    # Inicializamos una matriz de adyacencia para almacenar los pesos de las aristas
    num_items = max(max(seq) for seq in sequences)+1 # Asumiendo que los ítems son números consecutivos desde 0
    A = torch.zeros((num_items, num_items))

    # Procesamos cada secuencia de usuario
    for seq in sequences:
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                u, v = seq[i]-1, seq[j]-1
                distance = j - i +1 
                A[u-1, v-1] += 1 / distance
                A[v-1, u-1] += 1 / distance  # Grafo no dirigido
    
    # Convertimos la matriz de adyacencia en un grafo DGL
    src, dst = A.nonzero(as_tuple=True)
    weights = A[src, dst]
    
    g = dgl.graph((src, dst))
    g.edata['weight'] = weights
    
    return g


# Dividir las secuencias en entrenamiento y prueba
train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)


# Construir el grafo con las secuencias de entrenamiento
train_g = build_item_graph(train_sequences)
train_g = dgl.add_self_loop(train_g)

num_nodes = train_g.num_nodes()

# Inicializamos el GCN
gcn = GCN(in_feats=num_nodes, hidden_feats=8, out_feats=16)

# Usamos todas las representaciones como características iniciales
item_features = torch.eye(num_nodes)

# Pasamos el grafo y las características iniciales por el GCN para entrenamiento
item_embeddings = gcn(train_g, item_features)

# Supongamos que el entrenamiento es simplemente pasar el grafo, en un caso más complejo, deberías definir un optimizador y una función de pérdida.

# Función para obtener las predicciones
def get_predictions(model, g, item_features):
    with torch.no_grad():
        return model(g, item_features)

# Obtener las representaciones de los ítems para el conjunto de prueba
test_g = build_item_graph(test_sequences)  # Opcionalmente podrías usar todo el grafo, pero aquí mantenemos separado el conjunto de prueba
test_g = dgl.add_self_loop(test_g)
test_item_embeddings = get_predictions(gcn, test_g, item_features)



def precision_at_k(pred_scores, ground_truth, k):
    top_k_items = torch.topk(pred_scores, k).indices
    relevant_items = set(ground_truth)
    hits = sum([1 for item in top_k_items if item.item() in relevant_items])
    return hits / k

def hit_rate_at_k(pred_scores, ground_truth, k):
    top_k_items = torch.topk(pred_scores, k).indices
    relevant_items = set(ground_truth)
    hits = any([item.item() in relevant_items for item in top_k_items])
    return 1 if hits else 0

def recall_at_k(pred_scores, ground_truth, k):
    top_k_items = torch.topk(pred_scores, k).indices
    relevant_items = set(ground_truth)
    hits = sum([1 for item in top_k_items if item.item() in relevant_items])
    return hits / len(relevant_items)

def mrr(pred_scores, ground_truth):
    sorted_items = torch.argsort(pred_scores, descending=True)
    relevant_items = set(ground_truth)
    for rank, item in enumerate(sorted_items, 1):
        if item.item() in relevant_items:
            return 1 / rank
    return 0

def dcg_at_k(pred_scores, ground_truth, k):
    top_k_items = torch.topk(pred_scores, k).indices
    dcg = 0
    for i, item in enumerate(top_k_items):
        if item.item() in ground_truth:
            dcg += 1 / torch.log2(torch.tensor(i + 2))  # i+2 because index starts from 0
    return dcg

def ndcg_at_k(pred_scores, ground_truth, k):
    dcg = dcg_at_k(pred_scores, ground_truth, k)
    idcg = dcg_at_k(torch.tensor([1] * len(ground_truth)), ground_truth, k)
    return dcg / idcg if idcg > 0 else 0




# Métricas
def evaluate_metrics(test_sequences, item_embeddings, k):

    precision_list = []
    hit_rate_list = []
    recall_list = []
    mrr_list = []
    ndcg_list = []
    
    for seq in test_sequences:
        if len(seq) < 2:
            continue  # Saltar secuencias demasiado cortas para una evaluación significativa
        
        user_history = seq[:-4]
        ground_truth = seq[-3:]


        # Crear el vector de puntuaciones predichas
        user_embedding = item_embeddings[user_history].mean(dim=0)
        #pred_scores = torch.matmul(item_embeddings, user_embedding)
        
        pred_scores= predict_for_user(user_embedding,item_embeddings)

        # Calcular las métricas
        for item in user_history:
            pred_scores[item] = -float('inf')  # Asignar un valor muy bajo para 
        
        precision_list.append(precision_at_k(pred_scores, ground_truth, k))
        hit_rate_list.append(hit_rate_at_k(pred_scores, ground_truth, k))
        recall_list.append(recall_at_k(pred_scores, ground_truth, k))
        mrr_list.append(mrr(pred_scores, ground_truth))
        #ndcg_list.append(ndcg_at_k(pred_scores, ground_truth, k))
    
    # Promediar las métricas
    return {
        "Precision@K": np.mean(precision_list),
        "Hit Rate@K": np.mean(hit_rate_list),
        "Recall@K": np.mean(recall_list),
        "MRR": np.mean(mrr_list),
        #"NDCG@K": np.mean(ndcg_list)
    }

# Evaluar las métricas en el conjunto de prueba
k = 10  # Por ejemplo, top-5
metrics = evaluate_metrics(test_sequences, item_embeddings, k)

# Mostrar los resultados
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


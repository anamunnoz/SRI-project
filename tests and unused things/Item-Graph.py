import dgl
import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


# Función para cargar las secuencias desde un archivo .txt
def load_sequences_from_file(filename):
    sequences = []
    with open(filename, 'r') as file:
        for line in file:
            # Dividir la línea en elementos, omitir el primer elemento (ID de usuario)
            items = list(map(int, line.strip().split()[1:]))
            sequences.append(items)
    return sequences

# Cargar las secuencias desde el archivo .txt
filename = 'sequences.txt'  # Cambia esto por la ruta correcta del archivo
sequences = load_sequences_from_file(filename)


# Construcción del grafo
def build_item_graph(sequences):
    # Inicializamos una matriz de adyacencia para almacenar los pesos de las aristas
    num_items = max(max(seq) for seq in sequences)  # Asumiendo que los ítems son números consecutivos desde 0
    A = torch.zeros((num_items, num_items))

    # Procesamos cada secuencia de usuario
    for seq in sequences:
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                u, v = seq[i], seq[j]
                distance = j - i +1 
                A[u-1, v-1] += 1 / distance
                A[v-1, u-1] += 1 / distance  # Grafo no dirigido
    
    # Convertimos la matriz de adyacencia en un grafo DGL
    src, dst = A.nonzero(as_tuple=True)
    weights = A[src, dst]
    
    g = dgl.graph((src, dst))
    g.edata['weight'] = weights
    
    return g




# Construimos el grafo con las secuencias
g = build_item_graph(sequences)

 
# Añadir auto-bucles
g = dgl.add_self_loop(g)

# Número de nodos en el grafo
num_nodes = g.num_nodes()

# Verificar que el número de nodos coincida con las características
item_features = torch.eye(num_nodes)

# Implementación del GCN con ponderación de aristas
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

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

# Inicializamos el GCN
gcn = GCN(in_feats=num_nodes, hidden_feats=8, out_feats=16)

# Pasamos el grafo y las características iniciales por el GCN
item_embeddings = gcn(g, item_features)
print("Representaciones de los ítems:\n", item_embeddings)

def predict_for_user(user_sequence, item_embeddings, top_k=2):
    """
    Predice los ítems más probables para un usuario basado en su secuencia de interacciones.
    
    Args:
        user_sequence: Lista de ítems con los que el usuario ha interactuado.
        item_embeddings: Tensor de representaciones de los ítems.
        top_k: Número de ítems a recomendar.
    
    Returns:
        Lista de los top_k ítems recomendados.
    """
    # Embeddings de los ítems con los que el usuario ha interactuado
    user_item_embeddings = item_embeddings[user_sequence]

    print(user_item_embeddings)
    
    # Calcular la similitud del coseno entre los ítems interactuados y todos los ítems
    #similarity_scores = torch.mm(user_item_embeddings, item_embeddings.t())
    similarity_scores={}
    for i in range(len(item_embeddings)):
        similarity_scores[i]=torch.cosine_similarity(user_item_embeddings,item_embeddings[i])
    
    recommended_items= sorted(similarity_scores,key=lambda x: similarity_scores[x],reverse=True)[1:]
    # Sumar las similitudes para obtener una puntuación total para cada ítem
    #total_scores = similarity_scores.sum(dim=0)
    
    # Excluir los ítems que el usuario ya ha interactuado
    #for item in user_sequence:
    #    total_scores[item] = -float('inf')
    
    # Obtener los top_k ítems con las puntuaciones más altas
    #_, recommended_items = torch.topk(total_scores, top_k)
    
    return recommended_items[:top_k]

# Supongamos que el usuario ha interactuado con los ítems 0 y 2
user_sequence = [3]

# Predecir los top 2 ítems recomendados
recommended_items = predict_for_user(user_sequence, item_embeddings, top_k=10)

print("Ítems recomendados para el usuario:", recommended_items)


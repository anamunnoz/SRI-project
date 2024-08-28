import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


matrix=[[3,1,2,3,3],[4,3,4,3,5],[3,3,1,5,4],[1,5,5,2,1]]

def ajustar_matriz(m):
    promedio=np.mean(m,axis=1)
    return m-promedio[:,np.newaxis]

def matriz_de_similitud(m):
    item_matrix= m.T
    matriz_similitud= cosine_similarity(item_matrix)
    return matriz_similitud

#m=ajustar_matriz(matrix)

#print(m)

print(cosine_similarity([[1,None,0]],[[1,0,0]]))


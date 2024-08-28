import numpy as np
import random


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
num_items = max(max(seq) for seq in sequences)

# Ejemplo en pseudocódigo
grafo = {}
for secuencia in sequences:
    for i in range(len(secuencia)):
        for j in range(i+1,len(secuencia)):
            nodo_origen = secuencia[i]
            nodo_destino = secuencia[j]
            if nodo_origen not in grafo:
                grafo[nodo_origen] = {}
            if nodo_destino not in grafo[nodo_origen]:
                grafo[nodo_origen][nodo_destino] = 0
            grafo[nodo_origen][nodo_destino] += 1/ (j-i+1)

# Normalización
for nodo_origen in grafo:
    total = sum(grafo[nodo_origen].values())
    for nodo_destino in grafo[nodo_origen]:
        grafo[nodo_origen][nodo_destino] /= total


def random_walk(grafo, nodo_inicial, pasos=10):
    nodo_actual = nodo_inicial
    recorrido = [nodo_actual]
    for _ in range(pasos):
        if nodo_actual not in grafo or len(grafo[nodo_actual]) == 0:
            break
        nodos_destino = list(grafo[nodo_actual].keys())
        probabilidades = list(grafo[nodo_actual].values())
        nodo_actual = random.choices(nodos_destino, probabilidades)[0]
        recorrido.append(nodo_actual)
    return recorrido


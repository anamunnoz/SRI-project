from recbole.quick_start import run_recbole

# Especificamos la configuración para cargar el dataset
#config_dict = {
#    'dataset': 'ml-100k',  # Dataset de Movielens 100k
#    'model': 'GRU4Rec',  # Un modelo secuencial básico usando GRU
#    'epochs': 10,  # Número de épocas para entrenar
#    'train_batch_size': 2048,  # Tamaño de batch para entrenamiento
#    'eval_batch_size': 4096,  # Tamaño de batch para evaluación
#    'neg_sampling': None,  # Usaremos muestras negativas para entrenamiento
#    'train_neg_sample_args': None 
#    
#}

# Ejecutamos el proceso de entrenamiento y evaluación
#run_recbole(model='GRU4Rec', config_dict=config_dict)


# Configuración básica para SASRec
config_dict = {
    'model': 'SASRec',               # Especificar el modelo SASRec
    'dataset': 'ml-100k',            # Usar el dataset Movielens 100k
    'train_batch_size': 128,         # Tamaño del batch para entrenamiento
    'embedding_size': 64,            # Tamaño de los embeddings
    'max_seq_length': 50,            # Longitud máxima de la secuencia
    'hidden_size': 64,               # Tamaño de la capa oculta en el modelo SASRec
    'num_heads': 4,                  # Número de cabezas en la atención
    'num_layers': 4,                 # Número de capas en el modelo
    'dropout_prob': 0.2,             # Probabilidad de dropout para regularización
    'epochs': 50,                    # Número de épocas para entrenar
    'learning_rate': 0.001,          # Tasa de aprendizaje
    'neg_sampling': 'uniform',       # Estrategia de muestreo negativo
    'loss_type': 'BPR'               # Tipo de pérdida: BPR (Bayesian Personalized Ranking)
}

# Ejecutar el modelo
run_recbole(model='SASRec', config_dict=config_dict)

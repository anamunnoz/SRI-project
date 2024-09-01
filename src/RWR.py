import numpy as np
import random
#from sklearn.metrics.pairwise import cosine_similarity
import copy
import pandas as pd


import csv

# Inicializa un diccionario vacío
place_city_dict = {}

# Abre el archivo CSV
with open('tourism_with_id.csv', mode='r', encoding='utf-8') as file:
    # Crea un lector de CSV
    csv_reader = csv.DictReader(file)
    
    # Itera sobre las filas del archivo CSV
    for row in csv_reader:
        # Extrae Place_id y city de la fila actual
        place_id = row['Place_Id']
        city = row['City']
        
        # Agrega el par Place_id: city al diccionario
        place_city_dict[place_id] = city




#matrix=[[5,3,4,None,None],[3,1,2,3,3],[4,3,4,5,5],[3,3,1,4,5],[1,5,5,1,2]]

# Paso 1: Leer el CSV
# Supongamos que el CSV se llama 'ratings.csv'
df = pd.read_csv('tourism_rating.csv')

# Paso 2: Identificar el número de usuarios y lugares únicos
num_users = df['User_Id'].max()
num_places = df['Place_Id'].max()

# Paso 3: Crear la matriz con valores None
matriz = [[None for _ in range(num_places)] for _ in range(num_users)]

# Paso 4: Llenar la matriz con los valores correspondientes
for index, row in df.iterrows():
    user_id = row['User_Id']
    place_id = row['Place_Id']
    place_rating = row['Place_Ratings']
    matriz[user_id-1][place_id-1] = place_rating



# La matriz está lista para ser utilizada
matrix = matriz



def calculate_V(m):
    V=[]
    for row in np.transpose(m):
        V.append(sum([1 for rating in row if rating is not None]))
    return V



def adjust_matrix(m):
    adjusted_matrix=[]
    for row in m:
        mean=np.mean([x for x in row if x is not None])
        adjusted_row= [value - mean  if value is not None  else None for value in row  ]
        adjusted_matrix.append(adjusted_row)
    return adjusted_matrix



def None_cosine_similarity(x:list,y:list,norm_x):
    X=[]
    Y=[]
    for i in range(len(x)):
        if(x[i] is not None and y[i] is not None):
            X.append(x[i])
            Y.append(y[i])
    if not X: return 0
    return np.dot(X,Y)/(norm_x*np.linalg.norm(Y))




def matriz_de_similitud(m):
    result=[]
    for i in range(len(m)):
        norm=np.linalg.norm([value for value in m[i] if value is not None])
        result.append([None_cosine_similarity(m[i],m[j],norm) if i!=j else 0 for j in range(len(m))])
    return result


def calculate_review(item,V,V_mean):
    return 1/2 + (V[item]-V_mean)/max(V)

def calculate_rating(initial_matrix,user, item):
    return 1/2 + (initial_matrix[user][item]-np.mean([rating for rating in initial_matrix[user] if rating is not None]))/5

def calculate_UI_matrix(initial_matrix, II_matrix):
    V=calculate_V(initial_matrix)
    v_mean=np.mean(V)
    result=np.zeros((len(initial_matrix),len(initial_matrix[0])))
    for j in range(len(initial_matrix[0])):
        item_review=calculate_review(j,V,v_mean)
        for i in range(len(initial_matrix)):
            II_matrix[i][j]+=item_review
            if(initial_matrix[i][j] is not None):
                result[i][j]=calculate_rating(initial_matrix,i,j)+item_review
    return result




#m_adjusted=adjust_matrix(matrix)


def build_probability_matrix(UU_matrix,II_matrix,UI_matrix):
    n=len(UU_matrix)
    m=len(II_matrix)
    probability_matrix=np.zeros((n+m,n+m))
    for i in range(max(n,m)):
        for j in range(max(n,m)):
            if(i<n and j<m):
                probability_matrix[i][j+n]=UI_matrix[i][j]
                probability_matrix[j+n][i]=UI_matrix[i][j]
            if(i<n and j<n):
                probability_matrix[i][j]=UU_matrix[i][j]
            if(i<m and j<m):
                probability_matrix[i+n][j+n]=II_matrix[i][j]
            
    return probability_matrix



def adjust_probability_matrix(m):
    for i in range(len(m)):
        sum=np.sum(m[i])
        for j in range(len(m[i])):
            m[i][j]/=sum
    return m


def build_graph(m):
    graph = {}
    for i in range(len(m)):
        for j in range(len(m)):
            if i not in graph:
                graph[i] = {}
            if j not in graph[i]:
                graph[i][j] = 0
            graph[i][j]=m[i][j]
    return graph


def random_walk_with_restart(graph,M,initial_node,c):
    e=np.zeros(len(M))
    e[initial_node]=1
    r=M[initial_node]
    actual_node=initial_node
    path=[]
    stop_condition=False

    while not stop_condition:
        #nodos_destino = list(graph[actual_node].keys())
        #probabilidades = list(graph[actual_node].values())
        nodos_destino=[i for i in range(len(M))]
        actual_node = random.choices(nodos_destino, r)[0]
        path.append(actual_node)
        r_next=c*np.matmul(M,r)+ np.multiply((1-c),e)
        stop_condition= np.linalg.norm(r_next-r)<0.0000000000000001
        r=r_next
    return path,r


def recommend_items(path,r,n):
    aux={}
    for i in range(len(r)):
        aux[i]=r[i]
    sorted_items=sorted(aux,key=lambda x: aux[x],reverse=True)
    return [x-n for x in sorted_items if x in path and x>=n]

def clean_recommended_items(recommendation, user_interactions):
    return [x for x in recommendation if user_interactions[x] is None]



############################################################################################
#m_adjusted=matrix




#UU_matrix= matriz_de_similitud(m_adjusted)
#II_matrix= np.transpose(matriz_de_similitud(np.transpose(m_adjusted)))
#UI_matrix=calculate_UI_matrix(m_adjusted,II_matrix)

#pm=build_probability_matrix(UU_matrix,II_matrix,UI_matrix)
#apm=adjust_probability_matrix(pm)


#g=build_graph(apm)

#path,r= random_walk_with_restart(g,apm,0,0.95)

#ri=recommend_items(path,r,len(UU_matrix))
#final_recommendation=clean_recommended_items(ri,m_adjusted[0])

#print(final_recommendation)
#########################################################################################

def metric_users(M):
    users=[]
    
    m=copy.deepcopy(M)
    for i in range(len(M)):
        for train in M[i][:392]:
            if train is not None:
                for test in M[i][392:]:
                    if test is not None:
                        if sum([1 for x in M[i][:392] if x is not None]) >4:
                            users.append(i)
                            break
                break
    users=users[:100]
    for user in users:
        for j in range(392,len(m[user])):
            m[user][j]=None
       
    return users, m
    
metric_result={}

users,test_matrix=metric_users(matrix)


UU_matrix= matriz_de_similitud(test_matrix)
II_matrix= np.transpose(matriz_de_similitud(np.transpose(test_matrix)))
UI_matrix=calculate_UI_matrix(test_matrix,II_matrix)
pm=build_probability_matrix(UU_matrix,II_matrix,UI_matrix)
apm=adjust_probability_matrix(pm)
g=build_graph(apm)

for user in users:
    path,r= random_walk_with_restart(g,apm,user,0.95)
    ri=recommend_items(path,r,len(UU_matrix))
    final_recommendation=clean_recommended_items(ri,test_matrix[user])
    final_recommendation=[x for x in final_recommendation if x>=392]
    
    metric_result[user]=final_recommendation

#print(metric_result)

def precision_n(M,n,initial_matrix):
    precisions=[]

    for user in M:   
        #print(M[user])     
        recuperated=set(M[user][:n])
        relevants=[]
        for i in range(392,len(initial_matrix[user])):
            if initial_matrix[user][i] is not None and initial_matrix[user][i]>=3:
                relevants.append(i)
        relevants=set(relevants)
        #print(user)
        #print(recuperated)
        #print(relevants)
        
        try:
            precisions.append(len(recuperated.intersection(relevants))/len(recuperated))
        except:
            precisions.append(0)
    
    return np.mean(precisions)


def recall(M,n,initial_matrix):
    precisions=[]

    for user in M:   
        #print(M[user])     
        recuperated=set(M[user][:n])
        relevants=[]
        for i in range(392,len(initial_matrix[user])):
            if initial_matrix[user][i] is not None and initial_matrix[user][i]>=3:
                relevants.append(i)
        relevants=set(relevants)
        #print(user)
        #print(recuperated)
        #print(relevants)
        
        try:
            precisions.append(len(recuperated.intersection(relevants))/len(relevants))
        except:
            precisions.append(0)
    
    return np.mean(precisions)

def HR(M,n,initial_matrix):
    precisions=[]

    u=0
    for user in M:   
        #print(M[user])     
        recuperated=set(M[user][:n])
        relevants=[]
        for i in range(392,len(initial_matrix[user])):
            if initial_matrix[user][i] is not None and initial_matrix[user][i]>=3:
                relevants.append(i)
        relevants=set(relevants)
        if len(recuperated.intersection(relevants)) >=1:
            u+=1
    
    return u/len(M)

def MRR(M,n,initial_matrix):
    precisions=[]

    u=0
    for user in M:   
        #print(M[user])     
        recuperated=M[user][:n]
        relevants=[]
        for i in range(392,len(initial_matrix[user])):
            if initial_matrix[user][i] is not None and initial_matrix[user][i]>=3:
                relevants.append(i)
        relevants=set(relevants)
        for i in range(len(recuperated)):
            if recuperated[i] in relevants:
                u+=1/(i+1)
                break

    return u/len(M)

def nDCG(M,n,initial_matrix):
    precisions=[]

    u=0
    for user in M:   
        #print(M[user])     
        recuperated=M[user][:n]
        relevants={}
        for i in range(392,len(initial_matrix[user])):
            if initial_matrix[user][i] is not None and initial_matrix[user][i]>=3:
                relevants[i]=initial_matrix[user][i]
        
        sorted_relevants=sorted(relevants,key=lambda x: relevants[x],reverse=True)
        idcg=0
        for i in range(len(sorted_relevants)):
            idcg+= (2**relevants[sorted_relevants[i]] -1)/ np.log2(i+2)
        dcg=0
        for i in range(n):
            if initial_matrix[user][recuperated[i]] is not None:
                dcg+= (2**initial_matrix[user][recuperated[i]] -1)/np.log2(i+2)

    return dcg/idcg


print(nDCG(metric_result,10,matrix))




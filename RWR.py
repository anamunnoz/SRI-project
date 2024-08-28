import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity


matrix=[[5,3,4,4,None],[3,1,2,3,3],[4,3,4,3,5],[3,3,1,5,4],[1,5,5,2,1]]

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
        result.append([None_cosine_similarity(m[i],m[j],norm) if i!=j else 1 for j in range(len(m))])
    return result


def calculate_review(item,V,V_mean):
    return 1/2 + (V[item]-V_mean)/max(V)

def calculate_rating(initial_matrix,user, item):
    return 1/2 + (initial_matrix[user][item]-np.mean(initial_matrix[user]))/5

def calculate_UI_matrix(initial_matrix, II_matrix):
    V=calculate_V(initial_matrix)
    v_mean=np.mean(V)
    result=np.zeros(len(initial_matrix),len(initial_matrix[0]))
    for j in range(len(initial_matrix[0])):
        item_review=calculate_review(j,V,v_mean)
        for i in range(len(initial_matrix)):
            II_matrix[i][j]+=item_review
            if(initial_matrix[i,j] is not None):
                result[i][j]=calculate_rating(initial_matrix,i,j)+item_review
    return result




#m_adjusted=adjust_matrix(matrix)
m_adjusted=matrix
UU_matrix= matriz_de_similitud(m_adjusted)
II_matrix= np.transpose(matriz_de_similitud(np.transpose(m_adjusted)))
UI_matrix=calculate_UI_matrix(m_adjusted,II_matrix)







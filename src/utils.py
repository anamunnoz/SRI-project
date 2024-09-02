
import numpy as np
import random

def calculate_V(m):
    '''
    Build a list with the number of reviews for all locations \n
    # **Parameters:**
    - **m:** matrix for user-location's reviews

    ### **Returns:** \n
    A list with the number of reviews for all locations
    '''
    V=[]
    for row in np.transpose(m):
        V.append(sum([1 for rating in row if rating is not None]))
    return V


def None_cosine_similarity(x:list,y:list,norm_x):
    '''
    Compute cosine similaruty for vectors thay may have None values \n
    # **Parameters:**
    - **x:** first vector
    - **y:** second vector
    - **norm_x:** first vector's norm for simplify 1 to many vectors computations

    ### **Returns:** \n
    Cosine similarity betweens vectors
    '''
    X=[]
    Y=[]
    for i in range(len(x)):
        if(x[i] is not None and y[i] is not None):
            X.append(x[i])
            Y.append(y[i])
    if not X: return 0
    return np.dot(X,Y)/(norm_x*np.linalg.norm(Y))




def similarity_matrix(m):
    '''
    For a matrix, compute the cosine similarity matrix for all row pairs   \n
    # **Parameters:**
    - **m:** matrix
    ### **Returns:** \n
    Cosine similarity matrix
    '''    
    result=[]
    for i in range(len(m)):
        norm=np.linalg.norm([value for value in m[i] if value is not None])
        result.append([None_cosine_similarity(m[i],m[j],norm) if i!=j else 0 for j in range(len(m))])
    return result


def calculate_review(item,V,V_mean):
    '''
    Calculate F-review for a location\n
    # **Parameters:**
    - **item:** objetive location
    - **V:** number of reviews for all locations
    - **V_mean:** mean of V
    ### **Returns:** \n
    F-review value
    '''        
    return 1/2 + (V[item]-V_mean)/max(V)

def calculate_rating(m,user, item):
    '''
    Calculate F-rating for a user-location pair \n
    # **Parameters:**
    - **m:** matrix for user-location's reviews
    - **user:** objetive user
    - **item:** objetive location
    ### **Returns:** \n
    F-rating value
    '''        
    return 1/2 + (m[user][item]-np.mean([rating for rating in m[user] if rating is not None]))/5

def calculate_UI_matrix(initial_matrix, II_matrix,feature_relations):
    '''
    Build a matrix for represent users-locations relationships. \n
    Also adjust relations between locations. \n
    # **Parameters:**
    - **initial_matrix:** matrix for user-location's reviews
    - **II_matrix:** matrix that represents location-locations relationships
    - **feature_relations:** matrix for user-location's relationships based on user preferences and locations features
    ### **Returns:** \n
    matrix for users-locations relationships
    '''        

    V=calculate_V(initial_matrix)
    v_mean=np.mean(V)
    result=np.zeros((len(initial_matrix),len(initial_matrix[0])))
    for j in range(len(initial_matrix[0])):
        item_review=calculate_review(j,V,v_mean)
        for i in range(len(initial_matrix)):
            II_matrix[i][j]+=item_review
            if(initial_matrix[i][j] is not None):
                result[i][j]=calculate_rating(initial_matrix,i,j)+item_review+ feature_relations[i][j]
    return result


def build_probability_matrix(UU_matrix,II_matrix,UI_matrix):
    '''
    Build a matrix for represent nodes relationships. Nodes represents both users and locations \n
    # **Parameters:**
    - **UU_matrix:** matrix that represents user-user relationships
    - **II_matrix:** matrix that represents location-location relationships
    - **UI_matrix:** matrix that represents user-location relationships
    ### **Returns:** \n
    Matrix that represent nodes relationships.
    ###### Still needs to be adjusted to represent probabilities
    '''     
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
    '''
    Adjust a matrix for represent probabilities \n
    # **Parameters:**
    - **m:** objective matrix
    ### **Returns:** \n
    adjusted matrix
    '''     
    for i in range(len(m)):
        sum=np.sum(m[i])
        for j in range(len(m[i])):
            m[i][j]/=sum
    return m


def random_walk_with_restart(M,initial_node,c):
    '''
    Random walk for explore the nodes to find locations \n
    # **Parameters:**
    - **M:** probabilities matrix for transitions between nodes
    - **initial_node:** initial node for walk
    - **c:** 1 - probability of restart walk in the initial node
    ### **Returns:** \n
    - **path:** sequence of nodes for represent a walk
    - **r:** vector r, represents probablities for finish the walk after a sequence of iterations
    '''     
    e=np.zeros(len(M))
    e[initial_node]=1
    r=M[initial_node]
    actual_node=initial_node
    path=[]
    stop_condition=False

    while not stop_condition:
        nodos_destino=[i for i in range(len(M))]
        actual_node = random.choices(nodos_destino, r)[0]
        path.append(actual_node)
        r_next=c*np.matmul(M,r)+ np.multiply((1-c),e)
        stop_condition= np.linalg.norm(r_next-r)<0.0000000000000001
        r=r_next
    return path,r


def recommend_items(path,r,n):
    '''
    Sort nodes in a path according to nodes relevance \n
    # **Parameters:**
    - **path:** path
    - **r:** nodes relevance
    - **n:** count of users
    ### **Returns:** \n
    sorted nodes
    '''   
    aux={}
    for i in range(len(r)):
        aux[i]=r[i]
    sorted_items=sorted(aux,key=lambda x: aux[x],reverse=True)
    return [x-n for x in sorted_items if x in path and x>=n]

def clean_recommended_items(recommendation, user_interactions):
    '''
    Adjust the recommendation for remove locations visited by the user\n
    # **Parameters:**
    - **recommendation:** locations recommendateds for the user
    - **user_interactions:** locations visited by the user
    ### **Returns:** \n
    Final recommendation
    ''' 
    return [x for x in recommendation if user_interactions[x] is None]


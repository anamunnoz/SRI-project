import numpy as np


def precision_n(M,N,initial_matrix):
    '''
    Compute mean precision@N for a group of users recommendations \n
    # **Parameters:**
    - **M:** users and recommendations for them
    - **N:** Parameter N
    - **initial_matrix:** matrix with rating of users to evaluate precision

    ### **Returns:** \n
    mean precision@N
    '''
    precisions=[]

    for user in M:     
        recuperated=set(M[user][:N])
        relevants=[]
        for i in range(392,len(initial_matrix[user])):
            if initial_matrix[user][i] is not None and initial_matrix[user][i]>=3:
                relevants.append(i)
        relevants=set(relevants)        
        try:
            precisions.append(len(recuperated.intersection(relevants))/len(recuperated))
        except:
            precisions.append(0)
    
    return np.mean(precisions)


def recall(M,n,initial_matrix):
    '''
    Compute mean reacll@N for a group of users recommendations \n
    # **Parameters:**
    - **M:** users and recommendations for them
    - **N:** Parameter N
    - **initial_matrix:** matrix with rating of users to evaluate precision

    ### **Returns:** \n
    mean recall@N
    '''
    recalls=[]

    for user in M:   
        recuperated=set(M[user][:n])
        relevants=[]
        for i in range(392,len(initial_matrix[user])):
            if initial_matrix[user][i] is not None and initial_matrix[user][i]>=3:
                relevants.append(i)
        relevants=set(relevants)
        
        try:
            recalls.append(len(recuperated.intersection(relevants))/len(relevants))
        except:
            recalls.append(0)
    
    return np.mean(recalls)

def HR(M,n,initial_matrix):
    '''
    Compute HR@N for a group of users recommendations \n
    # **Parameters:**
    - **M:** users and recommendations for them
    - **N:** Parameter N
    - **initial_matrix:** matrix with rating of users to evaluate precision

    ### **Returns:** \n
    Hit Ratio
    '''
    u=0
    for user in M:    
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
    '''
    Compute MRR@N for a group of users recommendations \n
    # **Parameters:**
    - **M:** users and recommendations for them
    - **N:** Parameter N
    - **initial_matrix:** matrix with rating of users to evaluate precision

    ### **Returns:** \n
    Mean Reciprocal Rank
    '''
    u=0
    for user in M:        
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
    '''
    Compute nDCG@10 for a group of users recommendations \n
    # **Parameters:**
    - **M:** users and recommendations for them
    - **N:** Parameter N
    - **initial_matrix:** matrix with rating of users to evaluate precision

    ### **Returns:** \n
    Normalized Discounted Cumulative Gain
    '''
    values=[]
    for user in M:      
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
        values.append(dcg/idcg)
    
    return np.mean(values)


from RWR_Datasets import place_dict,user_category_frecuency,user_category_price,user_category_rating,age_category_frecuency,age_category_price,age_category_rating,users_age
from RWR_Datasets import rating_matrix
from metrics import precision_n,recall,HR,MRR,nDCG
from Knowledge import Knowledge_based
from ThreeR_Model import ThreeR_Model

def run_pipeline(N):
    knowledge_system=Knowledge_based(len(rating_matrix),len(rating_matrix[0]))
    knowledge_recomendations=knowledge_system(place_dict,user_category_frecuency,user_category_rating,user_category_price,age_category_frecuency,age_category_rating,age_category_price)
    three_relations_model=ThreeR_Model(rating_matrix,knowledge_recomendations)
    metric_result=three_relations_model()

    metrics_list=[]
    metrics_list.append(precision_n(metric_result,N,rating_matrix))
    metrics_list.append(recall(metric_result,N,rating_matrix))
    metrics_list.append(HR(metric_result,N,rating_matrix))
    metrics_list.append(MRR(metric_result,N,rating_matrix))
    metrics_list.append(nDCG(metric_result,N,rating_matrix))

    return metrics_list

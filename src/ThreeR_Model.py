import numpy as np
from utils import similarity_matrix, calculate_UI_matrix, build_probability_matrix, random_walk_with_restart, recommend_items, clean_recommended_items,adjust_probability_matrix
from RWR_Datasets import metric_users

class ThreeR_Model:
    def __init__(self,rating_matrix,feature_relations):
        self.matrix=rating_matrix
        self.metric_result={}
        self.rank_result={}
        self.feature_relations=feature_relations


    def __call__(self):
        '''
        Do the recommendation for predefined users  \n
        ### **Returns:** \n
        dictionary of **user_id: recomendation_list**
        '''
        users,test_matrix=metric_users(self.matrix)
        UU_matrix= similarity_matrix(test_matrix)
        II_matrix= np.transpose(similarity_matrix(np.transpose(test_matrix)))
        UI_matrix=calculate_UI_matrix(test_matrix,II_matrix,self.feature_relations)
        probability_matrix=build_probability_matrix(UU_matrix,II_matrix,UI_matrix)
        adjusted_probability_matrix=adjust_probability_matrix(probability_matrix)


        for user in users:
            path,r= random_walk_with_restart(adjusted_probability_matrix,user,0.98)
            self.rank_result[user]=r
            ri=recommend_items(path,r,len(UU_matrix))
            final_recommendation=clean_recommended_items(ri,test_matrix[user])
            final_recommendation=[x for x in final_recommendation if x>=392]
            
            self.metric_result[user]=final_recommendation
        return self.metric_result
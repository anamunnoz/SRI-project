import numpy as np
from RWR_Datasets import calculate_age_range
from RWR_Datasets import users_age

class Knowledge_based:
    def __init__(self,n,m):
        self.n=n
        self.m=m
        self.recommendations=np.zeros((n,m))
    
    def __call__(self,place_dict,user_category_frecuency,user_category_rating,user_category_price,age_category_frecuency,age_category_rating,age_category_price):
        '''
        Compute a rate for user-locations relations based on locations features like category and price, and users and comunity preferences  \n
        # **Parameters:**
        - **place_dict:** dictionary of **location_id: [category,price]**
        - **user_category_frecuency:** matrix that represents how much an user visits attractions of a category
        - **user_category_rating:** matrix that represents how much an user rates attractions of a category at mean
        - **user_category_price:** matrix that represents how much an user pays for attractions of a category at mean
        - **age_category_frecuency:** matrix that represents how much users in an age range visits attractions of a category
        - **age_category_rating:** matrix that represents how much users in an age range rates attractions of a category
        - **age_category_price:** matrix that represents how much users in an age range pays for attractions of a category
        ### **Returns:** \n
        matrix that represent an initial relavance of location to users
        '''
        for i in range(self.n):
            for j in range(self.m):
                self.recommendations[i][j]=1
                place_category=place_dict[j][0]
                self.recommendations[i][j]*=user_category_frecuency[i][place_category]*user_category_rating[i][place_category]
                price_diference=np.abs(user_category_price[i][place_category]-place_dict[j][1])
                if price_diference==0:
                    price_diference=1
                self.recommendations[i][j]*= 1/price_diference

                self.recommendations[i][j]*=age_category_frecuency[calculate_age_range(users_age[i])][place_category]*age_category_rating[calculate_age_range(users_age[i])][place_category]
                price_diference_age=np.abs(age_category_price[calculate_age_range(users_age[i])][place_category]-place_dict[j][1])
                if price_diference_age==0:
                    price_diference_age=1
                self.recommendations[i][j]*= 1/price_diference_age
        return self.recommendations

    

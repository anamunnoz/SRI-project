import csv
import pandas as pd
import numpy as np
import copy
from pathlib import Path


def calculate_age_range(age):
    '''
    Calculate age range for an age\n
    # **Parameters:**
    - **age:** age of user
    ### **Returns:** \n
    id of age range
    ''' 
    if age < 20:
        return 0  
    return (age - 20) // 5 + 1


def get_users_age(csv_name):
    '''
    Get users age\n
    # **Parameters:**
    - **csv:** csv's name
    ### **Returns:** \n
    age for all users
    ''' 
    df = pd.read_csv(csv_name)
    return df['Age'].tolist()


def create_category_dict(csv_name):
    '''
    Get a dictionary for relations between categories and categories ids\n
    # **Parameters:**
    - **csv:** csv's name
    ### **Returns:** \n
    dictionary of **category_name: id_category**
    ''' 
    categories_set = set()


    with open(csv_name, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            categories_set.add(row['Category'])

    category_dict = {category: i  for i, category in enumerate(sorted(categories_set))}
    return category_dict

category_dict=create_category_dict('../datasets/tourism_with_id.csv')

def build_rating_matrix(csv_name):
    '''
    Build a matrix for represent users-locations rating\n
    # **Parameters:**
    - **csv:** csv's name
    ### **Returns:** \n
    matrix for represent users-locations rating
    ''' 

    df = pd.read_csv(csv_name)

    num_users = df['User_Id'].max()
    num_places = df['Place_Id'].max()

    matriz = [[None for _ in range(num_places)] for _ in range(num_users)]

    for _, row in df.iterrows():
        user_id = row['User_Id']
        place_id = row['Place_Id']
        place_rating = row['Place_Ratings']
        matriz[user_id-1][place_id-1] = place_rating
    return matriz

rating_matrix=build_rating_matrix('../datasets/tourism_rating.csv')
    

def build_place_category_price(csv_name):
    '''
    Extract relations between locations categories and prices\n
    # **Parameters:**
    - **csv:** csv's name
    ### **Returns:** \n
    dictionary of **location_id: [category,price]**
    ''' 
    place_dict = {}

    with open(csv_name, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            place_dict[int(row['Place_Id'])-1] = [category_dict[row['Category']], int(row['Price'])]

    return place_dict
place_dict=build_place_category_price('../datasets/tourism_with_id.csv')

user_category_frecuency=np.zeros((len(rating_matrix),len(category_dict)))
'''matrix that represents how much an user visits attractions of a category'''

user_category_price=np.zeros((len(rating_matrix),len(category_dict)))
'''matrix that represents how much an user pays for attractions of a category at mean'''

user_category_rating=np.zeros((len(rating_matrix),len(category_dict)))
'''matrix that represents how much an user rates attractions of a category at mean'''
user_category_count=0

users_age=get_users_age('../datasets/user.csv')
max_age=max(users_age)
age_category_frecuency=np.zeros((calculate_age_range(max_age)+1,len(category_dict)))
'''matrix that represents how much users in an age range visits attractions of a category'''

age_category_price=np.zeros((calculate_age_range(max_age)+1,len(category_dict)))
'''matrix that represents how much users in an age range pays for attractions of a category'''

age_category_rating=np.zeros((calculate_age_range(max_age)+1,len(category_dict)))
'''matrix that represents how much users in an age range rates attractions of a category'''
age_category_count=np.zeros((calculate_age_range(max_age)+1,len(category_dict)))


#Calculate those values
for i in range(len(rating_matrix)):
    for j in range(len(rating_matrix[i])):
        if rating_matrix[i][j] is not None:
            user_category_frecuency[i][place_dict[j][0]]+=1
            user_category_price[i][place_dict[j][0]]+=place_dict[j][1]
            user_category_rating[i][place_dict[j][0]]+=rating_matrix[i][j]
            user_category_count+=1

            age_category_frecuency[calculate_age_range(users_age[i])][place_dict[j][0]]+=1
            age_category_price[calculate_age_range(users_age[i])][place_dict[j][0]]+=place_dict[j][1]
            age_category_rating[calculate_age_range(users_age[i])][place_dict[j][0]]+=rating_matrix[i][j]
            age_category_count[calculate_age_range(users_age[i])][place_dict[j][0]]+=1
    
    

    for j in range(len(rating_matrix[i])):
        if rating_matrix[i][j] is not None:
            user_category_frecuency[i][place_dict[j][0]]/=user_category_count
            user_category_price[i][place_dict[j][0]]/=user_category_count
            user_category_rating[i][place_dict[j][0]]/=user_category_count

for i in range(calculate_age_range(max_age)+1):
    count=sum(age_category_count[i])
    for j in range(len(category_dict)):
        age_category_frecuency[i][place_dict[j][0]]/=count
        age_category_price[i][place_dict[j][0]]/=count
        age_category_rating[i][place_dict[j][0]]/=count



def metric_users(M):
    '''
    Select users to evaluate metrics  \n
    # **Parameters:**
    - **M:** ratings matrix
    ### **Returns:** \n
    - **users:** users for metrics
    - **m:** matrix for text
    ''' 
    users=[]
    
    m=copy.deepcopy(M)
    for i in range(len(M)):
        for train in M[i][:392]:
            if train is not None:
                for test in M[i][392:]:
                    if test is not None and test>=3:
                        if sum([1 for x in M[i][:392] if x is not None]) >4:
                            users.append(i)
                            break
                break
    users=users[:100]
    for user in users:
        for j in range(392,len(m[user])):
            m[user][j]=None
       
    return users, m
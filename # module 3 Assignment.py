import pandas as pd
import numpy as np


socioeconomic_data = pd.read_csv('socioeconomic_data.csv')
community_area_names = pd.read_csv('community_area_names.csv')

merged_data = pd.merge(socioeconomic_data, community_area_names, left_on="community_area", right_on="community_area_name")

features = ['income', 'unemployment_rate', 'hardship_index']
X = merged_data[features].to_numpy()

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

distance_matrix = np.zeros((X.shape[0], X.shape[0]))

for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        distance_matrix[i, j] = euclidean_distance(X[i], X[j])

distance_df = pd.DataFrame(distance_matrix, index=merged_data['community_area_name'], columns=merged_data['community_area_name'])

def get_top_similar_neighborhoods(query_name, distance_matrix, top_n=10):
    similar_neighborhoods = distance_matrix[query_name].sort_values()[1:top_n+1] 
    return similar_neighborhoods

query_neighborhoods = ['Rogers Park', 'West Ridge', 'Uptown']

for query in query_neighborhoods:
    print(f"Top 10 neighborhoods most similar to {query}:")
    print(get_top_similar_neighborhoods(query, distance_df))
    print("\n" + "-"*50 + "\n")

# module 1 assignment
import requests
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# API endpoint for Chicago crime data
url = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"



# filter for domestic violence incidents
params = {
    "$where": "domestic = true",
    "$limit": 5000  # Limiting the results for initial analysis
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()  
    df = pd.DataFrame(data)  
    print(df.head()) 
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")

df['date'] = pd.to_datetime(df['date'])
df.dropna(subset=['latitude', 'longitude'], inplace=True)
print(df.info())

community_area_df = pd.read_csv('community_area_names.csv')
community_area_names = dict(zip(community_area_df['community_area'], community_area_df['community_area_name']))

if 'community_area' not in df.columns:
    print("Community area column not found. Please check the dataset for proper column names.")
else:
    df['community_area'] = pd.to_numeric(df['community_area'], errors='coerce')  
    df.dropna(subset=['community_area'], inplace=True) 
    df['community_area_name'] = df['community_area'].map(community_area_names)  

if df['community_area_name'].isnull().sum() > 0:
    print("Warning: Some community areas were not mapped correctly.")
    print(df[df['community_area_name'].isnull()].head())  # Print unmapped rows for debugging
else:
    print("All community areas mapped successfully.")

# Bar graph showing incidents by neighborhood
incidents_by_neighborhood = df['community_area_name'].value_counts().head(10)

# Plotting the bar graph
plt.figure(figsize=(10,12))
incidents_by_neighborhood.plot(kind='bar', color='skyblue')
plt.title('Top 10 Neighborhoods in Chicago with the Highest Domestic Violence Incidents')
plt.xlabel('Community Area')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=30)
plt.show()

# build network
community_data = df.groupby('community_area_name').size().reset_index(name='incident_count')
G = nx.Graph()

for index, row in community_data.iterrows():
    G.add_node(row['community_area_name'], size=row['incident_count'])

for i in range(len(community_data) - 1):
    G.add_edge(community_data.iloc[i]['community_area_name'], community_data.iloc[i + 1]['community_area_name'])

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Find top 3 important nodes by betweenness centrality
important_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
print("Top 3 Important Nodes by Betweenness Centrality: ", important_nodes)

# Visualizing the Network
plt.figure(figsize=(12, 10)) 
pos = nx.spring_layout(G, k=0.3, seed=42)  
node_size = [G.nodes[node]['size'] * 10 for node in G.nodes]  
nx.draw_networkx(G, pos, with_labels=True, node_size=node_size, node_color='skyblue', edge_color='gray', font_size=10)

plt.title('Domestic Violence Network of Chicago Neighborhoods (Spring Layout)')
plt.show()
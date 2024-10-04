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

# Bar graph showing incidents by neighborhood
incidents_by_neighborhood = df['community_area'].value_counts().head(10)

# Plotting the bar graph
plt.figure(figsize=(10, 6))
incidents_by_neighborhood.plot(kind='bar', color='skyblue')
plt.title('Top 10 Neighborhoods with the Highest Domestic Violence Incidents')
plt.xlabel('Community Area')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.show()

# build network
community_data = df.groupby('community_area').size().reset_index(name='incident_count')
G = nx.Graph()

for index, row in community_data.iterrows():
    G.add_node(row['community_area'], size=row['incident_count'])

for i in range(len(community_data) - 1):
    G.add_edge(community_data.iloc[i]['community_area'], community_data.iloc[i + 1]['community_area'])

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Find top 3 important nodes by betweenness centrality
important_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
print("Top 3 Important Nodes by Betweenness Centrality: ", important_nodes)

# Visualizing the Network
plt.figure(figsize=(10, 8))
node_size = [G.nodes[node]['size'] * 10 for node in G.nodes]  # Size nodes based on incident counts
nx.draw_networkx(G, with_labels=True, node_size=node_size, node_color='skyblue', edge_color='gray', font_size=10)
plt.title('Domestic Violence Network of Chicago Neighborhoods')
plt.show()
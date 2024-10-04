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

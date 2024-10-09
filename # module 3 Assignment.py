# module 3 Assignment

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import networkx as nx

url = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

params = {
    "$where": "domestic = true",
    "$limit": 5000  
}

response = requests.get(url, params=params)
data = response.json()
dv_data = pd.DataFrame(data)

dv_data['community_area'] = pd.to_numeric(dv_data['community_area'], errors='coerce')
dv_data.dropna(subset=['community_area'], inplace=True)
socioeconomic_data = pd.read_csv('socioeconomic_data.csv')
socioeconomic_data['community_area'] = pd.to_numeric(socioeconomic_data['community_area'], errors='coerce')
incident_counts = dv_data.groupby('community_area').size().reset_index(name='incident_count')

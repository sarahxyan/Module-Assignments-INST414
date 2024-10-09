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

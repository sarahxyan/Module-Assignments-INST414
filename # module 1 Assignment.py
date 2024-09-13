# module 1 assignment
import requests
import pandas as pd

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

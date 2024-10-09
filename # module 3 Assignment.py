# module 3 Assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dv_data = pd.read_csv('community_area_names.csv')
socioeconomic_data = pd.read_csv('socioeconomic_data.csv')

dv_data['community_area'] = dv_data['community_area'].astype(str)
socioeconomic_data['community_area'] = socioeconomic_data['community_area'].astype(str)

merged_df = dv_data.merge(socioeconomic_data, how='left', on='community_area')
